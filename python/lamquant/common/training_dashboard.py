"""
LamQuant Training Dashboard — Standard progress output for all training scripts.

Every training script imports this module and calls emit() once per N batches.
The cockpit parses the @DASH: prefix to render the 8-line fixed dashboard.
Scripts that don't use the cockpit still get readable single-line output.

Usage:
    from training_dashboard import TrainingDashboard

    dash = TrainingDashboard(
        model_name='Student Subband',
        gen='7.5',
        preset='production',
        total_epochs=550,
        device=device,
    )

    # In training loop:
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    dash.step(
        phase='QAT',       # Warmup / QAT / Fine / Align / Recover
        epoch=147,
        batch=1204,
        n_batches=2320,
        loss=0.0423,
        loss_parts={'mse': 0.031, 'R': 0.008, 'spec': 0.003},
        r=0.891,
        grad_norm=grad_norm,
    )

    # After validation:
    dash.update_val(val_r=0.891, best_r=0.894)

    # After checkpoint save:
    dash.saved_checkpoint('student_E147_R0.891.ckpt', enc_kb=40.4)
"""

import sys
import time
import torch
import subprocess


class TrainingDashboard:
    """Standard training progress emitter for all LamQuant training scripts."""

    def __init__(self, model_name='Model', gen='7.5', preset='custom',
                 total_epochs=100, device=None, emit_interval=50):
        self.model_name = model_name
        self.gen = gen
        self.preset = preset
        self.total_epochs = total_epochs
        self.device = device or torch.device('cpu')
        self.emit_interval = emit_interval

        self.start_time = time.time()
        self._batch_count = 0
        self._batch_timer = time.time()
        self._val_r = 0.0
        self._best_r = 0.0
        self._last_grad = 0.0
        self._last_sparse = 0.0
        self._last_alpha = 0.0
        self._last_tau = 0.0
        self._sparse_counter = 0

    def update_val(self, val_r, best_r=None):
        """Call after each validation pass."""
        self._val_r = val_r
        if best_r is not None:
            self._best_r = best_r
        elif val_r > self._best_r:
            self._best_r = val_r

    def update_ternary_stats(self, model, tau=None):
        """Update sparsity, alpha, tau from a ternary model. Call periodically.
        Also logs per-layer sparsity for Phase 3 diagnostics."""
        target = getattr(model, '_orig_mod', model)
        total = zero = 0
        alphas = []
        per_layer = {}
        for name, m in target.named_modules():
            if hasattr(m, 'lsq_alpha') and hasattr(m, 'weight'):
                w = m.weight.detach()
                alpha = torch.abs(m.lsq_alpha).detach()
                w_q = torch.round(torch.clamp(w / (alpha + 1e-8), -1, 1))
                n = w_q.numel()
                z = (w_q == 0).sum().item()
                total += n
                zero += z
                alphas.append(alpha.mean().item())
                per_layer[name] = {
                    'sparsity': 100.0 * z / max(n, 1),
                    'n_weights': n,
                    'alpha': alpha.mean().item(),
                }
        self._last_sparse = 100.0 * zero / max(total, 1)
        self._last_alpha = sum(alphas) / max(len(alphas), 1)
        self._per_layer_sparse = per_layer
        if tau is not None:
            self._last_tau = tau
        else:
            for m in target.modules():
                if hasattr(m, 'deadzone_tau'):
                    self._last_tau = m.deadzone_tau
                    break

    def log_per_layer_sparsity(self):
        """Print per-layer sparsity breakdown. Call at validation time."""
        if not hasattr(self, '_per_layer_sparse') or not self._per_layer_sparse:
            return
        print("  [Sparsity per layer]")
        for name, stats in sorted(self._per_layer_sparse.items(),
                                   key=lambda x: -x[1]['n_weights']):
            sp = stats['sparsity']
            n = stats['n_weights']
            a = stats['alpha']
            bar = '█' * int(sp / 5) + '░' * (20 - int(sp / 5))
            flag = ""
            if sp < 20 and n > 1000:
                flag = " ← LOW"
            print(f"    {name:<25s} {n:>7,}w  {sp:>5.1f}% {bar} α={a:.3f}{flag}")

    def saved_checkpoint(self, path, enc_kb=None):
        """Emit checkpoint save notification."""
        parts = [f"@SAVE: {path}"]
        if enc_kb is not None:
            parts.append(f"enc={enc_kb:.1f}KB")
        parts.append(f"R={self._best_r:.4f}")
        print(" ".join(parts), flush=True)

    def step(self, phase, epoch, batch, n_batches, loss, r=None,
             loss_parts=None, grad_norm=None, extra=None):
        """Emit progress. Call every batch; internally throttles to emit_interval."""
        if batch % self.emit_interval != 0 and batch != n_batches - 1:
            return

        self._batch_count += 1
        now = time.time()
        elapsed = now - self.start_time

        # Throughput
        bps = self.emit_interval / max(now - self._batch_timer, 0.001) if self._batch_count > 1 else 0
        self._batch_timer = now

        # ETA
        total_batches = self.total_epochs * n_batches
        done_batches = (epoch - 1) * n_batches + batch + 1
        if done_batches > 0:
            remaining = elapsed / done_batches * (total_batches - done_batches)
        else:
            remaining = 0
        eta_h, eta_rem = divmod(int(remaining), 3600)
        eta_m = eta_rem // 60

        if grad_norm is not None:
            self._last_grad = float(grad_norm)

        # Progress bar
        pct = (batch + 1) / max(n_batches, 1)
        filled = int(pct * 20)
        bar = '█' * filled + '▒' + '░' * max(0, 19 - filled)

        # Loss parts string
        lp = ""
        if loss_parts:
            lp = " (" + " ".join(f"{k}:{v:.3f}" for k, v in loss_parts.items()) + ")"

        # GPU stats (cached, not every call)
        gpu_pct = gpu_temp = 0
        gpu_mem = 0.0
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            if self._batch_count % 10 == 1:  # query nvidia-smi every 10 emissions
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                         '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=1)
                    vals = result.stdout.strip().split(',')
                    gpu_pct = int(vals[0].strip())
                    gpu_temp = int(vals[1].strip())
                    self._gpu_pct = gpu_pct
                    self._gpu_temp = gpu_temp
                except Exception:
                    pass
            else:
                gpu_pct = getattr(self, '_gpu_pct', 0)
                gpu_temp = getattr(self, '_gpu_temp', 0)

        r_str = f"{r:.4f}" if r is not None else "—"
        extra_str = ""
        if extra:
            extra_str = " " + " ".join(f"{k}={v}" for k, v in extra.items())

        # Structured output line — cockpit parses fields after @DASH:
        print(f"@DASH: phase={phase} epoch={epoch}/{self.total_epochs} "
              f"batch={batch+1}/{n_batches} pct={pct:.3f} "
              f"loss={loss:.4f}{lp} R={r_str} valR={self._val_r:.4f} bestR={self._best_r:.4f} "
              f"grad={self._last_grad:.2f} sparse={self._last_sparse:.1f} "
              f"tau={self._last_tau:.3f} alpha={self._last_alpha:.3f} lr={self._lr()} "
              f"gpu={gpu_pct} vram={gpu_mem:.1f} temp={gpu_temp} "
              f"bps={bps:.0f} eta={eta_h:02d}h{eta_m:02d}m "
              f"preset={self.preset} bar={bar}"
              f"{extra_str}",
              flush=True)

    def _lr(self):
        """Placeholder — scripts should pass lr via extra or we read from optimizer."""
        return "—"


def parse_dash_line(line):
    """Parse a @DASH: line into a dict. Used by the cockpit dashboard renderer."""
    if not line.startswith('@DASH:'):
        return None
    data = {}
    rest = line[6:].strip()
    # Parse key=value pairs (values may contain parenthetical groups)
    import re
    # Match key=value where value is everything up to the next key= or end
    for m in re.finditer(r'(\w+)=([^\s]+(?:\s*\([^)]*\))?)', rest):
        data[m.group(1)] = m.group(2)
    return data


def render_dashboard_lines(data, model_name='Student', gen='7.5'):
    """Render the 8-line dashboard from parsed @DASH: data.

    Returns a list of 8 strings. Used by the cockpit's _render_progress_loop.
    """
    if not data:
        return [''] * 8

    preset = data.get('preset', '?')
    phase = data.get('phase', '?')
    epoch_str = data.get('epoch', '?/?')
    batch_str = data.get('batch', '?/?')
    bar = data.get('bar', '░' * 20)
    pct = float(data.get('pct', '0')) * 100

    loss = data.get('loss', '?')
    lp_raw = data.get('lp', '')
    r = data.get('R', '?')
    val_r = data.get('valR', '?')
    best_r = data.get('bestR', '?')

    grad = data.get('grad', '?')
    sparse = data.get('sparse', '?')
    tau = data.get('tau', '?')
    alpha = data.get('alpha', '?')
    lr = data.get('lr', '?')

    gpu = data.get('gpu', '?')
    vram = data.get('vram', '?')
    temp = data.get('temp', '?')
    bps = data.get('bps', '?')
    eta = data.get('eta', '?')

    # Find loss parts in the raw line
    loss_detail = loss
    if '(' in data.get('loss', ''):
        loss_detail = data['loss']

    # Fixed-width box: 80 chars total (78 inner + 2 for ║ borders)
    W = 78
    title = f"LAMQUANT GEN {gen} — {model_name.upper()}"
    right = f"preset:{preset}"
    pad = max(1, W - 4 - len(title) - len(right))  # 4 = 2 leading + 2 trailing spaces
    header_inner = f"  {title}{' ' * pad}{right}  "[:W]

    # R display: show batch R (current), val R, best R
    r_display = f"R: {r}" if r != '—' and r != '0.0000' else "R: —"
    val_display = f"Val R: {val_r}" if val_r != '0.0000' else "Val R: —"
    best_display = f"Best R: {best_r}" if best_r != '0.0000' else "Best R: —"

    lines = [
        f"╔{'═' * W}╗",
        f"║{header_inner}║",
        f"╚{'═' * W}╝",
        f"  {phase} │ Epoch {epoch_str} │ Batch {batch_str} {bar} {pct:.0f}%",
        f"  Loss: {loss_detail} │ {r_display} │ {val_display} │ {best_display}",
        f"  Grad: {grad} │ Sparsity: {sparse}% │ τ: {tau} │ ᾱ: {alpha} │ LR: {lr}",
        f"  GPU: {gpu}% │ VRAM: {vram}/24G │ {temp}°C │ {bps} it/s │ ETA: {eta}",
        f"{'─' * (W + 2)}",
    ]
    return lines
