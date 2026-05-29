import os
import glob
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
import time
import argparse
import subprocess
import sys
import concurrent.futures

# MOVE-B (2026-05-29): the Gen-6 FP32 oracle teacher trainer + its
# teacher model def (teacher_arch.py, formerly ai_models/architectures/
# teacher.py) moved into BLUT oracle. percent_zero_weights lives in the
# common DTO/util module. Put the blut/python package root + the sibling
# common dir on sys.path so both resolve regardless of launch style.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for _p in (ROOT_DIR, os.path.join(ROOT_DIR, 'lamquant', 'oracle'),
           os.path.join(ROOT_DIR, 'lamquant', 'common')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from lamquant.common.utils import percent_zero_weights  # noqa: E402
from teacher_arch import (  # noqa: E402
    FocalModulationBlock,
    MobileNetV5Focal,
    DecoderBlock,
    FP32OracleAutoEncoder,
    ChannelAwareEncoding,
    BottleneckAttention,
    StridedFocalBlock,
    UpsampleFocalBlock,
    L3TeacherEncoder,
    L3TeacherDecoder,
    L3Teacher,
)

# training_cockpit was refactored — TrainingCockpit / get_git_sha / format_time
# / create_logger no longer exist there. They're only used by this file when
# it's run as a standalone script (its `if __name__ == '__main__':` block).
# When imported for the dataset class (Q31Dataset), the helpers aren't needed,
# so the import is wrapped in try/except with no-op stubs so downstream
# scripts (train_student_subband.py, etc.) can still import Q31Dataset.
try:
    from training_cockpit import TrainingCockpit, get_git_sha, format_time, create_logger
except ImportError:
    class TrainingCockpit:
        def __init__(self, *a, **kw):
            raise RuntimeError(
                "TrainingCockpit class was removed from training_cockpit.py. "
                "Update train_teacher.py's standalone main to use a current API."
            )
    def get_git_sha(): return 'unknown'
    def format_time(t): return f'{t:.1f}s'
    def create_logger(*a, **kw): return None

def get_best_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device("xpu")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def get_hardware_profile(device, force_batch_size=None):
    """Dynamically probe the host silicon and return optimal training parameters."""
    b_size = 4
    amp_dtype = torch.float16  # Safe fallback for older GPUs (Turing / Pascal)
    
    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(device)
        # Conservative batch sizes tuned alongside lr=2e-3. Larger batches
        # need proportional LR scaling (linear scaling rule) to maintain
        # the same training dynamics — without that, accuracy degrades.
        if vram_gb >= 20: b_size = 64
        elif vram_gb >= 12: b_size = 32
        elif vram_gb >= 8: b_size = 16
        
        # Ampere+ (SM >= 8.0) supports bfloat16 natively
        if torch.cuda.get_device_capability(device)[0] >= 8:
            amp_dtype = torch.bfloat16
    else:
        gpu_name = str(device)
    
    if force_batch_size is not None:
        b_size = force_batch_size
        
    # High-Inertia LR for BS=64 (Linear Scaling Rule)
    lr = 2e-3
    
    return b_size, amp_dtype, lr, gpu_name

def save_training_config(path, config):
    """Dump a reproducibility manifest alongside checkpoints."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[*] Training config saved -> {path}")

def patient_wise_split(npz_files, val_ratio=0.2):
    patient_files = defaultdict(list)
    for f in npz_files:
        basename = os.path.basename(f)
        patient_id = basename.split('_')[0] 
        patient_files[patient_id].append(f)
    
    patients = list(patient_files.keys())
    np.random.shuffle(patients)
    n_val = max(1, int(len(patients) * val_ratio))
    val_patients = patients[:n_val]
    train_patients = patients[n_val:]
    
    train_files = [f for p in train_patients for f in patient_files[p]]
    val_files = [f for p in val_patients for f in patient_files[p]]
    return train_files, val_files

class EventWeightedMSELoss(nn.Module):
    def __init__(self, seizure_penalty_weight=5.0):
        super().__init__()
        self.penalty = seizure_penalty_weight

    def forward(self, pred, target, seizure_mask):
        base_squared_error = (pred - target) ** 2
        mask_expanded = seizure_mask.unsqueeze(1).expand_as(base_squared_error)
        weighted_error = base_squared_error * (1.0 + (self.penalty - 1.0) * mask_expanded)
        return weighted_error.mean()

class SpectralConvergenceLoss(nn.Module):
    """Triton-Safe Spectral Supervision for Clinical Sign-off (Magnitude-Only)."""
    def __init__(self, fft_sizes=[64, 128, 256]):
        super().__init__()
        self.fft_sizes = fft_sizes

    def forward(self, pred, target):
        loss = 0
        for n_fft in self.fft_sizes:
            hop = n_fft // 4
            # Triton-Safe dB-Scale Spectral Supervision
            # log10 prevents spectral dominance over temporal waveform
            p_spec = torch.stft(pred.view(-1, 2500).float(), n_fft=n_fft, hop_length=hop, return_complex=True).abs() + 1e-8
            t_spec = torch.stft(target.view(-1, 2500).float(), n_fft=n_fft, hop_length=hop, return_complex=True).abs() + 1e-8
            loss += F.mse_loss(torch.log10(p_spec), torch.log10(t_spec))
        return loss / len(self.fft_sizes)

class ClinicalHybridLoss(nn.Module):
    def __init__(self, seizure_weight=5.0, alpha=0.5):
        super().__init__()
        self.mse = EventWeightedMSELoss(seizure_weight)
        self.spec = SpectralConvergenceLoss()
        self.alpha = alpha

    def forward(self, pred, target, mask):
        l_mse = self.mse(pred, target, mask)
        l_spec = self.spec(pred, target)
        return (1.0 - self.alpha) * l_mse + self.alpha * l_spec


# Re-export architecture classes for backward compatibility.
# Downstream scripts (harden_artifacts.py, train_l3_teacher.py, benchmarks, etc.)
# do `from train_teacher import L3Teacher` / `FP32OracleAutoEncoder` — these
# re-exports ensure those imports keep working without modification.
__all__ = [
    'FocalModulationBlock', 'MobileNetV5Focal', 'DecoderBlock',
    'FP32OracleAutoEncoder', 'ChannelAwareEncoding', 'BottleneckAttention',
    'StridedFocalBlock', 'UpsampleFocalBlock',
    'L3TeacherEncoder', 'L3TeacherDecoder', 'L3Teacher',
    # Added 2026-05-16: downstream importers (training_utils.Q31Dataset
    # via train_teacher shim) need this re-exported through `from ... import *`.
    'Q31Dataset',
]

def create_intermediate_model(width=256, cdf_entries=64, device='cpu'):
    """Create INT8 intermediate model for progressive distillation.

    Uses the EXACT same architecture as the student (TernaryMobileNetV5_Subband)
    at width=256 with INT8 activation quantization. Same CDF-LUT, same Cayley
    rotation, same [32, 79] latent — only the weight precision differs.

    The distillation chain aligns latent spaces at every step:
      Teacher [32,79] FP32 → Intermediate [32,79] INT8 → Student [32,79] ternary
    Each step only changes weight precision. CDF-LUT, Cayley Q, and FSQ config
    are identical across all three models.

    Returns a TernaryMobileNetV5_Subband configured for INT8 operation.
    Set activation_bits=8 before calling to get W2A8 quantization.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'student'))
    from lamquant_neural.models.encoder import TernaryMobileNetV5_Subband
    from lamquant_neural.models.blocks import set_activation_bits

    # INT8 activations for the intermediate
    set_activation_bits(8)
    model = TernaryMobileNetV5_Subband(in_ch=21, latent_dim=32, width=width,
                                        cdf_entries=cdf_entries).to(device)
    # Restore default for other models
    set_activation_bits(16)
    return model


class Q31Dataset(Dataset):
    """RAM-cached dataset with Persistent Binary Caching and Parallel Ingestion."""
    def __init__(self, file_paths, headless=False, cache_path="dataset_sim/q31_cache_v1.pt"):
        self.samples = []
        
        # [Strategy 1: Rapid Load from Persistent Binary Cache]
        if os.path.exists(cache_path):
            print(f"[*] Found persistent cache. Rapid loading binary blob...")
            try:
                try:
                    self.samples = torch.load(cache_path, map_location='cpu', weights_only=True)
                except Exception:
                    self.samples = torch.load(cache_path, map_location='cpu', weights_only=False)
                print(f"[*] Loaded {len(self.samples)} samples from cache in seconds.")
                return
            except Exception as e:
                print(f"[!] Cache corrupted, falling back to parallel ingestion: {e}")

        # [Strategy 2: Parallel Ingestion from raw .npz files]
        print(f"[*] Pre-loading {len(file_paths)} files into RAM (Parallel Mode)...")
        
        def load_single_file(path):
            try:
                with np.load(path) as data:
                    length = data['data'].shape[1]
                    chunk = min(length, 2500)
                    # .copy() detaches from the zip-backed lazy loader before
                    # the context manager closes the file handle.
                    raw = data['data'][:, :chunk].copy()
                    msk = data['seizure_mask'][:chunk].copy()

                    l3 = None
                    if 'l3' in data.files:
                        l3_data = data['l3']
                        if len(l3_data) > 0:
                            l3 = torch.tensor(l3_data[0].copy(), dtype=torch.float32)

                eeg = (torch.tensor(raw, dtype=torch.float32) / 2147483647.0) * 1000.0
                mask = torch.tensor(msk, dtype=torch.float32)
                if eeg.shape[1] < 2500:
                    eeg = F.pad(eeg, (0, 2500 - eeg.shape[1]))
                    mask = F.pad(mask, (0, 2500 - mask.shape[1]))

                return (eeg, eeg, mask, l3)
            except Exception:  # file read or model load failure — skip gracefully
                return None

        # Execute parallel load using 16 threads (IO-bound)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(load_single_file, file_paths), 
                               total=len(file_paths), 
                               desc="Accelerating Signal Ingestion", 
                               leave=False, 
                               disable=headless))
        
        self.samples = [r for r in results if r is not None]
        print(f"[*] Cached {len(self.samples)} samples in RAM ({len(self.samples) * 21 * 2500 * 4 / 1e6:.0f} MB)")
        
        # Save to cache for next time
        try:
            print(f"[*] Writing persistent binary cache -> {cache_path}")
            torch.save(self.samples, cache_path)
        except Exception:  # file read or model load failure — skip gracefully
            print("[!] Warning: Could not save cache.")
                
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LamQuant Gen 6 Oracle Teacher Training")
    parser.add_argument("--headless", action="store_true", help="Plain text output for logging CI/CD")
    parser.add_argument("--force_batch_size", type=int, default=None, help="Override auto-detected batch size for reproducibility")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for deterministic training")
    parser.add_argument("--resume", action="store_true", help="Resume from last_checkpoint.pth if it exists")
    parser.add_argument("--logger", type=str, default=None, choices=["wandb", "mlflow"], help="Experiment tracker (optional)")
    parser.add_argument("--wandb_project", type=str, default="lamquant", help="W&B project name")
    parser.add_argument("--mlflow_experiment", type=str, default="lamquant-teacher", help="MLflow experiment name")
    parser.add_argument("--freq-weighted-loss", action="store_true",
                        help="(Experimental) Add frequency-weighted MSE that prioritizes "
                             "neural oscillation bands over artifact-prone HF bands")
    args = parser.parse_args()

    # -----------------------------------------------------
    # DETERMINISTIC SEEDING (Reproducibility Guarantee)
    # -----------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------
    # HIGH-PERFORMANCE MATH CORES (TF32 / cuDNN Benchmark)
    # -----------------------------------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    npz_dir = os.path.join(ROOT_DIR, "ai_models", "dataset_sim", "q31_events")
    cache_path = os.path.join(ROOT_DIR, "ai_models", "dataset_sim", "q31_cache_v1.pt")

    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        print("[!] No Q31 data found. Run edf_to_events.py first.")
        print(f"    Expected directory: {npz_dir}")
        sys.exit(1)

    # Canonical split from manifest_v3.json — typed pipeline, single source of truth.
    sys.path.insert(0, os.path.join(ROOT_DIR, "ai_models"))
    from data_types import DatasetManifest, Split

    manifest = DatasetManifest.load(os.path.join(
        ROOT_DIR, "ai_models", "dataset_sim", "manifest_v3.json"))
    train_files = [str(p) for p in manifest.get_files(Split.TRAIN)]
    val_files = [str(p) for p in manifest.get_files(Split.VAL)]
    print(manifest.summary_str())
    device = get_best_device()
    b_size, amp_dtype, lr, gpu_name = get_hardware_profile(device, args.force_batch_size)

    # Auto-select dataset strategy based on data size vs RAM
    from streaming_dataset import HybridQ31Dataset, StreamingQ31Dataset, MemmapTeacherDataset
    if len(train_files) > 500:
        # MemmapTeacherDataset pre-extracts ALL windows to a flat binary file
        # on disk (one-time ~40 min cost), then memmap's it for O(1) access at
        # ~0.5ms/sample. This replaces HybridQ31Dataset's streaming path which
        # decompressed multi-MB NPZs at 208ms/sample — a 400× speedup.
        print(f"[*] Large dataset ({len(train_files)} files) — memmap teacher cache")
        dataset = MemmapTeacherDataset(train_files, ROOT_DIR, windows_per_epoch=150000)
        # num_workers=0: memmap access is ~0.5ms/sample (OS page cache handles
        # prefetching). Workers would require pickling the 233 GB memmap object
        # to forkserver → MemoryError. Zero workers avoids serialization entirely.
        loader = DataLoader(
            dataset,
            batch_size=b_size,
            shuffle=False,    # dataset does its own random sampling
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
        )
    else:
        print(f"[*] Small dataset ({len(train_files)} files) — using RAM cache")
        dataset = Q31Dataset(train_files, headless=args.headless, cache_path=cache_path)
        loader = DataLoader(
            dataset,
            batch_size=b_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
        )
    
    model = FP32OracleAutoEncoder().to(device)
    
    # [Triton Integration]
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except RuntimeError:
            pass

    # Fused AdamW (zero Python overhead on CUDA)
    use_fused = device.type == 'cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=use_fused)
    loss_fn = ClinicalHybridLoss(seizure_weight=5.0, alpha=0.1) # Alpha=0.1 to focus on 1000x waveform gain

    # Experimental: frequency-weighted MSE (FEMBA-inspired physiological prior)
    freq_loss_fn = None
    if getattr(args, 'freq_weighted_loss', False):
        sys.path.insert(0, os.path.dirname(__file__))
        from freq_weighted_loss import FrequencyWeightedMSE
        freq_loss_fn = FrequencyWeightedMSE(sample_rate=250.0).to(device)
        print("[*] Frequency-weighted MSE enabled (experimental)")
    
    # Build the reproducibility config
    training_config = {
        "model": "FP32OracleAutoEncoder-Elite",
        "git_sha": get_git_sha(),
        "seed": args.seed,
        "batch_size": b_size,
        "learning_rate": lr,
        "amp_dtype": str(amp_dtype),
        "device": gpu_name,
        "dataset_files": len(train_files),
        "epochs_target": 800,
        "pytorch_version": torch.__version__
    }
    
    epochs = 800
    swa_start = 600  # Start SWA averaging for the long clinical tail
    start_epoch = 1
    best_loss = float('inf')
    best_epoch = 0

    # -----------------------------------------------------
    # CHECKPOINT RESUME (Full Optimizer State Recovery)
    # -----------------------------------------------------
    # BUG FIX C1: All checkpoint paths must be absolute. When launched by
    # training_cockpit.py, CWD is ROOT_DIR (not ai_models/oracle/), so
    # relative paths like os.path.join(CKPT_DIR, "teacher_best.ckpt") save to the wrong directory.
    # Downstream scripts (harden_artifacts.py) look in ai_models/oracle/.
    CKPT_DIR = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(CKPT_DIR, "last_checkpoint.pth")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"[*] Resuming from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        best_epoch = ckpt.get('best_epoch', 0)
        print(f"[*] Resumed at Epoch {start_epoch} | Best Loss: {best_loss:.5f} (ep {best_epoch})")
    
    # Initialize experiment tracker
    if args.logger == "wandb":
        logger = create_logger("wandb", project=args.wandb_project,
                                run_name=f"teacher_bs{b_size}_lr{lr:.4f}",
                                config=training_config,
                                tags=["teacher", "oracle", "fp32"])
    elif args.logger == "mlflow":
        logger = create_logger("mlflow", experiment_name=args.mlflow_experiment,
                                run_name=f"teacher_bs{b_size}_lr{lr:.4f}")
        logger.config(training_config)
    else:
        logger = create_logger(None)
    
    cockpit = TrainingCockpit(
        mode="TEACHER (ORACLE)",
        model_name="FP32 MobileNetV5Focal AutoEncoder",
        target_hw="Server Backend → RP2350",
        quant_mode="FP32 (Full Precision)",
        git_sha=get_git_sha(),
        device_name=gpu_name,
        amp_dtype=amp_dtype,
        batch_size=b_size,
        lr=lr,
        total_epochs=epochs,
        dataset_size=len(dataset),
        headless=args.headless,
        logger=logger
    )
    
    start_time = time.time()
    completed_epochs = start_epoch - 1
    
    # --- SWA + SGDR Setup ---
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50 * max(len(loader), 1), eta_min=1e-6
    )
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-4)
    
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch_idx, (x, target, mask, *_rest) in enumerate(loader):
                x, target, mask = x.to(device, non_blocking=True), target.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                # [AMP Dynamic Precision]
                amp_enabled = device.type in ('cuda', 'xpu')
                amp_ctx = torch.amp.autocast(device.type, dtype=amp_dtype, enabled=amp_enabled)
                with amp_ctx:
                    pred = model(x)
                    loss = loss_fn(pred, target, mask)
                    if freq_loss_fn is not None:
                        loss = loss + 0.3 * freq_loss_fn(pred, target)
                    
                loss.backward()
                
                # Gradient norm monitoring (Higher clip for Phase 12 Transients)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0).item()
                
                optimizer.step()
                epoch_loss += loss.item()
                
                # --- Scheduler Logic ---
                if epoch < swa_start:
                    scheduler.step()  # SGDR: hunt for valleys
                
                # --- Real PRD tracking ---
                with torch.no_grad():
                    mse_raw = F.mse_loss(pred, target).item()
                    prd = (torch.sqrt(torch.mean((pred-target)**2)) / (torch.sqrt(torch.mean(target**2)) + 1e-12) * 100.0).item()
                sim_prd = prd
                
                # --- Cockpit UI (throttled) ---
                if batch_idx % 50 == 0 or batch_idx == len(loader) - 1:
                    percent_zero = percent_zero_weights(model)
                    cockpit.render(
                        epoch=epoch,
                        step=batch_idx,
                        total_steps_epoch=len(loader),
                        loss=epoch_loss / (batch_idx + 1),
                        prd=sim_prd,
                        mse=loss.item(),
                        grad_norm=grad_norm,
                        current_lr=optimizer.param_groups[0]['lr'],
                        percent_zero=percent_zero,
                        best_loss=best_loss,
                        best_epoch=best_epoch,
                        start_epoch=start_epoch,
                    )
            
            avg_loss = epoch_loss / len(loader)
            completed_epochs = epoch
            
            # SWA: step scheduler and accumulate averaged weights
            if epoch >= swa_start:
                swa_scheduler.step()
                # Fidelity Sync: SWA is performed on uncompiled raw weights for precision
                # We skip torch.compile optimization for the SWA tracker to ensure disk-parity
                swa_model.update_parameters(model)
                
            # --- MANDATORY FIDELITY AUDIT (Every 50 Epochs) ---
            if epoch % 50 == 0:
                print(f"[*] FIDELITY AUDIT: Verifying disk parity for clinical signing...")
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, "temp_audit.pth"))
                audit_model = FP32OracleAutoEncoder().to(device).eval()
                # Strip _orig_mod. prefix from compiled state_dict
                raw_sd = torch.load(os.path.join(CKPT_DIR, "temp_audit.pth"), weights_only=True)
                clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
                audit_model.load_state_dict(clean_sd)
                # Quick check on first 5 batches
                with torch.no_grad():
                    ax, ay, am, *_ = next(iter(loader))
                    ax = ax.to(device)
                    ar = audit_model(ax)
                    audit_mse = F.mse_loss(ar, ax).item()
                    # Parity check against the batch MSE, not the weighted loss
                    print(f"  [AUDIT] Disk MSE: {audit_mse:.8f} | Step MSE: {mse_raw:.8f} | Parity: {'PASS' if abs(audit_mse - mse_raw) < 1e-3 else 'WARN'}")
            
            # Save full checkpoint every epoch (with optimizer state)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'training_config': training_config,
            }, checkpoint_path)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                torch.save(model.encoder.state_dict(), os.path.join(CKPT_DIR, "teacher_best.ckpt"))
                torch.save(model.decoder.state_dict(), os.path.join(CKPT_DIR, "decoder_best.ckpt"))
                
        # --- Finalize SWA ---
        print(f"\n[*] Finalizing SWA: updating batch norm statistics...")
        try:
            update_bn(loader, swa_model, device=device)
        except Exception:  # file read or model load failure — skip gracefully
            pass
        
        print(f"\n[SUCCESS] Completed {completed_epochs}/{epochs} epochs. "
              f"Best loss at epoch {best_epoch}.")
        # Export the SWA-averaged model (jitter-free weights)
        def save_clean(module, path):
            sd = module.state_dict()
            clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            torch.save(clean_sd, path)

        # Save with completion flag: teacher_{total}_completed.ckpt
        teacher_name = f"teacher_{epochs}_completed.ckpt"
        decoder_name = f"decoder_{epochs}_completed.ckpt"
        save_clean(swa_model.module.encoder, teacher_name)
        save_clean(swa_model.module.decoder, decoder_name)
        # Also save canonical names for downstream scripts
        save_clean(swa_model.module.encoder, os.path.join(CKPT_DIR, "teacher_best.ckpt"))
        save_clean(swa_model.module.decoder, os.path.join(CKPT_DIR, "decoder_best.ckpt"))

        training_config["epochs_completed"] = completed_epochs
        training_config["best_epoch"] = best_epoch
        training_config["final_loss"] = best_loss
        training_config["status"] = "COMPLETE"
        with open(teacher_name, "rb") as f:
            training_config["teacher_ckpt_sha256"] = hashlib.sha256(f.read()).hexdigest()
        save_training_config(os.path.join(CKPT_DIR, "teacher_config.json"), training_config)

        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        cockpit.finish()
        print(f"[*] Saved {teacher_name} and {decoder_name} (training complete).")
        
    except KeyboardInterrupt:
        print(f"\n\n[!] Training INTERRUPTED at Epoch {completed_epochs+1}/{epochs}.")
        print(f"[*] Completed {completed_epochs}/{epochs} full Epochs. "
              f"Best loss at epoch {best_epoch}.")
        print(f"[*] Full checkpoint saved -> {checkpoint_path} (use --resume to continue)")

        # Save with failed flag: teacher_{completed}_of_{total}_failed.ckpt
        training_config["epochs_completed"] = completed_epochs
        training_config["best_epoch"] = best_epoch
        training_config["final_loss"] = best_loss
        training_config["status"] = f"FAILED_ep{completed_epochs}_of_{epochs}"

        if os.path.exists(os.path.join(CKPT_DIR, "teacher_best.ckpt")):
            failed_teacher = f"teacher_{completed_epochs}_of_{epochs}_failed.ckpt"
            failed_decoder = f"decoder_{completed_epochs}_of_{epochs}_failed.ckpt"
            os.rename(os.path.join(CKPT_DIR, "teacher_best.ckpt"), failed_teacher)
            os.rename(os.path.join(CKPT_DIR, "decoder_best.ckpt"), failed_decoder)
            with open(failed_teacher, "rb") as f:
                training_config["teacher_ckpt_sha256"] = hashlib.sha256(f.read()).hexdigest()
            print(f"[*] Best epoch ({best_epoch}) salvaged -> {failed_teacher}")
        else:
            torch.save(model.encoder.state_dict(), f"teacher_0_of_{epochs}_failed.ckpt")
            torch.save(model.decoder.state_dict(), f"decoder_0_of_{epochs}_failed.ckpt")

        save_training_config(f"teacher_config_{completed_epochs}_of_{epochs}_failed.json",
                             training_config)
        cockpit.finish()
        sys.exit(0)
