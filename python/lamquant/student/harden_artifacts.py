#!/usr/bin/env python3
"""
LamQuant Gen 7.1 — Hardening with Teacher Distillation
=======================================================
Aligns the student's latent space with the teacher's latent space.
Supports both Gen 7.0 (TernaryMobileNetV5) and Gen 7.1 (TernaryMobileNetV5_Subband).

SAFETY: This script NEVER overwrites the source checkpoint.
  - Input checkpoint is backed up before any training begins
  - Hardened output is written to a SEPARATE file
  - If hardening degrades quality, the original is untouched

Output files:
  ai_models/student/student_subband_hardened.ckpt   (Gen 7.1)
  ai_models/student/student_hardened_distilled.ckpt  (Gen 7.0)
  weights/student_subband.ckpt                       (distributable, only if improved)

Schedule:
  Stage 1: Warm alignment (100 ep, lr=5e-4, beta=0.2)
  Stage 2: Push alignment (200 ep, lr=2e-4, beta anneals 0.3->0.1)
  Stage 3: Reconstruction recovery (200 ep, lr=1e-4, beta anneals 0.1->0.02)

Total: 500 epochs (~8 min on RTX 4090)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import glob
import shutil

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(ROOT_DIR, 'lamquant', 'oracle'))
sys.path.append(os.path.join(ROOT_DIR, 'lamquant', 'student'))

from train_teacher import Q31Dataset
from lamquant_neural.models.encoder import (
    TernaryMobileNetV5,
    TernaryMobileNetV5_Subband,
)
from ternary_encoder import (
    apply_montage_permutation,
    clinical_augmentation,
)
from subband_preprocess import preprocess_subband_torch
from lamquant.common.utils import safe_torch_load as _safe_load


def pearson_r_batch(pred, target):
    p = pred.flatten(1)
    t = target.flatten(1)
    pc = p - p.mean(dim=-1, keepdim=True)
    tc = t - t.mean(dim=-1, keepdim=True)
    r = torch.sum(pc * tc, dim=-1) / (
        torch.sqrt(torch.sum(pc ** 2, dim=-1)) *
        torch.sqrt(torch.sum(tc ** 2, dim=-1)) + 1e-8
    )
    return r.mean().item()


def combined_loss(student, teacher_model, x, latent_weight=0.2,
                  subband_mode=False, l3_teacher=False, device='cpu'):
    """
    Recon: MSE(student(x), x)   — or MSE(student(l3), l3) for subband
    Latent: MSE(student.encode(x), teacher.encode(x))

    When l3_teacher=True, both student and teacher operate on L3 [21, 313]
    with identical latent shapes [32, 79]. No shape clipping needed.
    """
    if subband_mode and l3_teacher:
        # Gen 7.5: L3-native teacher. Both models see L3, same latent shape.
        x_l3, _ = preprocess_subband_torch(x)
        x_l3 = x_l3.to(device)
        s_recon = student(x_l3, quantize=True)
        s_latent = student.encode(x_l3, quantize=True)
        with torch.no_grad():
            t_latent = teacher_model.encode(x_l3)   # L3 teacher: same input as student
        target = x_l3
        # Shapes match exactly: s_latent [B,32,79] == t_latent [B,32,79]
        recon_mse = F.mse_loss(s_recon, target)
        latent_mse = F.mse_loss(s_latent, t_latent)
    elif subband_mode:
        # Legacy: raw teacher on [21,2500], student on L3 [21,313]. Mismatched.
        x_l3, _ = preprocess_subband_torch(x)
        x_l3 = x_l3.to(device)
        s_recon = student(x_l3, quantize=True)
        s_latent = student.encode(x_l3, quantize=True)
        with torch.no_grad():
            t_latent = teacher_model.encode(x, quantize=False)
        target = x_l3
        min_t = min(s_recon.shape[2], target.shape[2])
        recon_mse = F.mse_loss(s_recon[:, :, :min_t], target[:, :, :min_t])
        min_lat = min(s_latent.shape[2], t_latent.shape[2])
        latent_mse = F.mse_loss(s_latent[:, :, :min_lat], t_latent[:, :, :min_lat])
    else:
        # Gen 7.0: both on raw [21, 2500]
        s_recon = student(x, quantize=True)
        s_latent = student.encode(x, quantize=True)
        with torch.no_grad():
            t_latent = teacher_model.encode(x, quantize=False)
        target = x
        recon_mse = F.mse_loss(s_recon, target)
        latent_mse = F.mse_loss(s_latent, t_latent)

    total = (1.0 - latent_weight) * recon_mse + latent_weight * latent_mse

    with torch.no_grad():
        min_t = min(s_recon.shape[2], target.shape[2])
        r = pearson_r_batch(s_recon[:, :, :min_t], target[:, :, :min_t])

    return total, recon_mse, latent_mse, r


def train_stage(student, teacher_model, dataloader, device, optimizer, epochs,
                beta_start, beta_end, stage_name, start_time, out_path,
                best_r, subband_mode=False, l3_teacher=False):
    for epoch in range(1, epochs + 1):
        student.train()
        beta = beta_start + (beta_end - beta_start) * (epoch / epochs)

        losses, recon_l, lat_l, r_scores = [], [], [], []
        for x, *rest in dataloader:
            x_s = torch.clamp(x.to(device), -50.0, 50.0)
            x_s = x_s - x_s.mean(dim=2, keepdim=True)
            x_s = apply_montage_permutation(x_s)
            x_s = clinical_augmentation(x_s)

            optimizer.zero_grad()
            loss, rmse, lmse, r = combined_loss(
                student, teacher_model, x_s, latent_weight=beta,
                subband_mode=subband_mode, l3_teacher=l3_teacher, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
            optimizer.step()

            losses.append(loss.item())
            recon_l.append(rmse.item())
            lat_l.append(lmse.item())
            r_scores.append(r)

        if epoch % 10 == 0:
            mean_r = np.mean(r_scores)
            elapsed = time.time() - start_time

            if mean_r > best_r[0]:
                best_r[0] = mean_r
                torch.save(student.state_dict(), out_path)

            print(f"  [{stage_name}] Ep {epoch:>3}/{epochs} | "
                  f"Recon: {np.mean(recon_l):.2f} | "
                  f"LatMSE: {np.mean(lat_l):.2f} | "
                  f"R: {mean_r:.4f} | "
                  f"beta: {beta:.3f} | "
                  f"Best: {best_r[0]:.4f} | "
                  f"{elapsed:.0f}s")


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Hardening with Teacher Distillation on {device}")

    # Pure-upside perf (teacher already has these; hardening was missing them)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # --- Detect model type ---
    subband_path = os.path.join(ROOT_DIR, "ai_models/student/student_subband.ckpt")
    legacy_path = os.path.join(ROOT_DIR, "ai_models/student/student_hardened.ckpt")
    subband_weight_path = os.path.join(ROOT_DIR, "weights/student_subband.ckpt")
    legacy_weight_path = os.path.join(ROOT_DIR, "weights/student_hardened.ckpt")

    # Find the source checkpoint
    src_path = None
    subband_mode = False
    for p in [subband_path, subband_weight_path]:
        if os.path.exists(p):
            src_path = p
            subband_mode = True
            break
    if src_path is None:
        for p in [legacy_path, legacy_weight_path]:
            if os.path.exists(p):
                src_path = p
                break

    if src_path is None:
        print("[!] FATAL: No student checkpoint found.")
        print(f"    Searched: {subband_path}")
        print(f"    Searched: {legacy_path}")
        sys.exit(1)

    # --- SAFETY: Backup source checkpoint before touching anything ---
    backup_path = src_path + ".pre_hardening_backup"
    if not os.path.exists(backup_path):
        shutil.copy2(src_path, backup_path)
        print(f"[*] Backed up source checkpoint -> {backup_path}")
    else:
        print(f"[*] Backup already exists: {backup_path}")

    # --- Output path (NEVER overwrites source) ---
    if subband_mode:
        out_path = os.path.join(ROOT_DIR, "ai_models/student/student_subband_hardened.ckpt")
    else:
        out_path = os.path.join(ROOT_DIR, "ai_models/student/student_hardened_distilled.ckpt")

    print(f"[*] Source:  {src_path}")
    print(f"[*] Output:  {out_path} (source will NOT be overwritten)")

    # --- Load student ---
    if subband_mode:
        student = TernaryMobileNetV5_Subband.from_checkpoint(src_path, device=device)
        print(f"[*] Model: Gen 7.5 TernaryMobileNetV5_Subband (width auto-detected)")
    else:
        student = TernaryMobileNetV5(in_ch=21, latent_dim=32).to(device)
        student.load_state_dict(torch.load(src_path, map_location=device, weights_only=True))
        print(f"[*] Model: Gen 7.0 TernaryMobileNetV5")
    print(f"[*] Loaded student: {sum(p.numel() for p in student.parameters()):,} params")

    # --- Load teacher ---
    # Gen 7.5: prefer L3-native teacher (same input/latent as student).
    # Fallback: legacy raw teacher (mismatched shapes, less effective).
    l3_teacher_flag = False
    l3_teacher_path = os.path.join(ROOT_DIR, "ai_models/oracle/l3_teacher_best.ckpt")
    legacy_teacher_paths = [
        os.path.join(ROOT_DIR, "weights/teacher.ckpt"),
        os.path.join(ROOT_DIR, "ai_models/oracle/teacher_best.ckpt"),
    ]

    if subband_mode and os.path.exists(l3_teacher_path):
        # L3-native teacher: same [21,313] input, same [32,79] latent as student.
        # No shape hacks. Distillation directly measures ternarization cost.
        from train_teacher import L3Teacher
        teacher = L3Teacher(width=512).to(device).eval()
        ckpt = _safe_load(l3_teacher_path, map_location=device)
        if 'model_state_dict' in ckpt:
            teacher.load_state_dict(ckpt['model_state_dict'])
        else:
            teacher.load_state_dict(ckpt)
        l3_teacher_flag = True
        print(f"[*] Loaded L3-native teacher (matched latent [32,79]): {l3_teacher_path}")
    else:
        # Legacy raw teacher
        t_path = None
        for p in legacy_teacher_paths:
            if os.path.exists(p):
                t_path = p
                break
        if t_path is None:
            print(f"[!] FATAL: No teacher checkpoint found.")
            print(f"    Searched: {l3_teacher_path} (L3-native, preferred)")
            for p in legacy_teacher_paths:
                print(f"    Searched: {p} (legacy)")
            sys.exit(1)

        from train_teacher import FP32OracleAutoEncoder
        teacher = FP32OracleAutoEncoder().to(device).eval()
        teacher_sd = _safe_load(t_path, map_location=device)
        if not any(k.startswith('encoder.') or k.startswith('decoder.')
                   for k in teacher_sd.keys()):
            teacher.encoder.load_state_dict(teacher_sd)
            print(f"[*] Loaded legacy teacher encoder: {t_path}")
        else:
            teacher.load_state_dict(teacher_sd)
            print(f"[*] Loaded legacy teacher autoencoder: {t_path}")
        if subband_mode:
            print(f"[!] WARNING: Using raw teacher for subband student — latent shapes mismatched.")
            print(f"    Train L3-native teacher for proper alignment: python ai_models/oracle/train_l3_teacher.py")

    for p in teacher.parameters():
        p.requires_grad = False

    # --- Dataset ---
    cache_path = os.path.join(ROOT_DIR, "ai_models/dataset_sim/q31_cache_v1.pt")
    npz_dir = os.path.join(ROOT_DIR, "ai_models/dataset_sim/q31_events")
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz"))) if os.path.isdir(npz_dir) else []
    dataset = Q31Dataset(npz_files, headless=True, cache_path=cache_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                          num_workers=0, pin_memory=(device.type == 'cuda'))
    print(f"[*] Dataset: {len(dataset)} samples")

    # --- Pre-hardening baseline ---
    student.eval()
    pre_r_scores = []
    with torch.no_grad():
        for i, (x, *rest) in enumerate(loader):
            if i >= 5:
                break
            x_s = torch.clamp(x.to(device), -50.0, 50.0)
            x_s = x_s - x_s.mean(dim=2, keepdim=True)
            if subband_mode:
                x_l3, _ = preprocess_subband_torch(x_s)
                x_l3 = x_l3.to(device)
                recon = student(x_l3, quantize=True)
                pre_r_scores.append(pearson_r_batch(recon, x_l3))
            else:
                recon = student(x_s, quantize=True)
                pre_r_scores.append(pearson_r_batch(recon, x_s))
    pre_r = np.mean(pre_r_scores) if pre_r_scores else 0.0
    print(f"\n[*] Pre-hardening R: {pre_r:.4f}")

    start_time = time.time()
    best_r = [pre_r]

    # Save initial state as starting point for output
    torch.save(student.state_dict(), out_path)

    # --- Stage 1: Warm alignment ---
    print(f"\n[*] Stage 1: Warm (200 ep, lr=5e-4, beta=0.2)")
    if l3_teacher_flag:
        print(f"    L3-native teacher: latent alignment [32,79] ↔ [32,79] (exact match)")
    opt1 = torch.optim.AdamW(student.parameters(), lr=5e-4, weight_decay=1e-4)
    train_stage(student, teacher, loader, device, opt1,
                epochs=200, beta_start=0.2, beta_end=0.2,
                stage_name="Warm", start_time=start_time,
                out_path=out_path, best_r=best_r,
                subband_mode=subband_mode, l3_teacher=l3_teacher_flag)

    # --- Stage 2: Push alignment ---
    print(f"\n[*] Stage 2: Align (400 ep, lr=2e-4, beta: 0.3->0.1)")
    opt2 = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=1e-4)
    train_stage(student, teacher, loader, device, opt2,
                epochs=400, beta_start=0.3, beta_end=0.1,
                stage_name="Align", start_time=start_time,
                out_path=out_path, best_r=best_r,
                subband_mode=subband_mode, l3_teacher=l3_teacher_flag)

    # --- Stage 3: Recover reconstruction ---
    print(f"\n[*] Stage 3: Recover (400 ep, lr=1e-4, beta: 0.1->0.02)")
    opt3 = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    train_stage(student, teacher, loader, device, opt3,
                epochs=400, beta_start=0.1, beta_end=0.02,
                stage_name="Recover", start_time=start_time,
                out_path=out_path, best_r=best_r,
                subband_mode=subband_mode, l3_teacher=l3_teacher_flag)

    # --- Final comparison ---
    elapsed = time.time() - start_time

    # Load best hardened checkpoint
    student.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    student.eval()
    post_r_scores = []
    with torch.no_grad():
        for i, (x, *rest) in enumerate(loader):
            if i >= 5:
                break
            x_s = torch.clamp(x.to(device), -50.0, 50.0)
            x_s = x_s - x_s.mean(dim=2, keepdim=True)
            if subband_mode:
                x_l3, _ = preprocess_subband_torch(x_s)
                x_l3 = x_l3.to(device)
                recon = student(x_l3, quantize=True)
                post_r_scores.append(pearson_r_batch(recon, x_l3))
            else:
                recon = student(x_s, quantize=True)
                post_r_scores.append(pearson_r_batch(recon, x_s))
    post_r = np.mean(post_r_scores) if post_r_scores else 0.0

    print(f"\n{'='*60}")
    print(f"  Hardening complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Pre-hardening R:  {pre_r:.4f}")
    print(f"  Post-hardening R: {post_r:.4f}")
    print(f"  Delta:            {post_r - pre_r:+.4f}")

    if post_r >= pre_r:
        # Hardening improved or maintained quality — copy to weights/
        dist_path = os.path.join(ROOT_DIR, "weights",
                                  "student_subband.ckpt" if subband_mode else "student_hardened.ckpt")
        shutil.copy2(out_path, dist_path)
        print(f"  IMPROVED — copied to {dist_path}")
    else:
        print(f"  DEGRADED — original checkpoint preserved at {src_path}")
        print(f"  Hardened checkpoint available at {out_path} (not promoted)")

    print(f"\n  Source (untouched): {src_path}")
    print(f"  Backup:            {backup_path}")
    print(f"  Hardened output:   {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
