"""Loss ingredients (ADR 0050/0051) — the differentiable objective each trainer
optimises, extracted verbatim from the trainer inner loops so the same loss can
be selected from a recipe + carried on the stage cache key (cache_relevant).

Five specs live here:
  - ``joint_codec``          (student/train_joint.py — codec joint loss)
  - ``four_state_objective`` (snn/train_4state_controller.py — CE/ordinal/CRF
                              + spike + distill)
  - ``masked_recon_mse_time``  (snn/pretrain_ssl_tueg.py — masked-mean SSL loss)
  - ``masked_recon_mse_patch`` (student/pretrain_mae.py — MAE all-mean masked MSE)
  - ``teacher_mse``          (oracle/train_l3_teacher.py — plain F.mse_loss)
"""
