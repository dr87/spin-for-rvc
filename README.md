# spin-for-rvc

This project replaces contentvec with [spin](https://arxiv.org/pdf/2305.11072) for disentangling speaker information.

Run convert_lighting.py to convert the spin pylightning checkpoint to a standard pytorch .pth file
Run convert_spin_to_transformers.py to convert the new spin.pth checkpoint for a drop in replacement for RVC.

Pretrain's need to be finetuned again after extracting features with spin.
