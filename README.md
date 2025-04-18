# spin-for-rvc

## This project replaces contentvec with [spin](https://arxiv.org/pdf/2305.11072) for disentangling speaker information.

### 

Download a spin hubert checkpoint [here](https://github.com/vectominist/spin)

Run convert_lighting.py to convert the spin pylightning checkpoint to a standard pytorch .pth file

Run convert_spin_to_transformers.py to convert the new spin.pth checkpoint for a drop in replacement for RVC.

Pretrain's need to be finetuned again after extracting features with spin.

A premade 2048 cluster hubert-spin model can be downloaded [here](https://huggingface.co/dr87/spin-for-rvc/blob/main/2048-spin-transformers.zip)
