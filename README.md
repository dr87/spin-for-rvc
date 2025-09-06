# spin-for-rvc

## This project replaces contentvec with [spin](https://arxiv.org/pdf/2305.11072) for disentangling speaker information.

### 

Download a spin hubert checkpoint [here](https://github.com/vectominist/spin)   (Offical checkpoints have timbre bleed in RVC, it is recommend to download my trained model below)

Run convert_lighting.py to convert the spin pylightning checkpoint to a standard pytorch .pth file

Run convert_spin_to_transformers.py to convert the new spin.pth checkpoint for a drop in replacement for RVC.

Pretrain's need to be finetuned again after extracting features with spin.


## UPDATE:

~~The official checkpoints with layer 11 and 12 trained have timbre bleed. The current way to fix this is by training transformer layers 7-12 instead. Below is the currently accepted checkpoint for RVC and currently used on AIhub models. This model is trained based on librespeech 400+ hour dataset.~~

~~https://huggingface.co/dr87/spin-for-rvc/resolve/main/spin_layers_7_12.zip~~

Spinv2 is here. Timbre bleed is fixed, as well as slightly slurred speech with shorter datasets. Pronunciation should be more clear, and codebook has been reduced to 1024 for a number of reasons, as it just peforms better in RVC.

https://huggingface.co/dr87/spinv2_rvc/blob/main/spinv2.zip

