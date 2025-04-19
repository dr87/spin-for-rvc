# spin-for-rvc

## This project replaces contentvec with [spin](https://arxiv.org/pdf/2305.11072) for disentangling speaker information.

### 

Download a spin hubert checkpoint [here](https://github.com/vectominist/spin)

Run convert_lighting.py to convert the spin pylightning checkpoint to a standard pytorch .pth file

Run convert_spin_to_transformers.py to convert the new spin.pth checkpoint for a drop in replacement for RVC.

Pretrain's need to be finetuned again after extracting features with spin.


## UPDATE:

This is an updated spin checkpoint , 2048 clusters, I created that works better for the purpose of RVC generalization\

Librespeech 100 clean -> + additional 350 clean set\
Hours: 100 -> 450\
Speakers: 251 -> 1172

This converges models much faster, likely due to the massively increased speaker exposure, and even works with reasonable sized datasets without pretrain finetuning. (I used a 1 hour 45 minute on an existing pretrain, and speech adapted quickly)\

https://huggingface.co/dr87/spin-for-rvc/blob/main/SPIN_450H_FINETUNE_26768.zip
