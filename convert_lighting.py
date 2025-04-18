import torch

# Spin checkpoint file
spin_checkpoint_path = "spin_hubert_2048.ckpt"
checkpoint = torch.load(spin_checkpoint_path, map_location="cpu")

spin_state_dict = None
if 'state_dict' in checkpoint:
    spin_state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
     spin_state_dict = checkpoint['model']
else:
    spin_state_dict = checkpoint


filtered_state_dict = {}
for key, value in spin_state_dict.items():
    new_key = key.replace("model.", "", 1) 
    if not new_key.startswith("loss_module."):
         if not new_key.startswith("model."):
              filtered_state_dict[new_key] = value

filtered_state_dict_path = "spin_ckpt.pth"
torch.save(filtered_state_dict, filtered_state_dict_path)

print(f"Filtered state dict saved to: {filtered_state_dict_path}")
