import torch
from torch import nn
from transformers import HubertConfig, HubertModel

fairseq_state_dict = torch.load('spin_ckpt.pth', map_location='cpu')

class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # Spin uses pred_head but neither are used in forward pass, keep for backwards compatibility
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


hubert = HubertModelWithFinalProj(HubertConfig())


mapping = {
    "masked_spec_embed": "encoder.mask_emb",
    "encoder.layer_norm.bias": "encoder.encoder.layer_norm.bias",
    "encoder.layer_norm.weight": "encoder.encoder.layer_norm.weight",
    "encoder.pos_conv_embed.conv.bias": "encoder.encoder.pos_conv.0.bias",
    "encoder.pos_conv_embed.conv.weight_g": "encoder.encoder.pos_conv.0.weight_g",
    "encoder.pos_conv_embed.conv.weight_v": "encoder.encoder.pos_conv.0.weight_v",
    "feature_projection.layer_norm.bias": "encoder.layer_norm.bias",
    "feature_projection.layer_norm.weight": "encoder.layer_norm.weight",
    "feature_projection.projection.bias": "encoder.post_extract_proj.bias",
    "feature_projection.projection.weight": "encoder.post_extract_proj.weight",
}

# Convert encoder
for layer in range(12):
    for j in ["q", "k", "v"]:
        mapping[
            f"encoder.layers.{layer}.attention.{j}_proj.weight"
        ] = f"encoder.encoder.layers.{layer}.self_attn.{j}_proj.weight"
        mapping[
            f"encoder.layers.{layer}.attention.{j}_proj.bias"
        ] = f"encoder.encoder.layers.{layer}.self_attn.{j}_proj.bias"

    mapping[
        f"encoder.layers.{layer}.final_layer_norm.bias"
    ] = f"encoder.encoder.layers.{layer}.final_layer_norm.bias"
    mapping[
        f"encoder.layers.{layer}.final_layer_norm.weight"
    ] = f"encoder.encoder.layers.{layer}.final_layer_norm.weight"

    mapping[
        f"encoder.layers.{layer}.layer_norm.bias"
    ] = f"encoder.encoder.layers.{layer}.self_attn_layer_norm.bias"
    mapping[
        f"encoder.layers.{layer}.layer_norm.weight"
    ] = f"encoder.encoder.layers.{layer}.self_attn_layer_norm.weight"

    mapping[
        f"encoder.layers.{layer}.attention.out_proj.bias"
    ] = f"encoder.encoder.layers.{layer}.self_attn.out_proj.bias"
    mapping[
        f"encoder.layers.{layer}.attention.out_proj.weight"
    ] = f"encoder.encoder.layers.{layer}.self_attn.out_proj.weight"

    mapping[
        f"encoder.layers.{layer}.feed_forward.intermediate_dense.bias"
    ] = f"encoder.encoder.layers.{layer}.fc1.bias"
    mapping[
        f"encoder.layers.{layer}.feed_forward.intermediate_dense.weight"
    ] = f"encoder.encoder.layers.{layer}.fc1.weight"

    mapping[
        f"encoder.layers.{layer}.feed_forward.output_dense.bias"
    ] = f"encoder.encoder.layers.{layer}.fc2.bias"
    mapping[
        f"encoder.layers.{layer}.feed_forward.output_dense.weight"
    ] = f"encoder.encoder.layers.{layer}.fc2.weight"

# Convert Conv Layers
for layer in range(7):
    mapping[
        f"feature_extractor.conv_layers.{layer}.conv.weight"
    ] = f"encoder.feature_extractor.conv_layers.{layer}.0.weight"

# The layer norm within the conv block is also nested
    if layer != 0:
        continue 

    mapping[
        f"feature_extractor.conv_layers.{layer}.layer_norm.weight"
    ] = f"encoder.feature_extractor.conv_layers.{layer}.2.weight"
    mapping[
        f"feature_extractor.conv_layers.{layer}.layer_norm.bias"
    ] = f"encoder.feature_extractor.conv_layers.{layer}.2.bias"

hf_keys = set(hubert.state_dict().keys())
fair_keys = set(fairseq_state_dict.keys())

hf_keys -= set(mapping.keys())
fair_keys -= set(mapping.values())

for i, j in zip(sorted(hf_keys), sorted(fair_keys)):
    print(i, j)

print(hf_keys, fair_keys)
print(len(hf_keys), len(fair_keys))

new_state_dict = {}
for k, v in mapping.items():
    new_state_dict[k] = fairseq_state_dict[v]

print("Loading mapped state dict into Hugging Face model...")
x = hubert.load_state_dict(new_state_dict, strict=False)
hubert.eval()

# This saves a .safetensors version
hubert.save_pretrained(".")
print("Generated config.json")

weights_path = "pytorch_model.bin"
torch.save(hubert.state_dict(), weights_path)
print(f"Saved weights to {weights_path}")
