import os
import torch
import numpy as np
from transformers import AutoModelForImageClassification

def create_save_dir():
    """创建保存权重的目录"""
    base_dir = './weights'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def save_tensor_to_bin(tensor, filepath):
    """将张量保存为二进制文件"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    tensor.tofile(filepath)
    print(f"Saved to {filepath}, shape: {tensor.shape}")

def export_embeddings(model, base_dir):
    """导出embedding相关的权重"""
    # 保存patch embedding权重
    patch_embed = model.vit.embeddings.patch_embeddings.projection.weight
    save_tensor_to_bin(
        patch_embed,
        os.path.join(base_dir, "patch_embed_weight_fp.bin")
    )
    
    # 保存patch embedding偏置
    patch_embed_bias = model.vit.embeddings.patch_embeddings.projection.bias
    save_tensor_to_bin(
        patch_embed_bias,
        os.path.join(base_dir, "patch_embed_bias_fp.bin")
    )
    
    # 保存position embeddings
    pos_embed = model.vit.embeddings.position_embeddings
    save_tensor_to_bin(
        pos_embed,
        os.path.join(base_dir, "pos_embed_fp.bin")
    )
    
    # 保存cls token
    cls_token = model.vit.embeddings.cls_token
    save_tensor_to_bin(
        cls_token,
        os.path.join(base_dir, "cls_token_fp.bin")
    )

def export_layer_weights(model, base_dir):
    """导出transformer层的权重"""
    for i, layer in enumerate(model.vit.encoder.layer):
        layer_prefix = f"l{i}"
        
        # 保存attention权重
        # qkv weights
        qkv_weight = layer.attention.attention.qkv.weight
        save_tensor_to_bin(
            qkv_weight,
            os.path.join(base_dir, f"{layer_prefix}_attn_qkv_weight_fp.bin")
        )
        
        qkv_bias = layer.attention.attention.qkv.bias
        save_tensor_to_bin(
            qkv_bias,
            os.path.join(base_dir, f"{layer_prefix}_attn_qkv_bias_fp.bin")
        )
        
        # projection weights
        proj_weight = layer.attention.output.dense.weight
        save_tensor_to_bin(
            proj_weight,
            os.path.join(base_dir, f"{layer_prefix}_attn_proj_weight_fp.bin")
        )
        
        proj_bias = layer.attention.output.dense.bias
        save_tensor_to_bin(
            proj_bias,
            os.path.join(base_dir, f"{layer_prefix}_attn_proj_bias_fp.bin")
        )
        
        # 保存MLP权重
        # fc1
        mlp_fc1_weight = layer.intermediate.dense.weight
        save_tensor_to_bin(
            mlp_fc1_weight,
            os.path.join(base_dir, f"{layer_prefix}_mlp_fc1_weight_fp.bin")
        )
        
        mlp_fc1_bias = layer.intermediate.dense.bias
        save_tensor_to_bin(
            mlp_fc1_bias,
            os.path.join(base_dir, f"{layer_prefix}_mlp_fc1_bias_fp.bin")
        )
        
        # fc2
        mlp_fc2_weight = layer.output.dense.weight
        save_tensor_to_bin(
            mlp_fc2_weight,
            os.path.join(base_dir, f"{layer_prefix}_mlp_fc2_weight_fp.bin")
        )
        
        mlp_fc2_bias = layer.output.dense.bias
        save_tensor_to_bin(
            mlp_fc2_bias,
            os.path.join(base_dir, f"{layer_prefix}_mlp_fc2_bias_fp.bin")
        )
        
        # 保存LayerNorm权重
        # attention LayerNorm
        attn_ln_weight = layer.attention.attention.attention_head_norm.weight
        save_tensor_to_bin(
            attn_ln_weight,
            os.path.join(base_dir, f"{layer_prefix}_attn_ln_weight_fp.bin")
        )
        
        attn_ln_bias = layer.attention.attention.attention_head_norm.bias
        save_tensor_to_bin(
            attn_ln_bias,
            os.path.join(base_dir, f"{layer_prefix}_attn_ln_bias_fp.bin")
        )
        
        # mlp LayerNorm
        mlp_ln_weight = layer.layernorm_before.weight
        save_tensor_to_bin(
            mlp_ln_weight,
            os.path.join(base_dir, f"{layer_prefix}_mlp_ln_weight_fp.bin")
        )
        
        mlp_ln_bias = layer.layernorm_before.bias
        save_tensor_to_bin(
            mlp_ln_bias,
            os.path.join(base_dir, f"{layer_prefix}_mlp_ln_bias_fp.bin")
        )

def export_classifier(model, base_dir):
    """导出分类器权重"""
    classifier_weight = model.classifier.weight
    save_tensor_to_bin(
        classifier_weight,
        os.path.join(base_dir, "classifier_weight_fp.bin")
    )
    
    classifier_bias = model.classifier.bias
    save_tensor_to_bin(
        classifier_bias,
        os.path.join(base_dir, "classifier_bias_fp.bin")
    )

def main():
    print("Loading model...")
    model = AutoModelForImageClassification.from_pretrained(
        "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    )
    model.eval()
    
    base_dir = create_save_dir()
    print(f"\nExporting weights to {base_dir}")
    
    print("\nExporting embeddings...")
    export_embeddings(model, base_dir)
    
    print("\nExporting transformer layers...")
    export_layer_weights(model, base_dir)
    
    print("\nExporting classifier...")
    export_classifier(model, base_dir)
    
    print("\nExport completed!")

if __name__ == "__main__":
    main()