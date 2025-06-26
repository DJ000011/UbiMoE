import torch
import torchvision
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm

def load_cifar10():
    """加载CIFAR-10测试数据集"""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True
    )
    return test_set, classes

def compute_embeddings(model, processor, test_set, num_samples=1000):
    """计算图像的embedding并保存"""
    model.eval()
    
    # 创建进度条
    pbar = tqdm(total=min(num_samples, len(test_set)), desc="Processing embeddings")
    
    # 保存标签
    labels = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(test_set))):
            # 获取图像和标签
            image, label = test_set[i]
            labels.append(label)
            
            # 处理图像
            inputs = processor(images=image, return_tensors="pt")
            
            # 获取embedding (包含CLS token的完整embedding)
            embeddings = model.vit.embeddings(inputs['pixel_values'])
            
            # 保存embedding
            embedding_path = f'./embeddings/embedding_{i}.bin'
            embeddings.cpu().numpy().astype(np.float32).tofile(embedding_path)
            
            pbar.update(1)
    
    pbar.close()
    
    # 保存标签
    labels = np.array(labels, dtype=np.int32)
    labels.tofile('./embeddings/labels.bin')
    
    return len(labels)

def main():
    # 创建保存目录
    import os
    os.makedirs('./embeddings', exist_ok=True)
    
    # 加载模型和处理器
    print("Loading model and processor...")
    processor = AutoImageProcessor.from_pretrained(
        "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "MF21377197/vit-small-patch16-224-finetuned-Cifar10"
    )
    
    # 加载数据集
    print("Loading CIFAR-10 dataset...")
    test_set, classes = load_cifar10()
    
    # 计算并保存embeddings
    print("Computing and saving embeddings...")
    num_processed = compute_embeddings(model, processor, test_set, num_samples=1000)
    
    print(f"\nProcessed {num_processed} images")
    print("Embeddings saved to ./embeddings directory")
    
    # 保存分类层权重供后续使用
    classifier_weight = model.classifier.weight.detach().cpu().numpy()
    classifier_bias = model.classifier.bias.detach().cpu().numpy()
    classifier_weight.astype(np.float32).tofile('./embeddings/classifier_weight.bin')
    classifier_bias.astype(np.float32).tofile('./embeddings/classifier_bias.bin')

if __name__ == "__main__":
    main()