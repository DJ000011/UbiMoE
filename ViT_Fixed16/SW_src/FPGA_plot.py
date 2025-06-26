import numpy as np
from tqdm import tqdm

def load_classifier_params():
    """加载分类器参数"""
    classifier_weight = np.fromfile('./embeddings/classifier_weight.bin', dtype=np.float32)
    classifier_bias = np.fromfile('./embeddings/classifier_bias.bin', dtype=np.float32)
    
    # 重塑分类器权重
    classifier_weight = classifier_weight.reshape(10, -1)  # 10为类别数
    classifier_bias = classifier_bias.reshape(-1)
    
    return classifier_weight, classifier_bias

def compute_classification(fpga_output_path, num_samples, classifier_weight, classifier_bias):
    """计算分类结果"""
    correct = 0
    predictions = []
    
    # 加载真实标签
    true_labels = np.fromfile('./embeddings/labels.bin', dtype=np.int32)
    
    for i in tqdm(range(num_samples), desc="Computing classifications"):
        # 读取FPGA输出的transformer结果
        fpga_output = np.fromfile(f'{fpga_output_path}/transformer_output_{i}.bin', 
                                dtype=np.float32)
        
        # 重塑输出
        fpga_output = fpga_output.reshape(-1)  # 确保是一维数组
        
        # 计算分类层的输出
        logits = np.dot(classifier_weight, fpga_output) + classifier_bias
        
        # 获取预测类别
        pred_label = np.argmax(logits)
        predictions.append(pred_label)
        
        # 统计正确预测数
        if pred_label == true_labels[i]:
            correct += 1
    
    accuracy = 100 * correct / num_samples
    return accuracy, predictions

def main():
    # 加载分类器参数
    print("Loading classifier parameters...")
    classifier_weight, classifier_bias = load_classifier_params()
    
    # FPGA输出路径
    fpga_output_path = './fpga_output'  # 需要与FPGA输出目录匹配
    
    # 处理样本数量
    num_samples = 1000  # 需要与第一部分处理的样本数量一致
    
    # 计算分类结果
    print("\nComputing classifications...")
    accuracy, predictions = compute_classification(
        fpga_output_path, 
        num_samples, 
        classifier_weight, 
        classifier_bias
    )
    
    # 打印结果
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # 保存预测结果
    predictions = np.array(predictions, dtype=np.int32)
    predictions.tofile('./predictions.bin')
    print("Predictions saved to predictions.bin")

if __name__ == "__main__":
    main()