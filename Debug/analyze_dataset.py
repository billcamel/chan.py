import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

def convert_to_csv(data_path: str):
    """将特征数据转换为CSV格式并进行基本分析
    
    Args:
        data_path: 特征数据目录路径
    """
    # 加载数据
    try:
        X = np.load(os.path.join(data_path, "features.npy"))
        y = np.load(os.path.join(data_path, "labels.npy"))
        with open(os.path.join(data_path, "feature_meta.json"), "r") as f:
            feature_meta = json.load(f)
        with open(os.path.join(data_path, "data_info.json"), "r") as f:
            data_info = json.load(f)
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        return

    # 创建DataFrame
    df = pd.DataFrame(X, columns=list(feature_meta.keys()))
    df['label'] = y
    
    # 添加基本信息
    print("\n数据基本信息:")
    print(f"样本数: {len(df)}")
    print(f"特征数: {len(feature_meta)}")
    print(f"正样本比例: {df['label'].mean():.2%}")
    
    # 特征统计分析
    print("\n特征统计:")
    stats = df.describe()
    print("\n数值范围:")
    for col in df.columns:
        if col != 'label':
            non_zero = (df[col] != 0).mean()
            # print(f"{col}:")
            # print(f"  范围: [{df[col].min():.4f}, {df[col].max():.4f}]")
            # print(f"  均值: {df[col].mean():.4f}")
            # print(f"  标准差: {df[col].std():.4f}")
            # print(f"  非零比例: {non_zero:.2%}")
    
    # 保存CSV文件
    csv_dir = os.path.join(data_path, "csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        
    # 保存完整数据
    df.to_csv(os.path.join(csv_dir, "features_all.csv"), index=False)
    
    # 保存正样本数据
    df[df['label'] == 1].to_csv(os.path.join(csv_dir, "features_positive.csv"), index=False)
    
    # 保存负样本数据
    df[df['label'] == 0].to_csv(os.path.join(csv_dir, "features_negative.csv"), index=False)
    
    # 保存特征统计信息
    stats.to_csv(os.path.join(csv_dir, "feature_stats.csv"))
    
    # 计算特征相关性
    corr = df.corr()
    corr.to_csv(os.path.join(csv_dir, "feature_correlations.csv"))
    
    # 找出与标签相关性最强的特征
    label_corr = corr['label'].sort_values(ascending=False)
    print("\n与标签相关性最强的特征:")
    print(label_corr[1:11])  # 排除标签本身，显示前10个
    
    print(f"\nCSV文件已保存到: {csv_dir}")
    print("文件列表:")
    print("- features_all.csv (所有样本)")
    print("- features_positive.csv (正样本)")
    print("- features_negative.csv (负样本)")
    print("- feature_stats.csv (特征统计)")
    print("- feature_correlations.csv (特征相关性)")

if __name__ == "__main__":
    # 获取数据目录
    data_dir = "feature_data"
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        sys.exit(1)
        
    # 过滤掉隐藏文件和非目录
    data_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d)) and 
                 not d.startswith('.')]
    
    if not data_dirs:
        print(f"错误: 在 {data_dir} 中未找到有效的数据目录")
        sys.exit(1)
    
    # 如果没有指定目录，使用最新的
    if data_path is None:
        data_path = os.path.join(data_dir, 
                                max(data_dirs, 
                                    key=lambda x: os.path.getctime(os.path.join(data_dir, x))))
        print(f"使用最新数据目录: {data_path}")
    else:
        # 检查指定的目录
        if not os.path.exists(data_path):
            dir_name = os.path.basename(data_path)
            full_path = os.path.join(data_dir, dir_name)
            if os.path.exists(full_path):
                data_path = full_path
            else:
                print(f"警告: 指定的目录不存在: {data_path}")
                sys.exit(1)
    
    convert_to_csv(data_path) 