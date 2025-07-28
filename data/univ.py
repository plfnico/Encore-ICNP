import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm  # 进度条支持（可选）

import shutil

def process_csv_files(input_dir, output_dir):
    """
    处理所有CSV文件：
    1. 将时间戳转换为相对于文件首条记录的差值
    2. 仅保留新时间差和flow_size列
    3. 分块处理大文件避免内存溢出
    """
    try:
        shutil.rmtree(output_dir) 
    except:
        pass
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CSV文件路径
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"⚠️ 目录中未找到CSV文件: {input_dir}")
        return

    # 批量处理文件
    for file_path in tqdm(csv_files, desc="处理进度"):
        try:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"processed_{filename}")
            
            # 分块读取大文件（每块1万行）
            chunk_iter = pd.read_csv(file_path, chunksize=10000)
            first_chunk = True
            base_time = None  # 存储文件首条时间戳
            
            for chunk in chunk_iter:
                # 跳过空文件或缺失关键列的文件
                if chunk.empty or 'timestamp' not in chunk.columns or 'flow_size' not in chunk.columns:
                    continue
                
                # 记录首个时间戳（仅首次读取时获取）
                if first_chunk:
                    base_time = chunk['timestamp'].iloc[0]
                    first_chunk = False
                
                # 计算相对时间差
                chunk['relative_time'] = chunk['timestamp'] - base_time
                
                # 保留目标列并写入文件
                result = chunk[['relative_time', 'flow_size']]
                result.rename(columns={
        'relative_time': 'time',  # 原始时间戳列名
        'flow_size': 'size'        # 原始流量大小列名
    }, inplace=True)
                mode = 'w' if not os.path.exists(output_path) else 'a'
                header = (mode == 'w')  # 仅首次写入列名
                result.to_csv(output_path, mode=mode, header=header, index=False)
                
            print(f"✅ 生成: {output_path}")
            
        except Exception as e:
            print(f"❌ 处理失败 [{filename}]: {str(e)}")

# 调用示例
if __name__ == "__main__":
    process_csv_files(
        input_dir="univ/univ_flow/",  # 替换为你的输入目录
        output_dir="raw/app_univ/"           # 替换为输出目录
    )