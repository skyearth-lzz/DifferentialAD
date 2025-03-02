"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-02-20
@Description：
==================================================
"""
import gc
import os
import argparse

import torch.cuda
from torch.backends import cudnn

from model.anomaly_tranformer.solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    import torch
    torch.cuda.empty_cache()  # 清理CUDA缓存
    # 启用 cuDNN 的自动调优功能,主要进行以下优化:
    # 1. 针对当前硬件测试不同的卷积算法(如 GEMM、FFT、Winograd 等)
    # 2. 自动选择最快的数据布局和内存访问模式
    # 3. 为特定的输入尺寸和 batch size 优化计算路径
    # 4. 缓存最优配置以供后续使用
    # 注意:仅适用于输入尺寸固定的场景,否则反复寻找最优配置反而会降低性能
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        os.mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()

    # vars()函数用于返回对象的属性和属性值组成的字典
    # 这里将argparse.Namespace对象转换为字典,方便后续访问参数
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
