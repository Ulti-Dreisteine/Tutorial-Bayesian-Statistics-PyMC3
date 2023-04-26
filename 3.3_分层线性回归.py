# -*- coding: utf-8 -*-
"""
Created on 2023/04/25 16:01:02

@File -> 3.3_分层线性回归.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 分层线性回归
"""

import pandas as pd
import numpy as np
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":

    # #### 生成数据 #################################################################################

    N = 20
    M = 8  # 组数
    idxs = np.repeat(range(M - 1), N)
    idxs = np.append(idxs, 7)
    
    alpha_real = np.random.normal(2.5, 0.5, size=M)
    beta_real = np.random.beta(6, 1, size=M)
    eps_real = np.random.normal(0, 0.5, size=len(idxs))
    
    x_samples = np.random.normal(10, 1, len(idxs))
    y_samples = alpha_real[idxs] + beta_real[idxs] * x_samples + eps_real
    
    data = pd.DataFrame(np.c_[idxs, x_samples, y_samples], columns=["idx", "x", "y"])
    data["idx"] = data["idx"].astype(int)
    
    _, axes = plt.subplots(2, M // 2, figsize=(8, 4))
    for i in range(M):
        d = data[data["idx"] == i]
        ax = axes[i // (M // 2), i % (M // 2)]
        ax.scatter(d["x"], d["y"], s=8, zorder=1)
        ax.grid(True, zorder=-1)
        ax.legend([f"group: {i}"], loc="upper right")
    plt.tight_layout()
    
    plt.figure(figsize=(5, 5))
    plt.scatter(data["x"], data["y"], s=8)
    plt.grid()     
    
    # #### 分层模型 #################################################################################
    
    # TODO