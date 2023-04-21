# -*- coding: utf-8 -*-
"""
Created on 2023/04/21 17:13:15

@File -> 2.6.2_化学位移分层模型.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 化学位移
"""

from joypy import joyplot
import seaborn as sns
import pandas as pd
import arviz as az
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    
    # #### 数据描述 #################################################################################
    
    data = pd.read_csv("data/chemical_shifts_theo_exp.csv")
    
    # 观测值y
    data["diff"] = data["theo"] - data["exp"]
    y_obs = data["diff"].values
    
    # 分组
    data.rename(columns={"aa": "group"}, inplace=True)
    group_labels = data["group"].unique().tolist()
    dim = len(group_labels)
    group_label_idx_map = dict(zip(group_labels, range(dim)))
    data["group_idx"] = data["group"].apply(lambda x: group_label_idx_map[x])
    group_idxs = data["group_idx"].values
    
    # 分组观察数据, 绘制山脊图
    plt.figure()

    joyplot(
        data=data[["diff", "group"]], 
        by="group",
        figsize=(12, 8)
    )
    plt.title("Ridgeline Plot of Diff Values in 19 Groups", fontsize=20)
    plt.xlabel("value")
    plt.show()
    
    # #### 贝叶斯估计建模 ###########################################################################
    
    with pm.Model() as model:
        # 超先验
        mu_mu = pm.Normal("mu_mu", mu=0, sigma=30)
        sigma_mu = pm.HalfNormal("sigma_mu", sigma=10)
        sigma_sigma = pm.HalfNormal("sigma_sigma", sigma=120)
        
        # 各组分布
        mu = pm.Normal("mu", mu=mu_mu, sigma=sigma_mu, shape=dim)
        sigma = pm.HalfNormal("sigma", sigma=sigma_sigma, shape=dim)
        y = pm.Normal("y", mu=mu[group_idxs], sigma=sigma[group_idxs], observed=y_obs)
        
        # 采样
        trace = pm.sample(1000, chains=1)
        
    pm.model_to_graphviz(model)
    
    # 总结后验
    az.plot_forest(trace)
    plt.tight_layout()
    
    