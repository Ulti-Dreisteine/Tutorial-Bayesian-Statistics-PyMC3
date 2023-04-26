# -*- coding: utf-8 -*-
"""
Created on 2023/04/24 17:16:36

@File -> 3.2_相关系数估计.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 相关系数估计
"""

import pytensor
import numpy as np
import pymc as pm
import random
import arviz as az
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from setting import plt


def gen_data():
    np.random.seed(42)
    N = 100
    alpha_real = 2.5
    beta_real = 0.9
    eps = np.random.normal(0, 0.5, size=N)
    
    x = np.random.normal(10, 1, size=N)
    y_real = alpha_real + beta_real * x
    y_obs = y_real + eps
    
    return x, y_obs


def cal_pearson(x, y):
    return np.cov(np.c_[x, y].T)[0, 1] / (np.std(x) * np.std(y))
    

if __name__ == "__main__":
    
    # #### 生成样本 #################################################################################
    
    x_samples, y_samples = gen_data()
    pearson_real = cal_pearson(x_samples, y_samples)
    
    # #### 方法1: 使用自举法计算 #####################################################################
    
    coeffs = []
    rounds = 1000
    bt_size = 50
    idxs = np.arange(len(x_samples))
    
    for _ in range(rounds):
        idxs_bt = random.sample(list(idxs), bt_size)
        coeffs.append(cal_pearson(x_samples[idxs_bt], y_samples[idxs_bt]))
    
    # #### 方法2: 使用贝叶斯估计 #####################################################################
    
    # NOTE: 需要找到PearsonCorr (rho)、sigma_x、sigma_y与高斯分布的关系, 见BAP的Page 81
    data = np.c_[x_samples, y_samples]
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=[x_samples.mean(), y_samples.mean()], sigma=20, shape=2)
        sigma_x = pm.HalfNormal("sigma_x", sigma=20)
        sigma_y = pm.HalfNormal("sigma_y", sigma=20)
        rho = pm.Uniform("rho", -1., 1.)
        
        # NOTE: 两种代码都是可以的, 所以PyMC底层是基于pytensor的数据类型
        cov = pm.math.stack(
            ([sigma_x ** 2, sigma_x * sigma_y * rho], 
             [sigma_x * sigma_y * rho, sigma_y ** 2])
            )
        # cov = pytensor.tensor.stack(
        #     ([sigma_x ** 2, sigma_x * sigma_y * rho], 
        #      [sigma_x * sigma_y * rho, sigma_y ** 2])
        # )
        
        y_pred = pm.MvNormal("y", mu=mu, cov=cov, observed=data)
        
        trace = pm.sample(1000, chains=2)
    
    coeffs_real = trace["posterior"]["rho"].values
    
    az.plot_posterior({"PearsonCorr_Bootstrap": coeffs}, kind="hist")
    plt.xlim([0.7, 1.1])
    az.plot_posterior({"PearsonCorr_Bayesian": coeffs_real.flatten()}, kind="hist", range=(0.7, 1.1))
    plt.xlim([0.7, 1.1])
        