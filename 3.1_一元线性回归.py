import numpy as np
import arviz as az
import pymc as pm
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    
    # #### 生成样本 #################################################################################
    
    np.random.seed(42)
    N = 1000
    alpha_real = 2.5
    beta_real = 0.9
    eps = np.random.normal(0, 0.5, size=N)
    
    x = np.random.normal(10, 1, size=N)
    y_real = alpha_real + beta_real * x
    y_obs = y_real + eps
    
    # 画图
    plt.figure(figsize=(4, 4))
    plt.plot(x, y_real, "k")
    plt.scatter(x, y_obs, color="w", edgecolor="k")
    plt.show()
    
    # #### 贝叶斯参数估计 ############################################################################
    
    # 统计模型为: y ~ Norm(alpha + beta * x, sigma)
    # 其中, 待估计参数:
    #   alpha ~ Norm(mu=9, sigma=10)
    #   beta ~ Norm(mu=1, sigma=10)
    #   sigma ~ HalfNorm(sigma=10)
    
    with pm.Model() as linear_model:
        alpha = pm.Normal("alpha", 1, 10)
        beta = pm.Normal("beta", 1, 10)
        sigma = pm.HalfNormal("sigma", 10)
        # y_real = alpha + beta * x
        y_real = pm.Deterministic("y_real", alpha + beta * x)
        y = pm.Normal("y", y_real, sigma, observed=y_obs)
        
        trace = pm.sample(1000, chains=2, progressbar=False)
    
    pm.model_to_graphviz(linear_model)
    
    # 总结后验
    plt.figure()
    az.plot_trace(trace, figsize=(6, 4))
    # az.plot_forest(trace)
    plt.tight_layout()
    
    az.plot_pair(trace, var_names=["alpha", "beta"], figsize=(4, 4))
    
    # 回归结果置信区间
    plt.figure()
    y_real = trace["posterior"]["y_real"].values
    az.plot_hdi(x, y_real, hdi_prob=0.98, color="k")
    
    # 后验验证
    plt.figure()
    ppc = pm.sample_posterior_predictive(trace, linear_model, random_seed=42, progressbar=False)
    az.plot_ppc(ppc)