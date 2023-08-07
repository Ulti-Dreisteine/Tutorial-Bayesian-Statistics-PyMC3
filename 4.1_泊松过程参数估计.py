from pymc import DiscreteUniform, Exponential, Deterministic, Poisson, Uniform
import pymc as pm
import numpy as np
import arviz as az
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../" * 1))
sys.path.insert(0, BASE_DIR)

from setting import plt

if __name__ == "__main__":
    # 载入矿难观测数据
    disasters_array = np.array([
        4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
        3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
        2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
        1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
        0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
        3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
    idxs = np.arange(len(disasters_array))

    plt.figure(figsize=(8, 2))
    plt.scatter(np.arange(len(disasters_array)), disasters_array, s=6, c="w", edgecolors="k")
    plt.xlabel("number of year")
    plt.ylabel("number of events")
    
    with pm.Model() as model:
        # 分布转折点, 0至110的随机均匀离散分布
        switchpoint = DiscreteUniform("switch point", lower=0, upper=110)
        
        # 转折前后率参数
        mu_a = Exponential(r"$\mu_a$", 1.)
        mu_b = Exponential(r"$\mu_b$", 1.)
        
        rate = pm.math.switch(idxs < switchpoint, mu_a, mu_b)
        
        disasters = Poisson("disasters", mu=rate, observed=disasters_array)
    
        step = pm.Metropolis()  # 采用的迭代函数
        trace = pm.sample(1000, tune=2000, step=step, chains=2)
        
    pm.model_graph.model_to_graphviz(model)
    
    # 总结后验
    az.summary(trace, var_names=["switch point", r"$\mu_a$", r"$\mu_b$"])
    az.plot_trace(trace, figsize=(6, 4), compact=True, legend=False, combined=True)
    plt.tight_layout()