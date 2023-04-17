### Tutorial_Bayesian_Statistics_PyMC3
基于PyMC3的贝叶斯统计入门

### 1. 准备工作

本项目主要依赖的Python包有:

```
pymc
numpy
scipy
pandas
matplotlib
arviz
seaborn
```

#### 1.1 PyMC3安装

以下安装在python 3.8及以下版本测试有效, 更高python版本待测试:

**注意: 在Anaconda Prompt(管理员权限打开)中执行以下命令**

```
pip install pymc3 -i https://mirrors.aliyun.com/pypi/simple/
conda install numpy scipy mkl-service libpython m2w64-toolchain
conda install -c conda-forge blas
conda install -c conda-forge python-graphviz
```

**注意: 若在import pymc3时出现如下warning**

> WARNING (theano.tensor.blas): Using NumPy C-API based implementation for BLAS functions

则进行安装:

```
conda install mkl
conda install mkl-service
conda install blas
conda install -c conda-forge python-graphviz
```

然后在"C:\Users\Administrator等用户名"下新建".theanorc.txt"文件，里面输入:

```
[blas]
ldflags=-lmkl_rt
```

或

```
[blas]
ldflags=-lblas
```

<font color="red">安装完毕后可运行test.pymc_running_test.ipynb对PyMC进行运行测试</font >

#### 1.2 ArviZ安装

```
pip install arviz -i https://mirrors.aliyun.com/pypi/simple/
```

### 参考

* https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers