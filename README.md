# Tutorial_Bayesian_Statistics_PyMC3
基于PyMC3的贝叶斯统计入门

## 1. 准备工作

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

### 1.1 PyMC3安装

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

<font color="red">安装完毕后可运行test/pymc_running_test.ipynb对PyMC进行运行测试</font >

ArviZ安装

```
pip install arviz -i https://mirrors.aliyun.com/pypi/simple/
```

<font color="red">安装完毕后可运行test/arviz_running_test.py对Arviz进行运行测试</font >

### PyMC 4.0及以上版本安装

准备: Conda换源, 在Anaconda Prompt中运行:

```bash
conda config --set channel_priority strict 
conda config --set show_channel_urls yes
```

记事本打开"C:\Users\\**用户名**\\.condarc", 输入以下内容

```bash
channels:
  - conda-forge
  - defaults
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

Step 1: 安装环境

```bash
conda create -c conda-forge -n pymc_env python=3.8 "pymc>=4"  # 实际pymc==5.3.0
conda activate pymc_env
```

Step 2: 安装PyMC及相关包

```bash
conda install pymc
conda install ipykernel
conda install -c conda-forge python-graphviz
```

Step 3: 安装Jax

NOTE: 目前Windows上的PyMC还不支持Jax

## 关键点

* pm.Normal()等生成的随机参数为pytensor类型, 需要代入对应的pm.math.stack()和pytensor.tensor.stack()进行处理. 也就是说, 后续所有处理都必须在tensor前提下进行

## 参考

* PyMC官网: https://www.pymc.io/welcome.html
* 贝叶斯估计: https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers
