import arviz as az
import numpy as np
from scipy import stats

np.random.seed(1)

az.style.use("arviz-darkgrid")
az.plot_posterior({"$\\theta$": stats.beta.rvs(5, 11, size=10000)})