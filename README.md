FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.
=======================
**FixedEffectModel** is a Python Package designed and built by **Kuaishou DA ecology group**. It provides solutions for linear model with high dimensional fixed effects,including support for calculation in variance (robust variance and multi-way cluster variance), fixed effects, and standard error of fixed effects. It also supports model with instrument variables.

This version was forked version of the original [FixedEffectModel][OGR], which was altered to simplify it, add 
unit tests, fix runtime errors, and expand where relevant. This extension is more designed around running hundreds of thousands if not millions of most the same regression bar a single variable for genetic analysis.

# Installation
Install from pyPi via

```bash
pip install kdaHDFE
```

# Example

Unlike the original FixedEffectModel you **must** to use a formula akin to: 

'dependent variable ~ continuous variable|fixed_effect|clusters

If you want to cluster without FE you can just leave it blank as in BMI~Calories||PlaceOfBirth

```python
from kdaHDFE import HDFE
import pandas as pd

df = pd.read_csv('yourdata.csv')

# define model
formula = 'y~x+x2|id+firm|id+firm'

result1 = HDFE(df, formula = formula, robust=False, epsilon = 1e-8, ps_def= True,max_iter = 1e6)

```

# Requirements
- Python 3.6+
- Pandas and its dependencies (Numpy, etc.)
- Scipy and its dependencies
- statsmodels and its dependencies
- networkx

# Citation
If you use FixedEffectModel in your research, please consider citing the original author at:

Kuaishou DA Ecology. **FixedEffectModel: A Python Package for Linear Model with High Dimensional Fixed Effects.**<https://github.com/ksecology/FixedEffectModel>,2020.Version X

As well as this update if you did us this kdaHDFE extension at:
Baker, S. E. **kdaHDFE: A Python Package for Linear Model with High Dimensional Fixed Effects.**<https://github.com/sbaker-dev/kdaHDFE>,2021.Version X.


# Reference
[1] Simen Gaure(2019).  lfe: Linear Group Fixed Effects. R package. version:v2.8-5.1 URL:https://www.rdocumentation.org/packages/lfe/versions/2.8-5.1

[2] A Colin Cameron and Douglas L Miller. A practitioner’s guide to cluster-robust inference. Journal of human resources, 50(2):317–372, 2015.

[3] Simen Gaure. Ols with multiple high dimensional category variables. Computational Statistics & Data Analysis, 66:8–18, 2013.

[4] Douglas L Miller, A Colin Cameron, and Jonah Gelbach. Robust inference with multi-way clustering. Technical report, Working Paper, 2009.

[5] Jeffrey M Wooldridge. Econometric analysis of cross section and panel data. MIT press, 2010.


[OGR]: https://github.com/ksecology/FixedEffectModel
