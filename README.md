# Controlled Linear Regression

Consider the following linear model
```math
y = \beta_1 x_1 + \ldots + \beta_p x_p + \epsilon = \mathbf{x}^\text{T} \boldsymbol{\beta} +\epsilon,
```
and assume we observe the training data $`\{(y_n, x_{n,1}, \ldots, x_{n, p}),\ n = 1,\ldots, N\}`$, i.e. 
```math
	y_n = \beta_1 x_{n, 1} + \ldots + \beta_p x_{n, p} + \epsilon_n = \mathbf{x}_n^\text{T} \boldsymbol{\beta} +\epsilon_n, \quad n=1, \ldots, N,
```
where, given the design matrix $`\mathbf{X} \in \mathbb{R}^{N \times p}`$, the errors $`\{\epsilon_n, \ n=1,\ldots, N\}`$ are
- mean zero, i.e. $`\mathbb{E}[\epsilon_n | \mathbf{X}] = 0`$ for $n=1, \ldots, N$,
- homoscedastic, i.e. $`\mathbb{E}[\epsilon^2_n | \mathbf{X}] = \sigma^2 \in [0, \infty)`$ for $n=1, \ldots, N$, and
- uncorrelated, i.e. $`\mathbb{E}[\epsilon_n \epsilon_m|\mathbf{X}]=0`$ for $n,m=1, \ldots, N$ with $n\neq m$.

For any test observation $`\mathbf{x}_* = (x_{*,1}, \ldots, x_{*,p}) \in \mathbb{R}^p`$ we thus have the best possible prediction (in terms of MSE) for $y_*$ is
```math
	\mathbb{E}[ y_* | \mathbf{x}_*] = \mathbf{x}_*^\text{T} \boldsymbol{\beta} =: \hat{y}_*(\boldsymbol{\beta}),
```
and plugging in an estimator $\boldsymbol{\hat\beta}$ for $\boldsymbol{\beta}$ yields the predictor
```math
	\hat{y}_*(\boldsymbol{\hat\beta}) = \mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}.
```

Under the assumptions introduced above, the mean squared error of such a predictor can be decomposed as
```math
\mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}) - y_*)^2 \big| \mathbf{X}, \mathbf{x}_*\right] = \sigma^2 + \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right].
```
Under the assumptions discussed above minimizing the mean squared error of the predictor relative to the target $`y_*`$ is thus equivalent to minimizing the mean squared error of the predictor relative to the unfeasible best prediction $`\hat{y}_*(\boldsymbol{\beta})`$.

## Classic OLS estimation
The usual OLS estimator for $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p) \in \mathbb{R}^p$ is given by
```math
	\boldsymbol{\hat\beta}_{\mathbf{X}} = (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{y},
```
which, by the Gauss-Markov theorem, is known to be the best linear unbiased estimator (BLUE): for any $\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_p) \in \mathbb{R}^p$,
```math
	\mathbb{E}\left[(\boldsymbol{\lambda}^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}} - \boldsymbol{\lambda}^\text{T}\boldsymbol{\beta})^2 \big| \mathbf{X} \right] = \min_{\boldsymbol{\tilde{\beta}}_{\mathbf{X}} \in \text{LUE}(\mathbf{X}, \mathbf{y})} \mathbb{E}\left[ (\boldsymbol{\lambda}^\text{T} \boldsymbol{\tilde{\beta}}_{\mathbf{X}} - \boldsymbol{\lambda}^\text{T}\boldsymbol{\beta})\big| \mathbf{X} \right],
```
where $\text{LUE}(\mathbf{X}, \mathbf{y})$ is the set of all linear and unbiased estimator for $\boldsymbol{\beta}$, i.e.\ $`\boldsymbol{\tilde{\beta}}_{\mathbf{X}} = C(\mathbf{X}) \mathbf{y}`$ for some $\mathbf{X}$-measurable matrix $C(\mathbf{X})\in\mathbb{R}^{p\times N}$ and $`\mathbb{E}[\boldsymbol{\tilde{\beta}}_{\mathbf{X}}|\mathbf{X}] = \boldsymbol{\beta}`$. By applying the BLUE property, we can show that $`\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}})`$ is the best (in terms of mean squared error) predictor across all predictors formed from linear and unbiased estimators, i.e.
```math
  \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right]  \leq \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\tilde{\beta}}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right],
```
for all $\tilde{\beta}\in\text{LUE}(\mathbf{X}, \mathbf{y})$. Note that here we applied the mean-zero uncorrelated errors assumption.
	
## Controlled OLS estimation
Let us now assume we can additionally observe the "control" random variables $`\{\mathbf{z}_n = (z_{n,1}, \ldots, z_{n,k}) \in \mathbb{R}^k, \ n=1, \ldots, N\}`$. We shall assume the controls are available for training, i.e. when estimating $\boldsymbol{\beta}$, but for predicting, i.e.\ when forecasting $`y_*`$ we will have access to $`\mathbf{x}_*`$ but not to $`\mathbf{z}_*`$. 
The main rationale of controlled linear regression is to exploit as much information as possible in the training phase to make the coefficient estimators as precise as possible. 

Given the original design matrix $\mathbf{X}\in\mathbb{R}^{N\times p}$, we now assume the errors and the controls are jointly
- mean zero, i.e. $`\mathbb{E}[(\epsilon_n, \mathbf{z}_n) | \mathbf{X}] = \mathbf{0} \in\mathbb{R}^{k+1}`$ for $n=1, \ldots, N$,
- homoscedastic, i.e. $`\mathbb{E}[(\epsilon_n, \mathbf{z}_n)^{\otimes 2} | \mathbf{X}] = \Sigma \in \mathbb{R}^{(k+1) \times (k+1)}`$ for $n=1, \ldots, N$ and $\Sigma$ symmetric positive definite,
- uncorrelated, i.e. $`\mathbb{E}[(\epsilon_n, \mathbf{z}_n) {\otimes} (\epsilon_m, \mathbf{z}_m) | \mathbf{X}] = \mathbf{0} \in\mathbb{R}^{(k+1)\times(k+1)}`$ for $n,m=1, \ldots, N$ with $n\neq m$.

In what follows we partition $\Sigma$ as $\Sigma_{1,1} = \sigma^2 \in (0, \infty)$, $\Sigma_{1, 2:(k+1)} = \Sigma_{y, \mathbf{z}} = \Sigma_{2:(k+1), 1}^\text{T} \in \mathbb{R}^k$ and $\Sigma_{2:(k+1), 2:(k+1)} = \Sigma_{\mathbf{z}} \in \mathbb{R}^{k\times k}$.
Note that we can use the notation $\Sigma_{\mathbf{z}, Y}$ since for all $n=1, \ldots, N$ we have $`\text{Cov}(\mathbf{z}_n, Y_n | \mathbf{X}) = \mathbb{E}[ \mathbf{z}_n \epsilon_n | \mathbf{X}] =: \Sigma_{\mathbf{z}, Y}`$.
We use this notation, instead of the possibly more natural $\Sigma_{\mathbf{z}, \epsilon}$, as we believe that in application it is often easier to find highly correlated controls by considering the target rather than the residual. 

**Remark** Unless stated otherwise, we consider the original design matrix $\mathbf{X}$ and the new observation $\mathbf{x}_*\in\mathbb{R}^p$ to be fixed with random controls $\mathbf{Z}$ and errors $\mathbf{\epsilon}$.

### Theoretical optimal controlled OLS estimator 

As discussed above, fixing the design matrix $`\mathbf{X}\in\mathbb{R}^{N\times p}`$ and a test observation $`\mathbf{x}_* = (x_{*,1}, \ldots, x_{*,p}) \in \mathbb{R}^p`$, the predictor 
```math
\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) = \mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}},
```
is unbiased for the statistic $`\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) = \mathbf{x}_*^\text{T} \boldsymbol{\beta} \in \mathbb{R}`$. We hence introduce the control variate predictor
```math
\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}) = \hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) + \boldsymbol{\lambda}_1^\text{T} \mathbf{Z} \boldsymbol{\lambda}_2,
```
where $\mathbf{Z}\in\mathbb{R}^{N\times k}$ is the control design matrix while $`\boldsymbol{\lambda}_1 \in\mathbb{R}^{N}`$ and $`\boldsymbol{\lambda}_2 \in\mathbb{R}^{k}`$ are measurable in $\mathbf{X}$ and $`\mathbf{x}_*`$. 
Under the assumptions discussed above the controlled predictor can be shown to be unbiased and attains a minimum in variance when
```math
\boldsymbol{\lambda}_1^* = \mathbf{X} (\mathbf{X}^\text{T} \mathbf{X})^{-1}\mathbf{x}_*  \quad \text{and} \quad \boldsymbol{\lambda}_2^* = - \Sigma_{\mathbf{z}}^{-1} \Sigma_{\mathbf{z}, y}.
```

For any $`\mathbf{x}_* = (x_{*,1}, \ldots, x_{*,p}) \in \mathbb{R}^p`$ we thus have
```math
\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}, \mathbf{Z}, \boldsymbol{\lambda}^*) = \mathbf{x}_*^\text{T} (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{y} - \mathbf{x}_*^\text{T} (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{Z} \Sigma_{\mathbf{z}}^{-1}  \Sigma_{\mathbf{z}, y} = \mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} = \hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}),
```
yielding the unfeasible (in the sense that it depends on the unknown quantity $\Sigma$) estimator
```math
\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} =  (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} (\mathbf{y} - \mathbf{Z} \Sigma_{\mathbf{z}}^{-1}  \Sigma_{\mathbf{z}, y}) = \boldsymbol{\hat\beta}_{\mathbf{X}} - (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{Z} \Sigma_{\mathbf{z}}^{-1}  \Sigma_{\mathbf{z}, y}.
```

Note that, given $\mathbf{X}$, $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}`$ is unbiased (by mean zero property of the controls). Moreover, for any test observation $`\mathbf{x}_* = (x_{*,1}, \ldots, x_{*,p}) \in \mathbb{R}^p`$,
```math
  \text{Var}(\mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} | \mathbf{X}, \mathbf{x}_*) = (\sigma^2 - \Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1}  \Sigma_{\mathbf{z}, y})\mathbf{x}_*^\text{T} (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{x}_* \leq \sigma^2 \mathbf{x}_*^\text{T} (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{x}_* = \text{Var}(\mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}} | \mathbf{X}, \mathbf{x}_*),
```
with equality iff $\Sigma_{\mathbf{z}, y} = 0$. In other words, as long as the controls are correlated with the target, we obtain a better prediction by using $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}`$ instead of the OLS estimator $`\boldsymbol{\hat\beta}_{\mathbf{X}}`$ and the quality of the prediction increases as the correlation between $y$ and $\mathbf{z}$ increases. The variance reduction factor is constant across test observations and is given by 
```math
\frac{\text{Var}(\mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma} | \mathbf{X}, \mathbf{x}_*)}{\text{Var}(\mathbf{x}_*^\text{T} \boldsymbol{\hat\beta}_{\mathbf{X}} | \mathbf{X}, \mathbf{x}_*)} = \left(1 - \frac{\Sigma_{y, \mathbf{z}} \Sigma_{\mathbf{z}}^{-1}  \Sigma_{\mathbf{z}, y}}{\sigma^2}\right).
```
	
**Remark** Note that when $\mathbf{X} = 1 \in\mathbb{R}^{N}$, $`\mathbf{x}_* = 1`$ and $\mathbf{Z}\in\mathbb{R}^{N}$ we estimate $`\mu_* = \mathbb{E}[y]`$ with the OLS estimator $`\hat{\mu}_* = \hat{\beta} = \bar{\mathbf{y}}`$ and the simplest control variate estimator
  ```math
    \hat{\mu}_{*}^c = \bar{\mathbf{y}} - \frac{\mathrm{Cov}(y, z)}{\mathrm{Var}(z)} \bar{\mathbf{z}}.
  ```
  since $`\boldsymbol{\lambda}^*_1 = \frac{1}{N} 1\in\mathbb{R}^{N}`$ and $`\lambda^*_2 = - \frac{\mathrm{Cov}(y, z)}{\mathrm{Var}(z)}`$ which has reduced variance by a factor of $(1 - \mathrm{Corr}(y, z))$.
 
### Feasible optimal controlled OLS estimator 
In practice, the correlation matrix $\Sigma$ is usually unknown; hence, to make the estimator $\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}$ feasible, we need to estimate it. Under the assumptions discussed above, the most natural candidate is given by the sample estimates
```math
\hat{\Sigma}_{\mathbf{z}, y} = \frac{1}{N} \mathbf{Z}^\text{T} \mathbf{y} \quad \text{and} \quad \hat{\Sigma}_{\mathbf{z}} = \frac{1}{N} \mathbf{Z}^\text{T} \mathbf{Z},
```
yielding the feasible estimator
 ```math
\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \hat\Sigma} =  (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} (I - \mathbf{Z} (\mathbf{Z^\text{T}\mathbf{Z}})^{-1} \mathbf{Z}^\text{T}) \mathbf{y}.
```
This can be equivalently understood as regressing $\mathbf{y}$ on the control $\mathbf{Z}$ (i.e.\ projecting $\mathbf{y}$ onto the space spanned by $\mathbf{Z}$) and then regressing the residual onto $\mathbf{X}$. The resulting estimator will likely be biased (given $\mathbf{X}$), with its exact finite sample properties depending on the distribution of $(\mathbf{X}, \mathbf{Z}) | \boldsymbol{\epsilon}$.
	
When $\boldsymbol{\epsilon}$ depends linearly on $\mathbf{Z}$, i.e. 
```math
\boldsymbol{\epsilon} = \mathbf{Z} \boldsymbol{\alpha} + \boldsymbol{\eta},
```
we can write 
```math
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \mathbf{Z} \boldsymbol{\alpha} + \boldsymbol{\eta},
```
and hence we know that the joint OLS estimator obtained from the design matrix $(\mathbf{X}\ \mathbf{Z}) \in \mathbb{R}^{N \times (p+k)}$, i.e.
```math
\left(\begin{array}{c}
\hat{\boldsymbol{\beta}}_{\mathbf{X}, \mathbf{Z}} \\
\hat{\boldsymbol{\alpha}}_{\mathbf{X}, \mathbf{Z}}
\end{array}\right) = \left[
\left(\begin{array}{c}
		\mathbf{X}^\text{T} \\
		\mathbf{Z}^\text{T}
\end{array}\right) \left(\begin{array}{c}
		\mathbf{X} &
		\mathbf{Z}
\end{array}\right) \right]^{-1} \left(\begin{array}{c}
		\mathbf{X}^\text{T} \\
		\mathbf{Z}^\text{T}
\end{array}\right) \mathbf{y} = \left(\begin{array}{cc}
		\mathbf{X}^\text{T}\mathbf{X} & \mathbf{X}^\text{T}\mathbf{Z} \\
		\mathbf{Z}^\text{T}\mathbf{X} & \mathbf{Z}^\text{T}\mathbf{Z}
\end{array}\right)^{-1} \left(\begin{array}{c}
		\mathbf{X}^\text{T}\mathbf{y} \\
		\mathbf{Z}^\text{T}\mathbf{y}
\end{array}\right)
```
By using some simple algebraic manipulations for block matrices, we can extract the first $p$ entries of the joint OLS estimator, i.e.\ the estimator for $\boldsymbol{\beta}$, as
```math
\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}} = \boldsymbol{\hat{\beta}}_{\mathbf{X}, \mathbf{Z}, \hat{\Sigma}'} = \boldsymbol{\hat{\beta}} - (\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{X}^{\text{T}} \mathbf{Z} (\mathbf{Z}^\text{T}\mathbf{Z} - \mathbf{Z}^\text{T} \mathbf{X} (\mathbf{X}^\text{T}\mathbf{X})^{-1} \mathbf{X}^\text{T}\mathbf{Z})^{-1} (\mathbf{Z}^\text{T} \mathbf{y} - \mathbf{Z}^\text{T} \mathbf{X} (\mathbf{X}^\text{T}\mathbf{X})^{-1} \mathbf{X}^\text{T} \mathbf{y}),
```
i.e. $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \Sigma}`$ with $\Sigma$ estimated by
```math
\hat{\Sigma}'_{\mathbf{z}, y} = \frac{1}{N}\,  \mathbf{Z}^\text{T} (I - \mathbf{X} (\mathbf{X}^\text{T}\mathbf{X})^{-1} \mathbf{X}^\text{T}) \mathbf{y} \quad \text{and} \quad \hat{\Sigma}'_{\mathbf{z}} = \frac{1}{N}\,  \mathbf{Z}^\text{T}(I - \mathbf{X} (\mathbf{X}^\text{T}\mathbf{X})^{-1} \mathbf{X}^\text{T})\mathbf{Z}.
```
Note these are the covariance and variance estimators obtained when projecting $\mathbf{Z}$ onto the orthogonal complement of the space spanned by $\mathbf{X}$. Under the assumptions introduced above, these are also unbiased for $\Sigma_{\mathbf{z}, y}$ and $\Sigma_{\mathbf{z}}$. This provides a second feasible controlled estimator for $\boldsymbol{\beta}$.

### Theoretical properties
Assume $\boldsymbol{\epsilon}$ depends linearly on $\mathbf{Z}$, i.e. 
```math
\boldsymbol{\epsilon} = \mathbf{Z} \boldsymbol{\alpha} + \boldsymbol{\eta}.
```
If we fix both $\mathbf{X}$ and $\mathbf{Z}$ and assume $\boldsymbol{\eta}$ satisfies the Gauss-Markov conditions, then the joint OLS estimator is the BLUE for $(\boldsymbol{\beta}, \boldsymbol{\alpha})$. Extending the classic OLS estimator $`\boldsymbol{\hat{\beta}}_{\mathbf{X}}`$ to $`(\boldsymbol{\hat{\beta}}_{\mathbf{X}}, \boldsymbol{\hat{\alpha}}_{\mathbf{X}, \mathbf{Z}})`$ we obtain another linear and unbiased estimator for $(\boldsymbol{\beta}, \boldsymbol{\alpha})$. It follows from the Gauss-Markov theorem that for any $`\mathbf{x}_*\in\mathbb{R}^p`$, setting $`\boldsymbol{\lambda} = (\mathbf{x}_*, \mathbb{E}[\mathbf{z}_* | \mathbf{X}, \mathbf{Z}, \mathbf{x}_* ] = \boldsymbol{0})`$, one has 
```math
  \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{Z}, \mathbf{x}_*\right] \leq \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{Z}, \mathbf{x}_*\right],
```
and hence, by the tower property of conditional expectation,
```math
  \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right] \leq \mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right].
```
Using the forms of the variances of $\boldsymbol{\hat\beta_{\mathbf{X}}}$ and $\boldsymbol{\hat\beta_{\mathbf{X}, \mathbf{Z}}}$ we can compute the MSE reduction factor as
```math
\frac{\mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right]}{\mathbb{E}\left[(\hat{y}_*(\boldsymbol{\hat\beta}_{\mathbf{X}}) - \hat{y}_*(\boldsymbol{\beta}))^2 \big| \mathbf{X}, \mathbf{x}_*\right]} = \frac{\mathbf{x}_*^\text{T}\mathbb{E}[(\mathbf{X}^\text{T} (I - \mathbf{Z}(\mathbf{Z}^\text{T}\mathbf{Z})^{-1}\mathbf{Z}^\text{T})\mathbf{X})^{-1}\big| \mathbf{X}] \mathbf{x}_*}{\mathbf{x}_*^\text{T}(\mathbf{X}^\text{T} \mathbf{X})^{-1} \mathbf{x}_*},
```
which we note does not depend on $\sigma^2$. 

Given $\mathbf{X}$ and $\mathbf{Z}$, augmenting the first feasible controlled estimator $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \hat{\Sigma}}`$ to an estimator for $(\boldsymbol{\beta}, \boldsymbol{\alpha})$, yields a linear but biased (at least in the $\boldsymbol{\beta}$ components) estimator. Whether $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \hat{\Sigma}}`$ or $\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}}$ yields a better predictor cannot thus be deduced from the Gauss-Markov theorem. As we will see in the numerical experiments discussed in the next section, which estimator performs better depends on the properties of the data generating process.

### Empirical properties
The code in this github repo provides an implementation and comparsion of the three estimators:
- classic OLS estimator $`\boldsymbol{\hat\beta}_{\mathbf{X}}`$;
- first feasible control estimator $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}, \hat{\Sigma}}`$;
- second feasible control (joint OLS) estimator $`\boldsymbol{\hat\beta}_{\mathbf{X}, \mathbf{Z}}`$;

under a variety of assumptions on the data generating process (including the case where $\boldsymbol{\epsilon}$ does not depend linearly on $\mathbf{Z}$).
