# Theory notes

Notation: $N$ assets, $T$ observations, weights $w$ with $\sum_i w_i = 1$, expected returns $\mu$, covariance matrix $\Sigma$.

## 1. Mean-variance optimisation

A portfolio has expected return $w^\top\mu$ and variance $w^\top\Sigma w$. Markowitz's problem is

$$\max_w\ w^\top\mu - \frac{\lambda}{2}\,w^\top\Sigma\,w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1 .$$

With a Lagrange multiplier $\gamma$ for the budget constraint, $w = \frac{1}{\lambda}\Sigma^{-1}(\mu - \gamma\mathbf{1})$. Two special cases:

$$w_{\text{min var}} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^\top\Sigma^{-1}\mathbf{1}}, \qquad w_{\text{max Sharpe}} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}{\mathbf{1}^\top\Sigma^{-1}(\mu - r_f\mathbf{1})} .$$

Both depend on the **inverse** covariance matrix. In the eigenbasis $\Sigma = \sum_k \lambda_k v_k v_k^\top$, the inverse is $\sum_k \lambda_k^{-1} v_k v_k^\top$: the optimiser loads most heavily on the directions with the smallest eigenvalues. Those are the worst estimated, which is why estimation noise matters so much.

Adding $w \ge 0$ or a cap on weights gives a quadratic programme with no closed form, solved numerically.

## 2. Estimation error

The sample covariance $S = \frac{1}{T}X^\top X$ (with $X$ the centred returns) is unbiased entry by entry, but its eigenvalues are biased: large ones are too large, small ones too small. If $T < N$, $S$ is singular and at least $N - T$ eigenvalues are exactly zero, so the "minimum-variance" portfolio appears to have no risk at all.

For the minimum-variance portfolio built from $S$ with Gaussian returns, the in-sample variance is on average $(1 - q)$ times the true optimum and the out-of-sample variance $1/(1 - q)$ times it, with $q = N/T$: the predicted and realised risks differ by a factor $(1-q)^{-2}$ in variance.

Mean returns are harder still: the standard error of an annualised mean over $T$ years is $\sigma/\sqrt{T}$, about 6% per year after 17 years for a stock with 26% volatility.

## 3. Random matrix theory

**Marchenko-Pastur law.** Let $X$ be a $T \times N$ matrix of independent entries with variance $\sigma^2$, and $q = N/T \le 1$. As $N, T \to \infty$ with $q$ fixed, the eigenvalues of $\frac{1}{T}X^\top X$ have density

$$\rho(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi q\,\sigma^2\lambda}, \qquad \lambda_\pm = \sigma^2(1 \pm \sqrt{q})^2 .$$

The true matrix is $\sigma^2 I$, with all eigenvalues equal, yet the estimated ones fill $[\lambda_-, \lambda_+]$. The width of the band grows with $q$.

**Reading a spectrum.** An eigenvalue of an empirical correlation matrix that lies above $\lambda_+$ cannot be explained by noise. For stocks, the largest is the market mode, whose eigenvector has all components of the same sign, and the next ones are sector modes. Because the market mode carries a share $\lambda_1/N$ of the total variance, the band for the remaining modes uses $\sigma^2 = 1 - \lambda_1/N$ (Laloux et al., 1999).

**Eigenvalue clipping.** Keep the eigenvalues above $\lambda_+$, replace the others by their mean (which preserves the trace), keep the eigenvectors, and rescale the diagonal to one. The cleaned matrix has no artificially small eigenvalue left for the optimiser to exploit.

**Inverse participation ratio.** For a normalised eigenvector $v$, $\text{IPR} = \sum_i v_i^4$, and $1/\text{IPR}$ is the effective number of assets involved: $N$ for a uniform vector, about $N/3$ for a random Gaussian one.

## 4. Linear shrinkage (Ledoit and Wolf, 2004)

The estimator is a convex combination of the sample matrix and a multiple of the identity,

$$\hat\Sigma = \delta\,m\,I + (1 - \delta)\,S, \qquad m = \frac{\text{tr}(S)}{N}.$$

$S$ has no bias but a large variance; $mI$ has a bias but almost no variance. The intensity that minimises the expected squared Frobenius error is

$$\delta^\ast = \frac{b^2}{d^2}, \qquad d^2 = \|S - mI\|^2, \qquad b^2 = \min\Big(d^2,\ \frac{1}{T^2}\sum_t \|x_t x_t^\top - S\|^2\Big),$$

where $d^2$ measures how far $S$ is from the target and $b^2$ estimates how noisy $S$ is. Both are computed from the data, so the estimator has no tuning parameter. In the eigenbasis, shrinkage moves every eigenvalue towards the mean, $\lambda_k \to \delta m + (1-\delta)\lambda_k$: it is a linear version of what clipping does non-linearly.

## 5. Other portfolio constructions

**Risk parity.** The contribution of asset $i$ to the portfolio variance is $w_i(\Sigma w)_i$. The equal-risk-contribution portfolio makes them all equal. It is the solution, after normalisation, of the convex problem

$$\min_{y > 0}\ \frac{1}{2}y^\top\Sigma\,y - \frac{1}{N}\sum_i \ln y_i ,$$

whose first-order condition is $y_i(\Sigma y)_i = 1/N$. It uses no expected returns and never concentrates.

**Black-Litterman.** If a reference portfolio $w_{\text{ref}}$ is optimal for risk aversion $\lambda$, expected returns must be $\pi = \lambda\Sigma w_{\text{ref}}$. Views are written $P\mu = Q + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \Omega)$, and the prior is $\mu \sim \mathcal{N}(\pi, \tau\Sigma)$. The posterior mean is

$$\mu_{\text{BL}} = \pi + \tau\Sigma P^\top\big(P\,\tau\Sigma\,P^\top + \Omega\big)^{-1}(Q - P\pi) .$$

Without views it is $\pi$, and the optimiser returns $w_{\text{ref}}$. Each view moves the portfolio in the direction of the view only, in proportion to its confidence.

## 6. Backtesting

At each rebalancing date $t$: estimate on $[t - T, t)$, compute weights, hold over $[t, t + h)$ while weights drift with prices, pay a cost proportional to the traded amount $\sum_i |w_i^{\text{new}} - w_i^{\text{drifted}}|$.

The diagnostic used throughout is the ratio of realised to predicted variance,

$$\frac{\text{mean over } t \text{ of realised variance on } [t, t+h)}{\text{mean over } t \text{ of } w_t^\top\hat\Sigma_t\,w_t} ,$$

which is close to 1 for a portfolio chosen without optimisation and above 1 when the optimiser has selected directions whose risk is under-estimated.
