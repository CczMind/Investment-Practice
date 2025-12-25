# NUMERICAL AND SIMULATION TECHNIQUES IN FINANCE | GROUP PROJECT

## Instructions
Submit your Python program on iSpace into one zipped file containing:
- code files (.py or .ipynb files)
- a final report containing：
    - your written answers,
    - the user manual of your software, 
    - and the job allocation between the team members.

## Project description: pricing of barrier options by MonteCarlo simulations

### Barrier options

Barrier options are a class of exotic financial options whose payoff depends onwhether the underlying asset price touched a predefined barrier price during thelifetime of the option.

Classical examples include:

#### Knock-out barrier options:

- Up-and-out call option - $P_{T}=(S_{T}-K)^{+}\mathbf{1}\left\{\max_{0\leq t\leq T}S_{t}<B\right\}$, $0<S_0<B$

- Down-and-out call option - $P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$, $0<B<S_0$

- Up-and-out put option - $P_T=(K-S_T)^+\mathbf{1}\left\{\max_{0\leq t\leq T}S_t<B\right\}$, $0<S_0<B$

- Down-and-out put option - $P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$, $0<B<S_0$

#### Knock-in barrier options

- Up-and-in call option - $P_T=(S_T-K)^+\mathbf{1}\left\{\max_{0\leq t\leq T}S_t\geq B\right\}$, $0<S_0<B$

- Down-and-in call option - $P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t\leq B\right\}$, $0<B<S_0$

- Up-and-in put option - $P_{T}=(K-S_{T})^{+}\mathbf{1}\left\{\max_{0\leq t\leq T}S_{t}\geq B\right\}$, $0<S_0<B$

- Down-and-in put option - $P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t\leq B\right\}$, $0<B<S_0$

Here,
- $P_{T}\geq0$ is the final cash-flow of the option, received at time $T$ (the maturity of
the option), 
- $S_{T}\geq0$ is the price of the underlying asset at time $T$
- $K > 0$ is the
strike price of the option,
- $B > 0$ is the barrier price

##### 1. Monte Carlo pricing under Black-Scholes model

Suppose that the underlying asset price process follows a **geometric Brownian motion** with **constant drift** $r > 0$ and **constant volatility** $\sigma > 0$ under the risk-neutral pricing measure.

1. Describe how to estimate the initial premiums of these eight barrier options by Monte Carlo simulations using the Euler scheme on a uniform time grid with $N$ equal time intervals between $t = 0$ and the option maturity $t = T$

2. Implement these eight Monte Carlo estimators in Python. For your numerical tests, set $r = 0.05$, $\sigma = 0.20$, $T = 1$, $S_0 = K = 100$, and try a few different values of $B$.

3. Sensitivity analysis with respect to the number $M$ of Monte Carlo simulations: plot the estimated option premiums and $99\%$ (asymptotic) confidence intervals with respect to $M$ for each of these eight options (set $B$ to an "interesting" value based
on your previous tests).

4. Sensitivity analysis with respect to the number $N$ of time steps: fix $M$ to a large value ensuring numerical convergence (based on your results from the above sensitivity analysis), then plot the estimated option premiums with respect to $N$ for each of these eight options (set $B$ to the same values as before).

##### 2. Monte Carlo pricing under stochastic volatility model

Suppose now that the variance of the underlying asset price is not constant, but instead follows the 3/2 stochastic volatility model under the risk-neutral pricing measure, defined by

$$
\Large 
\begin{aligned}
dS_{t} & = rS_{t} dt + \sqrt{V_{t}}S_{t}dW_{t} \\
dV_{t} & = \kappa(\theta V_t - V_t^2) dt + \lambda V_t^{3/2} dB_t \\
d\langle W,B\rangle_{t} & = \rho dt
\end{aligned}
$$

In particular, the Brownian motions of the asset price and its variance are correlated, with correlation parameter $\rho\in(-1,1)$.

1. Redo all the questions from the previous section, with the constant volatility $\sigma$ replaced by the 3/2 stochastic volatility model. For the numerical experiments, use the same numerical parameters as in the previous section, as well as $V_0=\theta=0.20$, $\kappa=0.20$, $\lambda=0.67$, $\rho=-0.5$

2. Sensitivity analysis with respect to the volatility-of-volatility: fix $M$ to a large value ensuring numerical convergence, then plot the estimated option premiums with respect to $\lambda$ for each of these eight options (set $B$ to the same values as before). Conclude about the effect of stochastic volatility on the premiums of barrier options.

3. Sensitivity analysis with respect to the correlation: fix $M$ to a large value ensuring numerical convergence, then plot the estimated option premiums with respect to $\rho$ for each of these eight options (set $B$ to the same values as before). Conclude about the effect of the correlation between price and volatility on the premiums of barrier options.

---

Geometric Brownian motion: SDE: $dS_t=\mu S_tdt+\sigma S_tdW_t$, $\mu(t,x) = \mu x$, $\sigma(t,x) = \sigma x$

---
