Geometric Brownian motion: SDE: $dS_t=\mu S_tdt+\sigma S_tdW_t$, $\mu(t,x) = \mu x$, $\sigma(t,x) = \sigma x$
---
NUMERICAL AND SIMULATION TECHNIQUES IN FINANCE | GROUP PROJECT

# Instructions
Submit your Python program on iSpace into one zipped file containing:
- code files (.py or .ipynb files)
- a final report containing：
    - your written answers,
    - the user manual of your software, 
    - and the job allocation between the team members.

# Project description: pricing of barrier options by MonteCarlo simulations

## Barrier options

Barrier options are a class of exotic financial options whose payoff depends onwhether the underlying asset price touched a predefined barrier price during thelifetime of the option.

Classical examples include:

### Knock-out barrier options:

- Up-and-out call option - $P_{T}=(S_{T}-K)^{+}\mathbf{1}\left\{\max_{0\leq t\leq T}S_{t}<B\right\}$, $0<S_0<B$

- Down-and-out call option - $P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$, $0<B<S_0$

- Up-and-out put option - $P_T=(K-S_T)^+\mathbf{1}\left\{\max_{0\leq t\leq T}S_t<B\right\}$, $0<S_0<B$

- Down-and-out put option - $P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$, $0<B<S_0$

### Knock-in barrier options

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

#### 1. Monte Carlo pricing under Black-Scholes model

Suppose that the underlying asset price process follows a **geometric Brownian motion** with **constant drift** $r > 0$ and **constant volatility** $\sigma > 0$ under the risk-neutral pricing measure.

1. Describe how to estimate the initial premiums of these eight barrier options by Monte Carlo simulations using the Euler scheme on a uniform time grid with $N$ equal time intervals between $t = 0$ and the option maturity $t = T$

2. Implement these eight Monte Carlo estimators in Python. For your numerical tests, set $r = 0.05$, $\sigma = 0.20$, $T = 1$, $S_0 = K = 100$, and try a few different values of $B$.

3. Sensitivity analysis with respect to the number $M$ of Monte Carlo simulations: plot the estimated option premiums and $99\%$ (asymptotic) confidence intervals with respect to $M$ for each of these eight options (set $B$ to an "interesting" value based
on your previous tests).

4. Sensitivity analysis with respect to the number $N$ of time steps: fix $M$ to a large value ensuring numerical convergence (based on your results from the above sensitivity analysis), then plot the estimated option premiums with respect to $N$ for each of these eight options (set $B$ to the same values as before).

#### 2. Monte Carlo pricing under stochastic volatility model

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
请仔细阅读、理解以上小组项目要求。
---
# 注意：
以上的第 1 问中（“1. Monte Carlo pricing under Black-Scholes model”）我已经完成"上涨敲出看涨期权（Up-and-Out Call）"的全部代码了。
---
# 以下是已经完成的第 1 问的全部代码：
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tqdm
import warnings
warnings.filterwarnings('ignore')

def simulate_gbm_paths_euler(S0, r, sigma, T, N, M):
    """
    使用欧拉离散化方法模拟几何布朗运动（GBM）的路径
    
    参数:
    S0: 初始资产价格
    r: 无风险利率
    sigma: 波动率
    T: 总时间长度（年）
    N: 时间步数
    M: 模拟的路径数量
    
    返回:
    t: 时间点数组，形状为 (N+1,)
    S: 资产价格路径矩阵，形状为 (N+1, M)
    """
    t = np.linspace(0, T, N + 1)
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    
    S = np.zeros((N + 1, M))
    S[0, :] = S0
    
    Z = norm.rvs(size = (N, M))
    
    for n in tqdm.tqdm(range(N), desc="Simulating GBM paths", unit="step"):
        S[n + 1, :] = S[n, :] * (1 + r * dt + sigma * sqrt_dt * Z[n, :])
        
    return t, S

def compute_up_and_out_call_payoff(S_path, K, B):
    """
    计算上涨敲出看涨期权的payoff
    
    参数:
    S_path: 资产价格路径矩阵，形状为 (N+1, M)
    K: 执行价格
    B: 障碍价格（需满足 S0 < B）
    
    返回:
    payoff: 每条路径的payoff，形状为 (M,)
    """
    M = S_path.shape[1]
    
    payoffs = np.zeros(M)
    for i in range(M):                      # 检查每条路径是否触碰障碍
        max_price = np.max(S_path[:, i])    # 路径上的最高价格
        final_price = S_path[-1, i]         # 到期价格
        
        if max_price < B:                   # 未触碰障碍
            payoffs[i] = max(final_price - K, 0)
        else:                               # 触碰障碍，期权失效
            payoffs[i] = 0
    
    # # 向量化
    # max_prices = np.max(S_path, axis = 0)
    # final_prices = S_path[-1, :]
    # payoffs = np.where(max_prices < B, np.maximum(final_prices - K, 0), 0)
    
    return payoffs

def price_up_and_out_call_monte_carlo(S0, K, B, T, r, sigma, M, N, confidence_level = 0.99):
    """
    使用蒙特卡洛方法定价上涨敲出看涨期权
    
    参数:
    S0: 初始资产价格
    K: 执行价格
    B: 障碍价格（需满足 S0 < B）重要！！！
    T: 到期时间（年）
    r: 无风险利率
    sigma: 波动率
    M: 蒙特卡洛模拟路径数
    N: 时间步数
    confidence_level: 置信水平（默认0.99）
    
    返回:
    dict: 包含价格估计、置信区间等信息的字典
    """
    # 1. 模拟GBM路径
    t, S_paths = simulate_gbm_paths_euler(S0, r, sigma, T, N, M)
    
    # 2. 计算每条路径的payoff
    payoffs = compute_up_and_out_call_payoff(S_paths, K, B)
    
    # 3. 贴现payoff
    discount_factor = np.exp(-r * T)
    discounted_payoffs = payoffs * discount_factor
    
    # 4. 计算统计量
    price_estimate = np.mean(discounted_payoffs)
    price_std = np.std(discounted_payoffs, ddof = 1)
    price_se = price_std / np.sqrt(M)
    
    # 5. 计算置信区间
    z_value = norm.ppf((1 + confidence_level) / 2)
    ci_half_width = z_value * price_se
    ci_lower = price_estimate - ci_half_width
    ci_upper = price_estimate + ci_half_width
    
    return {
        'price_estimate': price_estimate,
        'price_std': price_std,
        'price_se': price_se,
        'confidence_interval': (ci_lower, ci_upper),
        'confidence_level': confidence_level,
        'discounted_payoffs': discounted_payoffs,
        'paths': S_paths,
        'time_grid': t
    }

def plot_paths_with_barrier(t, paths, B, num_paths_to_plot = 50):
    """
    绘制GBM路径并标出障碍水平
    
    参数:
    t: 时间网格
    paths: GBM路径矩阵
    B: 障碍价格
    num_paths_to_plot: 要绘制的路径数量
    """
    M = paths.shape[1]
    num_paths = min(num_paths_to_plot, M)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制路径
    for i in range(num_paths):
        max_price = np.max(paths[:, i])
        color = 'red' if max_price >= B else 'blue'
        alpha = 0.3 if max_price >= B else 0.1
        plt.plot(t, paths[:, i], color=color, alpha=alpha, linewidth=0.5)
    
    # 绘制障碍线
    plt.axhline(y=B, color='green', linestyle='--', linewidth=2, label=f'Barrier B = {B}')
    
    # 绘制执行价格线
    plt.axhline(y=paths[0, 0], color='black', linestyle=':', linewidth=1.5, label=f'Initial Price S0 = {paths[0, 0]}')
    
    plt.xlabel('Time (years)', fontsize=12)
    plt.ylabel('Asset Price', fontsize=12)
    plt.title(f'GBM Paths with Barrier (B={B}) - Red: Knocked Out, Blue: Active', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_sensitivity_to_M(S0, K, B, T, r, sigma, N, M_list):
    """
    分析期权价格对蒙特卡洛模拟路径数M的敏感性
    
    参数:
    M_list: 模拟路径数的列表
    其他参数同 price_up_and_out_call_monte_carlo
    """
    price_estimates = []
    ci_lowers = []
    ci_uppers = []
    standard_errors = []
    
    for M in tqdm.tqdm(M_list, desc="Sensitivity to M"):
        results = price_up_and_out_call_monte_carlo(S0, K, B, T, r, sigma, M, N)
        price_estimates.append(results['price_estimate'])
        ci_lowers.append(results['confidence_interval'][0])
        ci_uppers.append(results['confidence_interval'][1])
        standard_errors.append(results['price_se'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 图1：价格和置信区间
    ax1.plot(M_list, price_estimates, 'b-', linewidth=2, label='Price Estimate')
    ax1.fill_between(M_list, ci_lowers, ci_uppers, alpha=0.2, color='blue', label='99% CI')
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Monte Carlo Simulations (M)', fontsize=12)
    ax1.set_ylabel('Option Premium', fontsize=12)
    ax1.set_title('Price Estimate vs. M (log scale)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2：标准误差
    ax2.plot(M_list, standard_errors, 'r-', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Monte Carlo Simulations (M)', fontsize=12)
    ax2.set_ylabel('Standard Error (log scale)', fontsize=12)
    ax2.set_title('Standard Error vs. M', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'M_values': M_list,
        'price_estimates': price_estimates,
        'confidence_intervals': list(zip(ci_lowers, ci_uppers)),
        'standard_errors': standard_errors
    }

def analyze_sensitivity_to_N(S0, K, B, T, r, sigma, M, N_list):
    """
    分析期权价格对时间步数N的敏感性
    
    参数:
    N_list: 时间步数的列表
    其他参数同 price_up_and_out_call_monte_carlo
    """
    price_estimates = []
    
    for N in tqdm.tqdm(N_list, desc="Sensitivity to N"):
        results = price_up_and_out_call_monte_carlo(S0, K, B, T, r, sigma, M, N)
        price_estimates.append(results['price_estimate'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_list, price_estimates, 'g-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Time Steps (N)', fontsize=12)
    plt.ylabel('Option Premium', fontsize=12)
    plt.title(f'Sensitivity to Discretization (M={M})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'N_values': N_list,
        'price_estimates': price_estimates
    }

def run_comprehensive_analysis():
    """
    运行综合测试分析
    """
    # 基础参数设置（与项目要求一致）
    S0 = 100
    K = 100
    B = 120  # 障碍价格（S0 < B）
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    print("=" * 60)
    print("UP-AND-OUT CALL OPTION - MONTE CARLO PRICING ANALYSIS")
    print("=" * 60)
    print(f"Parameters: S0={S0}, K={K}, B={B}, T={T}, r={r}, σ={sigma}")
    print()
    
    # 1. 单次定价示例
    print("1. Single Pricing Example:")
    print("-" * 40)
    
    M = 10000
    N = 252  # 假设交易日数
    
    results = price_up_and_out_call_monte_carlo(S0, K, B, T, r, sigma, M, N)
    
    print(f"   Monte Carlo Paths: M = {M}")
    print(f"   Time Steps: N = {N}")
    print(f"   Price Estimate: {results['price_estimate']:.4f}")
    print(f"   Standard Error: {results['price_se']:.6f}")
    print(f"   {int(results['confidence_level']*100)}% Confidence Interval: [{results['confidence_interval'][0]:.4f}, {results['confidence_interval'][1]:.4f}]")
    print()
    
    # 2. 绘制路径示例
    print("2. Visualizing Sample Paths:")
    print("-" * 40)
    
    # 用小M快速生成路径用于可视化
    t, sample_paths = simulate_gbm_paths_euler(S0, r, sigma, T, 100, 200)
    plot_paths_with_barrier(t, sample_paths, B, num_paths_to_plot=100)
    
    # 3. 对M的敏感性分析
    print("3. Sensitivity Analysis to M (Number of Simulations):")
    print("-" * 40)
    
    M_list = [100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    M_sensitivity = analyze_sensitivity_to_M(S0, K, B, T, r, sigma, N, M_list)
    
    # 4. 对N的敏感性分析
    print("4. Sensitivity Analysis to N (Number of Time Steps):")
    print("-" * 40)
    
    N_list = [10, 25, 50, 100, 252, 500, 1000]
    N_sensitivity = analyze_sensitivity_to_N(S0, K, B, T, r, sigma, M, N_list)
    
    print()
    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    return {
        'single_pricing': results,
        'M_sensitivity': M_sensitivity,
        'N_sensitivity': N_sensitivity,
    }

if __name__ == "__main__":
    analysis_results = run_comprehensive_analysis()
---
# 以下是一些额外的内容、代码示例：
---
### 1.2 Simulate a Brownian motion

In order to simulate a stochastic process, we first need to perform a time discretization. 

Introduce the time grid $\mathcal{T}=\{t_0,t_1,\ldots,t_N\}$ such that $t_n=n\times\Delta_t$ where $\Delta_t:=\frac{T}{N}$.

Drawing a Monte Carlo path from a Brownian motion can be achieved as follows:

1. Simulate $N$ independent draws $G_n, n = 1, ..., N$ from a standard Gaussian variable

2. Set $W_{t_0} = 0$ and for every $n = 0, 1, ..., N - 1$, define:

$$
W_{t_{n+1}}=W_{t_n}+\sqrt{\Delta_t}G_n
$$
---
### 1.3 Simulating multiple BM paths

import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns

def generate_brownian_motion(T = 1.0, N = 100, M = 10_000, init = 0.0) -> tuple:
    """
    生成布朗运动路径
    
    参数:
    T: 总时间长度
    N: 时间步数
    M: 模拟的路径数量
    init: 初始值 (默认值: 0.0)
    
    返回:
    t: 时间点数组，形状为 (N+1,)
    W: 布朗运动路径矩阵，形状为 (N+1, M)
    """
    dt = T / N
    sqrtdt = np.sqrt(dt)
    
    # 初始化存储矩阵：行代表时间，列代表不同路径
    W = np.zeros([N+1, M])
    W[0, :] = init
    
    for n in tqdm.tqdm(range(N), desc = "Generating Brownian paths", unit = "step"):
        W[n+1, :] = W[n, :] + sqrtdt * norm.rvs(size = M)
    
    t = np.linspace(0, T, N + 1)
    
    return (t, W)

def plot_comprehensive_brownian_analysis(t, W, T = 1.0, num_paths = 5, n_band = 10):
    """
    绘制随机运动的综合分析图
    
    参数:
    t: 时间点数组
    W: 布朗运动路径矩阵
    T: 总时间长度
    num_paths: 显示的样本路径数量
    n_band: 扇形图的颜色带数量
    """
    fig = plt.figure(figsize = (20, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)
    
    # 样本路径图
    for i in range(min(num_paths, W.shape[1])):
        ax1.plot(t, W[:, i], linewidth=2)
    ax1.set_xlim([0, T])
    ax1.set_xlabel('Time (t)', fontsize = 14)
    ax1.set_ylabel('Brownian Motion W(t)', fontsize = 14)
    ax1.set_title(f'Sample Paths of Brownian Motion ({min(num_paths, W.shape[1])} paths)', fontsize = 16)
    ax1.grid(True, alpha=0.3)
    
    # 扇形图
    col1 = sns.color_palette("PuBuGn_r", n_band // 2)
    q_val = np.linspace(0.0, 1.0, n_band + 1)
    q_val[0] = 0.01
    q_val[-1] = 0.99
    Q = np.quantile(W, q_val, axis = 1)
    for i in range(n_band):
        col_idx = np.abs(int(i - n_band // 2 + 0.5))
        ax2.fill_between(t, Q[i+1, :], Q[i, :], color = col1[col_idx], zorder = 20)
    ax2.set_xlim([0, T])
    ax2.set_xlabel('Time (t)', fontsize = 14)
    ax2.set_ylabel('Brownian Motion W(t)', fontsize = 14)
    ax2.set_title("Brownian Motion Quantile Fan Chart", fontsize = 16)
    ax2.grid(True, alpha = 0.3)
    
    # 路径叠加扇形图
    col2 = sns.color_palette("mako_r", n_band // 2)
    M_display = min(num_paths, W.shape[1])
    for i in range(M_display):
        ax3.plot(t, W[:, i], color = "black", linewidth = 2, alpha = 1, zorder = 1)     # zorder 值越大的元素，会被绘制在越上层
    for i in range(n_band):
        col_idx = np.abs(int(i - n_band // 2 + 0.5))
        ax3.fill_between(t, Q[i+1, :], Q[i, :], color = col2[col_idx], zorder = 20, alpha = 1)
    ax3.set_xlim([0, T])
    ax3.set_xlabel('Time (t)', fontsize = 14)
    ax3.set_ylabel('Brownian Motion W(t)', fontsize = 14)
    ax3.set_title(f"Paths and Fan Chart Overlay Analysis ({M_display} paths)", fontsize = 16, loc = "left")
    ax3.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    # plt.savefig("comprehensive_brownian_analysis.svg", format = "svg")
    plt.show()

T, N, M, init = 1.0, 100, 50_000, 0.0
num_paths, num_band = 1000, 20
t, X = generate_brownian_motion(T = T, N = N, M = M, init = init)
plot_comprehensive_brownian_analysis(t, X, T = T, num_paths = num_paths, n_band = num_band)
---
### 1.4 Simulate an arithmetic Brownian motion

An $\color{yellow}\text{arithmetic Brownian motion}$ $X=(X_t)_{0\leq t\leq T}$ is the sum of a Brownian motion and a linear drift:

$$
X_t=X_0+\mu t+\sigma W_t
$$
- $\mu$ - drift parameter
- $\sigma$ - volatility parameter

1. Simulate the Brownian motion $W$ as explained before;
2. Multiply these paths by the constant $\sigma$;
3. Add the affine function $X_0 + \mu t$ to each path.

---
### 1.5 Geometric Brownian motion

A stochastic process $S=(S_t)_{0\leq t\leq T}$ is said to follow a $\color{yellow}\text{geometric Brownian motion}$ (GBM) if it is satisfies the following **Stochastic Differential Equation** (SDE):

$$
dS_t=\mu S_tdt+\sigma S_tdW_t
$$

- Starting from a fixed value $S_0 > 0$
- $\mu$ - Drift parameter
- $\sigma$ - Volatility parameter
- $W = (W_t)_{0\leqslant t\leqslant T}$ - Brownian motion

$$
S_t=S_0+\int_0^t\mu S_udu+\int_0^t\sigma S_udW_u
$$

This SDE is known to admit the following strong solution:

$$
\color{yellow}
S_t=S_0\exp\left(\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t\right)
$$
---
### 1.6 Simulate a geometric Brownian motion

The best way to simulate a GBM is to take advantage of the strong solution of the GBM SDE:

1. Simulate the arithmetic Brownian motion as explained before:

$$
\left(\mu-\frac{\sigma^2}{2}\right)t+\sigma W_t
$$

2. Take the exponential of each path and multiply the results by $S_0$
---
### 1.7 General Stochastic Differential Equation

Let the stochastic process $S=(S_t)_{0\leq t\leq T}$ be the solution of the following general stochastic differential equation:

$$
dS_t=\mu(t,S_t)dt+\sigma(t,S_t)dW_t
$$

- Starting from a fixed value $S_0 > 0$
- $\mu, \sigma$ - Real functions
- $W=(W_t)_{0\leq t\leq T}$ - Brownian motion

> For the solution $S$ to exist, the two functions $\mu$ and $\sigma$ must satisfy some regularity conditions (for example Lipschitz-continuity).

In the general case, a general SDE of this type does not admit an explicit strong solution. 

We thus need to use a simulation algorithm which does not require the knowledge of the solution of the SDE.
---
### 1.8 Simulate an SDE - Euler Scheme

The easiest and most popular way to simulate an SDE is the $\color{yellow}\text{Euler time discretization scheme}$. 

We use the same time grid as before $\mathcal{T}=\{t_0,t_1,\ldots,t_N\}$ such that $t_n=n\times\Delta_t$ where $\Delta_t:=\frac{T}{N}$.

The Euler scheme works as follows:

$$
S_{t_{n+1}}=S_{t_n}+\mu(t_n,S_{t_n})\Delta_t+\sigma(t_n,S_{t_n})(W_{t_{n+1}}-W_{t_n})
$$
- Starting from a fixed value $S_{t_0} = S_0$

To implement it in practice, we use the fact that $W_{t_{n+1}}-W_{t_n}\sim\mathcal{N}(0,\Delta_t)$:

$$
S_{t_{n+1}}=S_{t_n}+\mu(t_n,S_{t_n})\Delta_t+\sigma(t_n,S_{t_n})\sqrt{\Delta_t}G_n
$$
- $G_n, n = 1, ..., N$ - $N$ independent standard Gaussian variables.
---
#### 1.10 Arithmetic Brownian motion
def generate_arithmetic_brownian_motion(T = 1.0, N = 100, M = 10_000, init = 0.0, mu = 0.0, sigma = 1.0) -> tuple:
    """
    生成算术布朗运动路径
    
    参数:
    T: 总时间长度
    N: 时间步数
    M: 模拟的路径数量
    init: 初始值 (默认值: 0.0)
    mu: 漂移系数 (默认值: 0.0)
    sigma: 波动率系数 (默认值: 1.0)
    
    返回:
    t: 时间点数组，形状为 (N+1,)
    W: 路径矩阵，形状为 (N+1, M)。对于标准布朗运动，W 表示 W_t；对于算术布朗运动，W 表示 S_t。
    """
    dt = T / N
    sqrtdt = np.sqrt(dt)
    
    # 初始化存储矩阵：行代表时间，列代表不同路径
    X = np.zeros([N + 1, M])
    X[0, :] = init
    
    for n in tqdm.tqdm(range(N), desc = "Generating Arithmetic Brownian paths", unit = "step"):
        X[n + 1, :] = X[n, :] + mu * dt + sigma * sqrtdt * norm.rvs(size = M)
    
    t = np.linspace(0, T, N + 1)
    
    return (t, X)

def plot_comprehensive_arithmetic_brownian_analysis(t, X, T=1.0, num_paths=5, n_band=10, mu=0.1, sigma=0.2):
    """
    绘制算术布朗运动的综合分析图
    
    参数:
    t: 时间点数组
    X: 算术布朗运动路径矩阵
    T: 总时间长度
    num_paths: 显示的样本路径数量
    n_band: 扇形图的颜色带数量
    mu: 漂移系数 (用于标题显示)
    sigma: 波动率系数 (用于标题显示)
    """
    fig = plt.figure(figsize=(20, 6))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # 样本路径图
    for i in range(min(num_paths, X.shape[1])):
        ax1.plot(t, X[:, i], linewidth = 2)
    ax1.set_xlim([0, T])
    ax1.set_xlabel('Time ($t$)')
    ax1.set_ylabel('Arithmetic Brownian Motion $X(t)$')
    ax1.set_title(f'Sample Paths of Arithmetic Brownian Motion ($\\mu$={mu}, $\\sigma$={sigma}) {min(num_paths, X.shape[1])} paths', loc="left")
    ax1.grid(True, alpha=0.3)
    
    # 路径叠加扇形图
    col1 = sns.color_palette("mako_r", n_band // 2)
    M_display = min(num_paths, X.shape[1])
    # 分位数计算
    q_val = np.linspace(0.0, 1.0, n_band + 1)
    q_val[0] = 0.01
    q_val[-1] = 0.99
    Q = np.quantile(X, q_val, axis=1)
    # 路径绘制
    for i in range(M_display):
        ax2.plot(t, X[:, i], color="black", linewidth=2, alpha=1, zorder=1)
    for i in range(n_band):
        col_idx = np.abs(int(i - n_band // 2 + 0.5))
        ax2.fill_between(t, Q[i+1, :], Q[i, :], color=col1[col_idx], zorder=20, alpha=1)
    ax2.set_xlim([0, T])
    ax2.set_xlabel('Time ($t$)')
    ax2.set_title(f"Arithmetic Brownian Motion: {M_display} paths + fanchart ($\\mu$={mu}, $\\sigma$={sigma})", loc="left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig("comprehensive_arithmetic_brownian_analysis.svg", format="svg")
    plt.show()

T, N, M, init = 1.0, 100, 50_000, 100
mu, sigma = 5.0, 20.0
num_paths, num_band = 1000, 30

t, X = generate_arithmetic_brownian_motion(T = T, N = N, M = M, init = init, mu = mu, sigma = sigma)
plot_comprehensive_arithmetic_brownian_analysis(t, X, T = T, num_paths = num_paths, n_band = num_band, mu = mu, sigma = sigma)
---
### 2.1 Correlated Brownian motions

- $W=(W_t)_{0\leqslant t\leqslant T}$, $B=(B_t)_{0\leqslant t\leqslant T}$ - Two standard Brownian motions

If $\forall 0\leqslant s < t$,
$$
\mathbb{C}\mathrm{orr}\left(W_t-W_s,B_t-B_s\right)={\color{yellow}\rho}
$$
then, $W$, $B$ are $\color{yellow}\text{correlated Brownian motions}$ with constant correlation $\color{yellow}\rho$

In particular,
$$
\mathbb{C}\mathrm{ov}\left(W_t-W_s,B_t-B_s\right) = {\color{yellow}\rho}(t-s)
$$

$(W_t-W_s,B_t-B_s)$ - The random vector follows a bivariate Gaussian distribution:
$$
\begin{bmatrix}W_t-W_s\\B_t-B_s\end{bmatrix}\sim\mathcal{N}\left(\begin{bmatrix}0\\0\end{bmatrix},\begin{bmatrix}t-s&\rho(t-s)\\\rho(t-s)&t-s\end{bmatrix}\right)
$$
---
### 2.2 Simulate correlated Brownian motions

- Time grid $\mathcal{T}=\{t_0,t_1,\ldots,t_N\}$, s.t., $t_n=n\times\Delta_t$, where $\Delta_t:=\frac{T}{N}$

1. Simulate $N$ independent draws $(G_n, H_n)$, $n = 1, ..., N$ from a bivariate standard Gaussian vector

2. Set $W_{t_0}=B_{t_0}=0$, and $\forall \, n = 0, 1, ..., N - 1$:
$$
\begin{aligned}
&W_{t_{n+1}} = W_{t_n}+\sqrt{\Delta_t}G_n \\ 
&B_{t_{n+1}} = B_{t_n}+\sqrt{\Delta_t}\left({\color{yellow}\rho} G_n + \sqrt{1 - {\color{yellow}\rho}^2}H_n\right)
\end{aligned}
$$
---
def generate_correlated_brownian_motions(T = 1.0, N = 1000, M = 100, rho = 0.0):
    """
    生成两个相关的布朗运动路径
    
    参数:
    T: 总时间长度
    N: 时间步数
    M: 模拟的路径数量（蒙特卡洛模拟）
    rho: 相关系数
    
    返回:
    t: 时间点数组，形状为 (N+1,)
    W: 第一个布朗运动路径矩阵，形状为 (N+1, M)
    B: 第二个布朗运动路径矩阵，形状为 (N+1, M)
    """
    dt = T / N
    sqrtdt = np.sqrt(dt)
    
    # 初始化存储矩阵
    W = np.zeros([N + 1, M])
    B = np.zeros([N + 1, M])
    
    # 生成相关布朗运动路径
    for n in range(N):
        # 生成独立的标准正态随机变量
        G = norm.rvs(size = M)
        H = norm.rvs(size = M)
        
        # 更新路径
        W[n + 1, :] = W[n, :] + sqrtdt * G
        B[n + 1, :] = B[n, :] + sqrtdt * (rho * G + np.sqrt(1 - rho**2) * H)
    
    t = np.linspace(0, T, N + 1)
    
    return t, W, B

def plot_single_path_comparison_simple(t, W, B, rho, path_idx = None):
    """
    绘制相关布朗运动的单条路径对比分析图
    
    该函数生成两个子图：
    1. 时间序列图：显示随机选择的单条路径中两个相关布朗运动的时间演化
    2. 散点图：展示所有路径中两个布朗运动值的相关性，并突出显示选中的路径
    
    参数:
    t: 时间点数组
    W: 第一个布朗运动路径矩阵
    B: 第二个布朗运动路径矩阵
    rho: float: 相关系数，范围[-1, 1]
    path_idx: 指定的路径索引，如果为None则随机选择一条
    """
    # 随机选择一条路径（如果未指定）
    if path_idx is None:
        path_idx = np.random.randint(0, W.shape[1])
    
    # 创建图形和子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    # 1. 单条路径的时间序列对比
    axes[0].plot(t, W[:, path_idx], linewidth = 2, label=f'$W(t)$', color='blue')
    axes[0].plot(t, B[:, path_idx], linewidth = 2, label=f'$B(t)$', color='red')
    axes[0].set_xlabel('Time ($t$)', fontsize = 14)
    axes[0].set_ylabel('Value', fontsize = 14)
    axes[0].set_title(f'Time Evolution of Correlated Brownian Motions ($\\rho = {rho}$)', loc='left', fontsize = 16)
    axes[0].legend()
    axes[0].grid(True, alpha = 0.3)
    
    # 2. 两个布朗运动值的对比图（散点图）- 所有时间点
    W_flat = W.flatten()
    B_flat = B.flatten()
    actual_rho = np.corrcoef(W_flat, B_flat)[0, 1]  # 计算实际相关系数
    # 绘制散点图（抽样以减少重叠）
    sample_size = min(2_000, len(W_flat))
    indices = np.random.choice(len(W_flat), sample_size, replace = False)
    axes[1].scatter(
        W_flat[indices], B_flat[indices], 
        alpha = 0.5, s = 100, linewidth = 0.1,
        color = sns.color_palette("mako")[2], edgecolor = "white"
    )
    axes[1].set_aspect('equal', 'box')
    axes[1].set_xlabel('$W(t)$', fontsize = 14)
    axes[1].set_ylabel('$B(t)$', fontsize = 14)
    axes[1].set_title(f'Correlation Analysis of Brownian Motions \n Target $\\rho = {rho}$, Actual $\\rho = {actual_rho:.4f}$', loc='left', fontsize = 16)
    axes[1].grid(True, alpha = 0.3)
    # 标记选中的路径的所有点
    axes[1].scatter(
        W[:, path_idx], B[:, path_idx], 
        label = 'Selected path',
        s = 100, linewidth = 0.5,
        color = 'red', edgecolor = 'black'
    )
    axes[1].legend()
    
    plt.tight_layout(pad = 3, w_pad = 1.0)
    # plt.savefig("correlated_brownian_motions_comparison.svg", format = "svg")
    plt.show()

def plot_correlated_brownian_motions(t, W, B, rho, M):
    """
    绘制相关布朗运动的散点图
    
    参数:
    t: 时间点数组
    W: 第一个布朗运动路径矩阵
    B: 第二个布朗运动路径矩阵
    rho: 相关系数
    M: 路径数量
    """
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    
    # 两个布朗运动的散点图
    num_paths_to_show = min(30, M)
    for i in range(num_paths_to_show):
        ax.plot(B[:, i], W[:, i], linewidth = 2, alpha = 0.9)
    # ax.set_xlim([-3.5, 3.5])
    # ax.set_ylim([-3.5, 3.5])
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$B(t)$', fontsize = 14)
    ax.set_ylabel('$W(t)$', fontsize = 14)
    ax.set_title(f'Correlated Brownian Motions ($\\rho = {rho}, M = {num_paths_to_show}$)', loc='left', fontsize = 16)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # plt.savefig("correlated_brownian_motions.svg", format = "svg")
    plt.show()

T, N, M, rho = 1.0, 1000, 1000, -0.75
t, W, B = generate_correlated_brownian_motions(T = T, N = N, M = M, rho = rho)
plot_single_path_comparison_simple(t, W, B, rho)
plot_correlated_brownian_motions(t, W, B, rho, M)
---
### 2.3 Stochastic volatility models

A one-factor $\color{yellow}\text{stochastic volatility model}$ is the solution $({\color{yellow}S}, {\color{red}V})$ of a bivariate SDE of the type:

$$
\begin{aligned}
&d{\color{yellow}S}_t = \mu(t,{\color{yellow}S}_t)dt + \sigma(t,{\color{yellow}S}_t,V_t)dW_t \\
&d{\color{red}V}_t = \alpha(t,{\color{red}V}_t)dt + \lambda(t,{\color{red}V}_t)dB_t \\
&d \langle W,B \rangle_t = \rho dt
\end{aligned}
$$
- Starting from a fixed value $({\color{yellow}S}_0, {\color{red}V}_0)$
- $\mu, \sigma, \alpha, \lambda$ - Real functions
- $(W,B)=(W_t,B_t)_{0\leqslant t\leqslant T}$ - Correlated bivariate Brownian motion

1. ${\color{yellow}S}$ - Asset price process
2. ${\color{red}V}$ - Variance/volatility process (depending on the function $\sigma$)
---
### 2.4 Simulate a stochastic volatility model

The solution of a stochastic volatility model can be easily simulated using the $\color{yellow}\text{Euler scheme}$.

- $\mathcal{T}=\{t_0,t_1,\ldots,t_N\}$ - Uniform time grid with $t_n=n\Delta_t$ and $\Delta_t=\frac{T}{N}$

$$
\begin{aligned}
&{\color{yellow}S}_{t_{n+1}} = {\color{yellow}S}_{t_n} + \mu(t_n,{\color{yellow}S}_{t_n})\Delta_t + \sigma(t_n,{\color{yellow}S}_{t_n},{\color{red}V}_{t_n})\left(\sqrt{\Delta_t}G_n\right) \\
&{\color{red}V}_{t_{n+1}} = {\color{red}V}_{t_n} + \alpha(t_n,{\color{red}V}_{t_n})\Delta_t + \lambda(t_n,{\color{red}V}_{t_n})\sqrt{\Delta_t}\left(\rho G_n+\sqrt{1-\rho^2}H_n\right)
\end{aligned}
$$
- Starting from a fixed value $({\color{yellow}S}_0, {\color{red}V}_0)$

where we used the fact that:

$$
\begin{aligned}
&W_{t_{n+1}} - W_{t_n} = \sqrt{\Delta_t}G_n \\
&B_{t_{n+1}} - B_{t_n} = \sqrt{\Delta_t}\left(\rho G_n+\sqrt{1-\rho^2}H_n\right)
\end{aligned}
$$
---
请完全阅读、理解以上所有内容，一步一步来，确保完全理解。
---
请按照要求以及根据以上内容、代码示例。
随后，参考第一部分的“上涨敲出看涨期权（Up-and-Out Call）蒙特卡洛定价”的完整代码，请生成完整，详细、思路通顺、逻辑清晰、结构完整的“#### 2. Monte Carlo pricing under stochastic volatility model”的“上涨敲出看涨期权（Up-and-Out Call）蒙特卡洛定价”的内容以及Python代码，要函数化，向量化，最简洁。
---
现在，做个深呼吸，一步一步地思考。