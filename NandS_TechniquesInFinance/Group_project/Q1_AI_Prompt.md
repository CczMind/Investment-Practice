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

- Up-and-out call option - `$\Large P_{T}=(S_{T}-K)^{+}\mathbf{1}\left\{\max_{0\leq t\leq T}S_{t}<B\right\}$`, `$\Large 0<S_0<B$`

- Down-and-out call option - `$\Large P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$`, `$\Large 0<B<S_0$`

- Up-and-out put option - `$\Large P_T=(K-S_T)^+\mathbf{1}\left\{\max_{0\leq t\leq T}S_t<B\right\}$`, `$\Large 0<S_0<B$`

- Down-and-out put option - `$\Large P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t>B\right\}$`, `$\Large 0<B<S_0$`

### Knock-in barrier options

- Up-and-in call option - `$\Large P_T=(S_T-K)^+\mathbf{1}\left\{\max_{0\leq t\leq T}S_t\geq B\right\}$`, `$\Large 0<S_0<B$`

- Down-and-in call option - `$\Large P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t\leq B\right\}$`, `$\Large 0<B<S_0$`

- Up-and-in put option - `$\Large P_{T}=(K-S_{T})^{+}\mathbf{1}\left\{\max_{0\leq t\leq T}S_{t}\geq B\right\}$`, `$\Large 0<S_0<B$`

- Down-and-in put option - `$\Large P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\leq t\leq T}S_t\leq B\right\}$`, `$\Large 0<B<S_0$`

Here,
- `$\Large P_{T}\geq0$` is the final cash-flow of the option, received at time `$\Large T$` (the maturity of
the option), 
- `$\Large S_{T}\geq0$` is the price of the underlying asset at time `$\Large T$`
- `$\Large K > 0$` is the
strike price of the option,
- `$\Large B > 0$` is the barrier price

#### 1. Monte Carlo pricing under Black-Scholes model

Suppose that the underlying asset price process follows a **geometric Brownian motion** with **constant drift** `$\Large r > 0$` and **constant volatility** `$\Large \sigma > 0$` under the risk-neutral pricing measure.

1. Describe how to estimate the initial premiums of these eight barrier options by Monte Carlo simulations using the Euler scheme on a uniform time grid with `$\Large N$` equal time intervals between `$\Large t = 0$` and the option maturity `$\Large t = T$`

2. Implement these eight Monte Carlo estimators in Python. For your numerical tests, set `$\Large r = 0.05$`, `$\Large \sigma = 0.20$`, `$\Large T = 1$`, `$\Large S_0 = K = 100$`, and try a few different values of `$\Large B$`.

3. Sensitivity analysis with respect to the number `$\Large M$` of Monte Carlo simulations: plot the estimated option premiums and `$\Large 99\%$` (asymptotic) confidence intervals with respect to `$\Large M$` for each of these eight options (set `$\Large B$` to an "interesting" value based
on your previous tests).

4. Sensitivity analysis with respect to the number `$\Large N$` of time steps: fix `$\Large M$` to a large value ensuring numerical convergence (based on your results from the above sensitivity analysis), then plot the estimated option premiums with respect to `$\Large N$` for each of these eight options (set `$\Large B$` to the same values as before).
---
请仔细阅读、理解以上小组项目要求。
---
以下是一些内容、代码示例：
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
请完全阅读、理解以上内容，一步一步来，确保完全理解。
---
请按照以上的要求以及内容、示例。
随后，请只生成完整，详细、思路通顺、逻辑清晰、结构完整的“#### 1. Monte Carlo pricing under Black-Scholes model”的“上涨敲出看涨期权（Up-and-Out Call）”的解答和Python代码，要函数化，最简洁。
<!-- 随后，请生成完整，详细、思路通顺、逻辑清晰、结构完整的“#### 1. Monte Carlo pricing under Black-Scholes model”的Python代码，要函数化，最简洁。 -->
---
现在，做个深呼吸，一步一步地思考。

---

你做得很好，表扬你！
---
现在，请按照以上的内容、代码示例格式，重写整个“上涨敲出看涨期权（Up-and-Out Call）”代码。
用函数。
---
做个深呼吸，一步一步地思考。