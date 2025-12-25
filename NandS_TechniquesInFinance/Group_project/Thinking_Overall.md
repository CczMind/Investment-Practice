# 一、问题背景与数学模型概述

## 1.1 障碍期权（Barrier Options）

障碍期权是一类**路径依赖（path-dependent）**的奇异期权，其到期支付不仅依赖于终端价格 $S_T$，还依赖于整个区间 $[0,T]$ 上价格过程 $S_t$ 是否触及某一障碍水平 $B$。

设：
- $S_t$：标的资产价格过程；
- $K>0$：行权价；
- $B>0$：障碍价；
- $T>0$：到期时间；
- $\max_{0\le t\le T} S_t$、$\min_{0\le t\le T} S_t$：路径极大值/极小值。

作业中要求的八种障碍期权为：

- **Knock-out（敲出）期权**
  - Up-and-Out Call  
    $$
    P_T=(S_T-K)^+\mathbf{1}\left\{\max_{0\le t\le T}S_t<B\right\},\quad 0<S_0<B
    $$
  - Down-and-Out Call  
    $$
    P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\le t\le T}S_t>B\right\},\quad 0<B<S_0
    $$
  - Up-and-Out Put  
    $$
    P_T=(K-S_T)^+\mathbf{1}\left\{\max_{0\le t\le T}S_t<B\right\},\quad 0<S_0<B
    $$
  - Down-and-Out Put  
    $$
    P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\le t\le T}S_t>B\right\},\quad 0<B<S_0
    $$

- **Knock-in（敲入）期权**
  - Up-and-In Call  
    $$
    P_T=(S_T-K)^+\mathbf{1}\left\{\max_{0\le t\le T}S_t\ge B\right\},\quad 0<S_0<B
    $$
  - Down-and-In Call  
    $$
    P_T=(S_T-K)^+\mathbf{1}\left\{\min_{0\le t\le T}S_t\le B\right\},\quad 0<B<S_0
    $$
  - Up-and-In Put  
    $$
    P_T=(K-S_T)^+\mathbf{1}\left\{\max_{0\le t\le T}S_t\ge B\right\},\quad 0<S_0<B
    $$
  - Down-and-In Put  
    $$
    P_T=(K-S_T)^+\mathbf{1}\left\{\min_{0\le t\le T}S_t\le B\right\},\quad 0<B<S_0
    $$

在无套利、完备市场假设下，期初价格为贴现后的条件期望：
$$
\Pi_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[P_T]
$$
其中 $r$ 为无风险利率，$\mathbb{Q}$ 为风险中性测度。

## 1.2 Q1：Black–Scholes / GBM 模型

在 **Q1** 中，标的资产价格在风险中性测度下满足几何布朗运动（GBM）：
$$
dS_t = r S_t dt + \sigma S_t dW_t,
$$
其中：
- $r>0$ 为无风险利率；
- $\sigma>0$ 为常数波动率；
- $W_t$ 为标准布朗运动。

其显式解为：
$$
S_t = S_0\exp\left(\left(r-\tfrac{1}{2}\sigma^2\right)t + \sigma W_t\right).
$$

在数值实现中，我们使用**时间离散化**和蒙特卡洛模拟近似上述过程

## 1.3 Q2：3/2 随机波动率模型

在 **Q2** 中，常数波动率 $\sigma$ 被更现实的 **3/2 随机波动率模型**所替代。设 $V_t$ 表示瞬时方差过程，则在风险中性测度下：
$$
\begin{aligned}
dS_t &= r S_t dt + \sqrt{V_t} S_t\, dW_t, \\
dV_t &= \kappa(\theta V_t - V_t^2) dt + \lambda V_t^{3/2} dB_t,\\
d\langle W, B\rangle_t &= \rho\, dt,
\end{aligned}
$$
其中：
- $V_0 = \theta = 0.20$：初始方差与长期均值；
- $\kappa=0.20$：均值回复速度；
- $\lambda=0.67$：波动率的波动率（vol-vol）；
- $\rho=-0.5$：价格与方差的相关系数；
- $W_t$、$B_t$ 为相关布朗运动。

3/2 模型通过非线性项 $V_t^{3/2}$ 允许更厚尾的波动率分布，刻画波动率聚集、波动率爆发等现象，对障碍期权这类强路径依赖产品尤为重要。

## 1.4 蒙特卡洛估计与置信区间

给定任意一类障碍期权，其理论价格为：
$$
\Pi_0 = e^{-rT}\mathbb{E}[P_T].
$$

采用蒙特卡洛方法，生成 $M$ 条独立路径，得到样本贴现支付 $\{X_i\}_{i=1}^M$，其中：
$$
X_i = e^{-rT} P_T^{(i)}.
$$
则估计量与样本标准差为：
$$
\hat{\Pi}_M = \frac{1}{M}\sum_{i=1}^M X_i,\quad
\hat{\sigma}_M = \sqrt{\frac{1}{M-1}\sum_{i=1}^M (X_i - \hat{\Pi}_M)^2 }.
$$

利用中心极限定理，在 $M$ 足够大时：
$$
\hat{\Pi}_M \approx \mathcal{N}\left(\Pi_0, \frac{\sigma^2}{M} \right),
$$
因此给出 **99% 置信区间**：
$$
\left[\hat{\Pi}_M - z_{0.995}\frac{\hat{\sigma}_M}{\sqrt{M}},\,
      \hat{\Pi}_M + z_{0.995}\frac{\hat{\sigma}_M}{\sqrt{M}}\right],
\quad z_{0.995}\approx 2.576.
$$