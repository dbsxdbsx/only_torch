# Beta 分布作为连续 SAC 策略的备选方案

> 参考论文：Chou et al., *Improving Stochastic Policy Gradients in Continuous Control with Deep RL using the Beta Distribution*, ICML 2017
>
> 论文副本：[`SAC+Beta.pdf`](../../.doc/paper/RL/SAC+Beta.pdf)

## 问题：Gaussian 策略的边界偏差

标准 SAC 用 Gaussian + tanh 压缩来处理有界动作空间（如 [-1, 1]）。但 Gaussian 的无限支撑会导致：

1. **概率溢出**：部分概率密度落在有效范围外
2. **边界偏差**：截断或压缩都会在边界附近引入梯度估计偏差
3. **额外复杂性**：tanh 压缩需要 Jacobian 修正项 `log|det(∂tanh/∂x)|`

维度越高，溢出越严重——论文中 Humanoid（17 维）的改善最为显著。

## Beta 分布的核心优势

Beta 分布的支撑天然有界（[0, 1]，可缩放到 [a, b]）：

| 特性 | Gaussian + tanh | Beta |
|------|----------------|------|
| 边界偏差 | tanh 大致消除，但非完美 | **完全无偏** |
| 需要 Jacobian 修正 | 需要 | **不需要** |
| 能表示均匀分布 | 不能 | 能（α=β=1） |
| entropy 计算 | 需要修正项 | 直接计算 |

## 对 Target Entropy 的影响

Beta 在 [-1, 1] 上的熵有**有限上界**，这改变了 target entropy 的设定逻辑：

```
Gaussian [-1,1]：H ∈ (-∞, +∞)    → target = -d（经验值，无天然锚点）
Beta [-1,1]：   H ∈ (-∞, ln(2)]  → target 可用比例系数 c × ln(2) × d（有天然锚点）
```

Beta 在这方面介于离散和 Gaussian 之间——上界有限（像离散），下界无限（像连续）。

## 注意事项

1. **论文用的是 TRPO/ACER，非 SAC**：SAC 的 tanh 已部分缓解边界问题，Beta 的相对优势可能小于论文中展示的幅度
2. **限制 α,β > 1（单峰）**：无法表示多峰分布（Normalizing Flows 可以扩展 Gaussian 为多峰）
3. **Fisher 信息矩阵趋零**：策略变确定时普通梯度下降失效，可能需要自然梯度
4. **重参数化不如 Gaussian 直接**：通常需用 Kumaraswamy 近似或 Implicit Reparameterization Gradients

## 对本项目的启示

- 未来实现连续 SAC 时，可将 Beta 作为**可选的分布后端**，与 Gaussian+tanh 并列
- 在高维连续动作 + 严格边界控制的场景中优先考虑 Beta
- Target Entropy 的设定方式需要根据分布类型切换（比例系数 vs 经验值 -d）
