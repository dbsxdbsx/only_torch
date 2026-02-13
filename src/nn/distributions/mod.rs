/*
 * @Author       : 老董
 * @Date         : 2026-02-12
 * @LastEditTime : 2026-02-13
 * @Description  : 概率分布模块
 *
 * 提供常用概率分布的计算图内实现，支持重参数化采样和梯度传播。
 * 主要用于 SAC-Discrete / SAC-Continuous / Hybrid SAC 等强化学习算法。
 *
 * # 包含分布
 * - `Categorical` — 离散分类分布（SAC-Discrete）
 * - `Normal` — 正态分布（重参数化采样、log_prob、entropy）
 * - `TanhNormal` — Squashed Gaussian（SAC-Continuous 标准策略）
 *
 * # 设计原则
 * - **需要梯度 → Var，不需要梯度 → Tensor**（唯一例外：`sample()`）
 * - 构造时预计算并缓存共享的 Var 中间值，避免冗余图节点
 * - 不引入新的 Node 类型，组合现有计算图节点
 * - API 参考 PyTorch `torch.distributions`
 */

pub mod categorical;
pub mod normal;
pub mod tanh_normal;

pub use categorical::Categorical;
pub use normal::Normal;
pub use tanh_normal::TanhNormal;
