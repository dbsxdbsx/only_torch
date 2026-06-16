//! MyZero 统一算法（v0.25）
//!
//! only_torch 的终极 model-based RL 算法——从 canonical MuZero 出发，
//! 以消融实验方式逐增量叠加组件（consistency / value prefix / target /
//! SVE / Gumbel），最终覆盖全动作空间与全状态类型。
//!
//! 核心哲学：**奥卡姆剃刀**——每叠一个组件必须用消融证明其价值，保证不回归。
//!
//! # 消融序列（CartPole-v1 为 sanity 主线）
//!
//! | 步骤 | 开关 | 说明 |
//! |------|------|------|
//! | S0 | 全关 | canonical MuZero（base） |
//! | S1 | +consistency | SimSiam 自监督 |
//! | S2 | +value_prefix | LSTM 累计 reward 前缀 |
//! | S3 | +target_net | EMA/hard 同步 |
//! | S4 | +SVE | search value blend |
//! | S5 | +Gumbel | 连续/混合动作搜索 |
//!
//! # 代码组织
//!
//! - **入库**：`MyZeroConfig`（统一配置）+ `FeatureSet`（消融开关）
//! - **留示例**：三网络结构、训练循环、self-play 主流程
//! - **按需复用**：需要 `muzero/` 或 `efficientzero/` 的组件时复制到此或公共位置，
//!   旧件打 `TODO(my_zero)` 标记

pub mod config;
pub mod feature;

pub use config::MyZeroConfig;
pub use feature::FeatureSet;
