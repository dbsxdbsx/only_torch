//! MyZero 模块测试索引
//!
//! 单元测试（自源码文件内嵌迁入，一文件对应一个源模块）+ 手动档 bench（`--ignored`）。

// ---- 单元测试（按源模块分文件）----
mod action;
mod builder;
mod checkpoint;
mod component;
mod config;
mod consistency;
mod model_io;
mod n_step;
mod reanalyze;
mod recipe;
mod sampled_params;
mod sve;
mod target;
mod target_net;
mod value_prefix;
mod value_transform;

// ---- 集成 / 等价性 / 复现性 ----
mod batch_train_equivalence;
mod reanalyze_writeback;
mod seed_reproducibility;
mod value_head_capacity;

// ---- 手动档 bench（--ignored）----
mod baseline_matrix_bench;
mod completed_q_cartpole_bench;
mod loss_coef_ablation_bench;
