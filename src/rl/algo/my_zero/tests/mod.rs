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
mod conv_repr;
mod model_io;
mod n_step;
mod obs_pipeline;
mod obs_transform;
mod reanalyze;
mod recipe;
mod sampled_params;
mod sve;
mod target;
mod target_net;
mod temperature;
mod value_encoding;
mod value_prefix;
mod value_transform;

// ---- 集成 / 等价性 / 复现性 ----
mod batch_train_equivalence;
mod network_decode;
mod reanalyze_writeback;
mod seed_reproducibility;
mod value_head_capacity;

// ---- 手动档 bench（--ignored）----
mod baseline_matrix_bench;
mod completed_q_cartpole_bench;
mod hl_gauss_ablation_bench;
mod loss_coef_ablation_bench;
mod obs_symlog_ablation_bench;
mod pong_image_ablation_bench;
mod spike_cnn_mcts_bench;
