// ===== Phase 1: 已适配/几乎适配的测试 =====
mod debug; // Debug 模块测试（节点类型枚举和调试工具）
mod graph_handle; // Graph 高层 API 测试（PyTorch 风格）
mod graph_parameters; // GraphInner 参数注册表测试（方案 C Step 2.3）
mod module_trait; // Module trait 测试
mod node_inner; // NodeInner 核心数据结构测试（方案 C）
mod optimizer; // 优化器测试（PyTorch 风格 API）
mod var_ops; // Var 算子重载和链式调用测试

// ===== Phase 2a: 已完全适配的 node 文件（old=0）=====
mod node_amax; // Amax 节点（沿轴取最大值，DQN 选最优动作 Q 值）
mod node_amin; // Amin 节点（沿轴取最小值，Double DQN 选保守 Q 值）
mod node_gather; // Gather 节点（张量按索引收集，强化学习用，动态索引）
mod node_maximum; // Maximum 节点（逐元素取最大值，PPO/TD3 等需要可微分 max）
mod node_minimum; // Minimum 节点（逐元素取最小值，PPO clipping、TD3 双 Q）
mod node_zeros_like; // ZerosLike 节点测试（动态零张量）

// ===== Phase 2b: node_*.rs（旧 API 函数已用 #[cfg(any())] 禁用）=====
mod node_abs;
mod node_add;
mod node_avg_pool2d;
mod node_bce;
mod node_conv2d;
mod node_divide;
mod node_dropout;
mod node_flatten;
mod node_huber;
mod node_identity;
mod node_input;
mod node_leaky_relu;
mod node_ln;
mod node_log_softmax;
mod node_mae;
mod node_mat_mul;
mod node_max_pool2d;
mod node_mean;
mod node_mse;
mod node_multiply;
mod node_parameter;
mod node_reshape;
mod node_select;
mod node_sigmoid;
mod node_sign;
mod node_softmax;
mod node_softmax_cross_entropy;
mod node_softplus;
mod node_stack;
mod node_state;
mod node_step;
mod node_subtract;
mod node_sum;
mod node_tanh;

// ===== Phase 3: 复杂测试（旧 API 函数已用 #[cfg(any())] 禁用，layer 已完全适配） =====
mod graph_basic;
mod graph_forward;
mod graph_backward;
mod graph_dynamic;
mod layer_avg_pool2d;
mod layer_conv2d;
mod layer_gru;
mod layer_linear;
mod layer_lstm;
mod layer_max_pool2d;
mod layer_rnn;
mod recurrent_basic;
mod recurrent_bptt;
mod gradient_flow_control;
mod batch_mechanism;
mod bptt_pytorch_comparison;
mod save_load;

// ===== 已移除/归档 =====
// mod var_transition; // Var 过渡期测试（方案 C Step 2.2）- 迁移完成后不再需要
// mod criterion; // Criterion 已移除，见文档 4.3 节
// mod model_state; // ModelState 已移除，见文档 4.1 节
