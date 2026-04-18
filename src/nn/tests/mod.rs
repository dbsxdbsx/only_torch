// ===== Phase 1: 已适配/几乎适配的测试 =====
mod debug; // Debug 模块测试（节点类型枚举和调试工具）
mod graph_handle; // Graph 高层 API 测试（PyTorch 风格）
mod graph_parameters; // GraphInner 参数注册表测试
mod module_trait; // Module trait 测试
mod node_inner; // NodeInner 核心数据结构测试
mod optimizer; // 优化器测试（PyTorch 风格 API）
mod var_ops; // Var 算子重载和链式调用测试

// ===== Phase 2a: 已完全适配的 node 文件（old=0）=====
mod node_amax; // Amax 节点（沿轴取最大值，DQN 选最优动作 Q 值）
mod node_amin; // Amin 节点（沿轴取最小值，Double DQN 选保守 Q 值）
mod node_gather; // Gather 节点（张量按索引收集，强化学习用，动态索引）
mod node_maximum; // Maximum 节点（逐元素取最大值，PPO/TD3 等需要可微分 max）
mod node_minimum; // Minimum 节点（逐元素取最小值，PPO clipping、TD3 双 Q）
mod node_zeros_like; // ZerosLike 节点测试（动态零张量）

// ===== 节点测试 =====
mod node_abs;
mod node_add;
mod node_avg_pool2d;
mod node_batch_norm;
mod node_layer_norm;
mod node_rms_norm;
mod node_bce;
mod node_clip;
mod node_concat;
mod node_conv2d;
mod node_detach;
mod node_divide;
mod node_dropout;
mod node_elu;
mod node_exp;
mod node_flatten;
mod node_gelu;
mod node_hard_sigmoid;
mod node_hard_swish;
mod node_hard_tanh;
mod node_huber;
mod node_identity;
mod node_input;
mod node_leaky_relu;
mod node_ln;
mod node_log10;
mod node_log2;
mod node_log_softmax;
mod node_mae;
mod node_mat_mul;
mod node_max_pool2d;
mod node_mean;
mod node_mish;
mod node_mse;
mod node_multiply;
mod node_narrow;
mod node_negate;
mod node_pad;
mod node_parameter;
mod node_permute;
mod node_pow;
mod node_reciprocal;
mod node_relu;
mod node_relu6;
mod node_repeat;
mod node_reshape;
mod node_select;
mod node_selu;
mod node_sigmoid;
mod node_sign;
mod node_softmax;
mod node_softmax_cross_entropy;
mod node_softplus;
mod node_sort;
mod node_split;
mod node_sqrt;
mod node_square;
mod node_squeeze;
mod node_stack;
mod node_state;
mod node_step;
mod node_subtract;
mod node_sum;
mod node_swish;
mod node_tanh;
mod node_topk;
mod node_unsqueeze;
mod node_where_cond;

// ===== 图基础设施测试 =====
mod cse_dedup; // CSE（公共子表达式消除）节点去重机制
mod graph_node_group_context; // NodeGroupContext 节点分组上下文机制
mod graph_visualization; // 计算图可视化（DOT cluster 生成）

// ===== 概率分布测试 =====
mod distribution_categorical; // Categorical 分布（离散分类，SAC-Discrete / Hybrid SAC）
mod distribution_normal; // Normal 分布（log_prob / entropy / rsample / 梯度）
mod distribution_tanh_normal; // TanhNormal 分布（Squashed Gaussian，SAC-Continuous 标准策略）

// ===== 其他单元测试 =====
mod shape; // DynamicShape 动态维度形状系统测试
mod var_init; // Init 参数初始化策略测试
mod scheduler; // LR 调度器测试（CosineAnnealingLR, StepLR, LambdaLR）

// ===== 复杂测试 =====
mod batch_mechanism;
mod gradient_flow_control;
mod graph_backward;
mod graph_basic;
mod graph_dynamic;
mod graph_forward;
mod layer_avg_pool2d;
mod layer_attention;
mod layer_batch_norm;
mod layer_group_norm;
mod layer_instance_norm;
mod layer_embedding;
mod layer_layer_norm;
mod layer_rms_norm;
mod layer_conv2d;
mod layer_gru;
mod layer_linear;
mod layer_lstm;
mod layer_max_pool2d;
mod layer_rnn;
mod save_load;
mod model_save;

// ===== ONNX 双向桥接测试 =====
mod onnx;

// ===== 已移除/归档 =====
// mod var_transition; // 已删除（过渡期测试，迁移完成后不再需要）
// mod criterion; // Criterion 已移除，见文档 4.3 节
// mod model_state; // ModelState 已移除，见文档 4.1 节
// mod recurrent_basic; // 旧式显式时间步测试，已被展开式 RNN 设计取代
// mod recurrent_bptt; // 旧式 BPTT 测试，标准 backward() 已覆盖
// mod bptt_pytorch_comparison; // 旧式 BPTT PyTorch 对照，已被 layer_rnn 等测试覆盖
