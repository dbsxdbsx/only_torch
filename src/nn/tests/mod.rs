mod batch_mechanism; // Batch 机制测试
mod debug; // Debug 模块测试（节点类型枚举和调试工具）
mod graph_parameters; // GraphInner 参数注册表测试（方案 C Step 2.3）
mod node_inner; // NodeInner 核心数据结构测试（方案 C）
mod var_transition; // Var 过渡期测试（方案 C Step 2.2）
mod bptt_pytorch_comparison; // BPTT PyTorch 对照测试
mod criterion; // Criterion（损失函数封装）智能缓存测试
mod gradient_flow_control; // 梯度流控制机制测试（no_grad、detach、retain_graph）
mod graph_backward;
mod graph_basic;
mod graph_dynamic; // M4: 动态图扩展能力测试（NEAT 友好性）
mod graph_forward;
mod graph_handle; // Graph 高层 API 测试（PyTorch 风格）
mod layer_avg_pool2d; // AvgPool2d 层便捷函数
mod layer_conv2d; // Conv2d 层便捷函数
mod layer_gru; // GRU 层便捷函数
mod layer_linear; // Linear 层测试（包含 linear() 遗留 API 和 Linear 结构体推荐 API）
mod layer_lstm; // LSTM 层便捷函数
mod layer_max_pool2d; // MaxPool2d 层便捷函数
mod layer_rnn; // RNN 层测试（展开式设计）
mod model_state; // ModelState（模型状态管理器）智能缓存测试
mod module_trait; // Module trait 测试
mod node_abs; // Abs 绝对值函数（|x|，梯度为 sign(x)）
mod node_add;
mod node_avg_pool2d; // AvgPool2d 节点（2D 平均池化）
mod node_bce; // BCE 节点（Binary Cross Entropy，二元交叉熵损失）
mod node_conv2d; // Conv2d 节点（2D 卷积）
mod node_divide; // Divide 节点（逐元素除法）
mod node_dropout; // Dropout 节点（正则化，训练时随机丢弃）
mod node_flatten; // Flatten 节点（展平）
mod node_huber; // Huber Loss 节点（Smooth L1 Loss，强化学习标准损失函数）
mod node_identity; // Identity 节点（恒等映射，用于 detach_node）
mod node_input; // InputVariant（BasicInput, TargetInput, SmartInput）
mod node_leaky_relu; // LeakyReLU/ReLU 激活函数
mod node_ln; // Ln 节点（自然对数，用于计算 log 概率）
mod node_log_softmax; // LogSoftmax 节点（数值稳定的 log(softmax)）
mod node_mae; // MAE 节点（Mean Absolute Error，平均绝对误差损失）
mod node_mat_mul;
mod node_max_pool2d; // MaxPool2d 节点（2D 最大池化）
mod node_mse; // MSE 节点（Mean Squared Error，均方误差损失）
mod node_multiply;
mod node_parameter;
mod node_reshape; // Reshape 节点（形状变换）
mod node_select; // Select 节点（张量索引选择，RNN 展开式设计用，固定索引）
mod node_gather; // Gather 节点（张量按索引收集，强化学习用，动态索引）
mod node_maximum; // Maximum 节点（逐元素取最大值，PPO/TD3 等需要可微分 max）
mod node_minimum; // Minimum 节点（逐元素取最小值，PPO clipping、TD3 双 Q）
mod node_amax; // Amax 节点（沿轴取最大值，DQN 选最优动作 Q 值）
mod node_amin; // Amin 节点（沿轴取最小值，Double DQN 选保守 Q 值）
mod node_sigmoid;
mod node_sign; // Sign 符号函数（正→1, 负→-1, 零→0）
mod node_softmax; // Softmax 激活函数（沿最后一维归一化为概率分布）
mod node_softmax_cross_entropy;
mod node_softplus; // SoftPlus 激活函数（ReLU 的平滑近似）
mod node_stack; // Stack 节点（多张量堆叠/拼接）
mod node_sum; // Sum 节点（归约求和，支持全局和按轴模式）
mod node_mean; // Mean 节点（归约求均值，支持全局和按轴模式）
mod node_state; // State 节点测试（RNN 时间状态）
mod node_step;
mod node_subtract; // Subtract 节点（逐元素减法）
mod node_tanh;
mod node_zeros_like; // ZerosLike 节点测试（动态零张量）
mod optimizer; // 优化器测试（PyTorch 风格 API）
mod recurrent_basic; // 循环/记忆机制基础测试
mod recurrent_bptt; // BPTT 通过时间反向传播测试
mod save_load; // 参数保存/加载测试
mod var_ops; // Var 算子重载和链式调用测试
