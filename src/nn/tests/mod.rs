mod batch_mechanism; // Batch 机制测试
mod bptt_pytorch_comparison; // BPTT PyTorch 对照测试
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
mod layer_rnn; // RNN 层便捷函数
mod module_trait; // Module trait 测试（V2 API）
mod node_add;
mod node_avg_pool2d; // AvgPool2d 节点（2D 平均池化）
mod node_divide; // Divide 节点（逐元素除法）
mod node_conv2d; // Conv2d 节点（2D 卷积）
mod node_flatten; // Flatten 节点（展平）
mod node_input;
mod node_leaky_relu; // LeakyReLU/ReLU 激活函数
mod node_mat_mul;
mod node_max_pool2d; // MaxPool2d 节点（2D 最大池化）
mod node_mse_loss; // MSELoss 节点（均方误差损失）
mod node_multiply;
mod node_parameter;
mod node_perception_loss;
mod node_reshape; // Reshape 节点（形状变换）
mod node_scalar_multiply;
mod node_sigmoid;
mod node_sign; // Sign 符号函数（正→1, 负→-1, 零→0）
mod node_softmax_cross_entropy;
mod node_softplus; // SoftPlus 激活函数（ReLU 的平滑近似）
mod node_state; // State 节点测试（RNN 时间状态）
mod node_step;
mod node_subtract; // Subtract 节点（逐元素减法）
mod node_tanh;
mod optimizer; // 优化器测试（PyTorch 风格 API）
mod recurrent_basic; // Phase 1: 循环/记忆机制基础测试
mod recurrent_bptt; // Phase 2: BPTT 通过时间反向传播测试
mod save_load; // 参数保存/加载测试
mod var_ops; // V2 API: Var 算子重载和链式调用测试
