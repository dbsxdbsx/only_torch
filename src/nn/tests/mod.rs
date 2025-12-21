mod batch_mechanism; // Batch 机制测试
mod graph_backward;
mod graph_basic;
mod graph_dynamic; // M4: 动态图扩展能力测试（NEAT 友好性）
mod graph_forward;
mod node_add;
mod node_flatten; // Flatten 节点（展平）
mod node_input;
mod node_leaky_relu; // LeakyReLU/ReLU 激活函数
mod node_mat_mul;
mod node_multiply;
mod node_parameter;
mod node_perception_loss;
mod node_reshape; // Reshape 节点（形状变换）
mod node_scalar_multiply;
mod node_sigmoid;
mod node_softmax_cross_entropy;
mod node_step;
mod node_tanh;
mod optimizer; // 优化器测试模块（包含 basic, sgd, adam, trait_tests 子模块）
