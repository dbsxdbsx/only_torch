# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| input_1 | Input | [256, 784] | - | - |
| input_2 | Input | [256, 64] | - | - |
| input_3 | Input | [256, 1] | - | - |
| g_w1 | Parameter | [64, 128] | 8,192 | - |
| g_b1 | Parameter | [1, 128] | 128 | - |
| mat_mul_1 | MatMul | [256, 128] | - | input_2, g_w1 |
| mat_mul_2 | MatMul | [256, 128] | - | input_3, g_b1 |
| add_1 | Add | [256, 128] | - | mat_mul_1, mat_mul_2 |
| leaky_relu_1 | LeakyReLU | [256, 128] | - | add_1 |
| g_w2 | Parameter | [128, 784] | 100,352 | - |
| g_b2 | Parameter | [1, 784] | 784 | - |
| mat_mul_3 | MatMul | [256, 784] | - | leaky_relu_1, g_w2 |
| mat_mul_4 | MatMul | [256, 784] | - | input_3, g_b2 |
| add_2 | Add | [256, 784] | - | mat_mul_3, mat_mul_4 |
| sigmoid_1 | Sigmoid | [256, 784] | - | add_2 |
| d_w1 | Parameter | [784, 128] | 100,352 | - |
| d_b1 | Parameter | [1, 128] | 128 | - |
| mat_mul_5 | MatMul | [256, 128] | - | input_1, d_w1 |
| mat_mul_6 | MatMul | [256, 128] | - | input_3, d_b1 |
| add_3 | Add | [256, 128] | - | mat_mul_5, mat_mul_6 |
| leaky_relu_2 | LeakyReLU | [256, 128] | - | add_3 |
| d_w2 | Parameter | [128, 1] | 128 | - |
| d_b2 | Parameter | [1, 1] | 1 | - |
| mat_mul_7 | MatMul | [256, 1] | - | leaky_relu_2, d_w2 |
| mat_mul_8 | MatMul | [256, 1] | - | input_3, d_b2 |
| add_4 | Add | [256, 1] | - | mat_mul_7, mat_mul_8 |
| sigmoid_2 | Sigmoid | [256, 1] | - | add_4 |
| mat_mul_9 | MatMul | [256, 128] | - | sigmoid_1, d_w1 |
| mat_mul_10 | MatMul | [256, 128] | - | input_3, d_b1 |
| add_5 | Add | [256, 128] | - | mat_mul_9, mat_mul_10 |
| leaky_relu_3 | LeakyReLU | [256, 128] | - | add_5 |
| mat_mul_11 | MatMul | [256, 1] | - | leaky_relu_3, d_w2 |
| mat_mul_12 | MatMul | [256, 1] | - | input_3, d_b2 |
| add_6 | Add | [256, 1] | - | mat_mul_11, mat_mul_12 |
| sigmoid_3 | Sigmoid | [256, 1] | - | add_6 |
| input_4 | Input | [256, 1] | - | - |
| input_5 | Input | [256, 1] | - | - |
| mse_loss_1 | MSELoss | [1, 1] | - | sigmoid_2, input_4 |
| mse_loss_2 | MSELoss | [1, 1] | - | sigmoid_3, input_5 |
| mse_loss_3 | MSELoss | [1, 1] | - | sigmoid_3, input_4 |

**总参数量**: 210,065  
**可训练参数**: 210,065
