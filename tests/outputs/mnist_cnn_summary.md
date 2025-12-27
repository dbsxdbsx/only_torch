# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| x | Input | [512, 1, 28, 28] | - | - |
| y | Input | [512, 10] | - | - |
| conv1_K | Parameter | [8, 1, 5, 5] | 200 | - |
| conv1_conv | Conv2d | [512, 8, 28, 28] | - | x, conv1_K |
| conv1_b | Parameter | [1, 8] | 8 | - |
| conv1_out | ChBiasAdd | [512, 8, 28, 28] | - | conv1_conv, conv1_b |
| relu1 | LeakyReLU | [512, 8, 28, 28] | - | conv1_out |
| avg_pool1_out | AvgPool2d | [512, 8, 14, 14] | - | relu1 |
| conv2_K | Parameter | [16, 8, 3, 3] | 1,152 | - |
| conv2_conv | Conv2d | [512, 16, 14, 14] | - | avg_pool1_out, conv2_K |
| conv2_b | Parameter | [1, 16] | 16 | - |
| conv2_out | ChBiasAdd | [512, 16, 14, 14] | - | conv2_conv, conv2_b |
| relu2 | LeakyReLU | [512, 16, 14, 14] | - | conv2_out |
| max_pool2_out | MaxPool2d | [512, 16, 7, 7] | - | relu2 |
| flatten | Flatten | [512, 784] | - | max_pool2_out |
| fc1_W | Parameter | [784, 64] | 50,176 | - |
| fc1_b | Parameter | [1, 64] | 64 | - |
| fc1_ones | Input | [512, 1] | - | - |
| fc1_xW | MatMul | [512, 64] | - | flatten, fc1_W |
| fc1_b_broadcast | MatMul | [512, 64] | - | fc1_ones, fc1_b |
| fc1_out | Add | [512, 64] | - | fc1_xW, fc1_b_broadcast |
| relu3 | LeakyReLU | [512, 64] | - | fc1_out |
| fc2_W | Parameter | [64, 10] | 640 | - |
| fc2_b | Parameter | [1, 10] | 10 | - |
| fc2_ones | Input | [512, 1] | - | - |
| fc2_xW | MatMul | [512, 10] | - | relu3, fc2_W |
| fc2_b_broadcast | MatMul | [512, 10] | - | fc2_ones, fc2_b |
| fc2_out | Add | [512, 10] | - | fc2_xW, fc2_b_broadcast |
| loss | SoftmaxCE | [1, 1] | - | fc2_out, y |

**总参数量**: 52,266  
**可训练参数**: 52,266
