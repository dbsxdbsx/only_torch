# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| input_1 | Input | [256, 8] | - | - |
| input_2 | Input | [256, 1] | - | - |
| fc1_W | Parameter | [8, 128] | 1,024 | - |
| fc1_b | Parameter | [1, 128] | 128 | - |
| fc2_W | Parameter | [128, 64] | 8,192 | - |
| fc2_b | Parameter | [1, 64] | 64 | - |
| fc3_W | Parameter | [64, 32] | 2,048 | - |
| fc3_b | Parameter | [1, 32] | 32 | - |
| fc4_W | Parameter | [32, 1] | 32 | - |
| fc4_b | Parameter | [1, 1] | 1 | - |
| mat_mul_1 | MatMul | [256, 128] | - | input_1, fc1_W |
| add_1 | Add | [256, 128] | - | mat_mul_1, fc1_b |
| softplus_1 | SoftPlus | [256, 128] | - | add_1 |
| mat_mul_2 | MatMul | [256, 64] | - | softplus_1, fc2_W |
| add_2 | Add | [256, 64] | - | mat_mul_2, fc2_b |
| softplus_2 | SoftPlus | [256, 64] | - | add_2 |
| mat_mul_3 | MatMul | [256, 32] | - | softplus_2, fc3_W |
| add_3 | Add | [256, 32] | - | mat_mul_3, fc3_b |
| softplus_3 | SoftPlus | [256, 32] | - | add_3 |
| mat_mul_4 | MatMul | [256, 1] | - | softplus_3, fc4_W |
| add_4 | Add | [256, 1] | - | mat_mul_4, fc4_b |
| mse_loss_1 | MSELoss | [1, 1] | - | add_4, input_2 |

**总参数量**: 11,521  
**可训练参数**: 11,521
