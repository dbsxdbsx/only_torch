# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| input_1 | Input | [7, 2] | - | - |
| w | Parameter | [2, 1] | 2 | - |
| b | Parameter | [1, 1] | 1 | - |
| mat_mul_1 | MatMul | [7, 1] | - | input_1, w |
| input_2 | Input | [7, 1] | - | - |
| mat_mul_2 | MatMul | [7, 1] | - | input_2, b |
| add_1 | Add | [7, 1] | - | mat_mul_1, mat_mul_2 |
| input_3 | Input | [7, 1] | - | - |
| mse_loss_1 | MSELoss | [1, 1] | - | add_1, input_3 |

**总参数量**: 3  
**可训练参数**: 3
