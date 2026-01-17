# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| input_1 | Input | [3, 1] | - | - |
| input_2 | Input | [1, 1] | - | - |
| w | Parameter | [1, 3] | 3 | - |
| b | Parameter | [1, 1] | 1 | - |
| mat_mul_1 | MatMul | [1, 1] | - | w, input_1 |
| add_1 | Add | [1, 1] | - | mat_mul_1, b |
| sign_1 | Sign | [1, 1] | - | add_1 |
| mat_mul_2 | MatMul | [1, 1] | - | input_2, add_1 |
| perception_loss_1 | PerceptionLoss | [1, 1] | - | mat_mul_2 |

**总参数量**: 4  
**可训练参数**: 4
