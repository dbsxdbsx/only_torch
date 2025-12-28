# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| x | Input | [3, 1] | - | - |
| label | Input | [1, 1] | - | - |
| w | Parameter | [1, 3] | 3 | - |
| b | Parameter | [1, 1] | 1 | - |
| mat_mul_1 | MatMul | [1, 1] | - | w, x |
| add_1 | Add | [1, 1] | - | mat_mul_1, b |
| sign_1 | Sign | [1, 1] | - | add_1 |
| loss_input | MatMul | [1, 1] | - | label, add_1 |
| loss | PerceptionLoss | [1, 1] | - | loss_input |

**总参数量**: 4  
**可训练参数**: 4
