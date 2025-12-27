# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| x | Input | [1, 1] | - | - |
| w | Parameter | [1, 1] | 1 | - |
| b | Parameter | [1, 1] | 1 | - |
| xw | MatMul | [1, 1] | - | x, w |
| ones | Input | [1, 1] | - | - |
| bias_broadcast | MatMul | [1, 1] | - | ones, b |
| y_pred | Add | [1, 1] | - | xw, bias_broadcast |
| y_true | Input | [1, 1] | - | - |
| loss | MSELoss | [1, 1] | - | y_pred, y_true |

**总参数量**: 2  
**可训练参数**: 2
