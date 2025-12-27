# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| x | Input | [512, 784] | - | - |
| y | Input | [512, 10] | - | - |
| fc1_W | Parameter | [784, 128] | 100,352 | - |
| fc1_b | Parameter | [1, 128] | 128 | - |
| fc1_ones | Input | [512, 1] | - | - |
| fc1_xW | MatMul | [512, 128] | - | x, fc1_W |
| fc1_b_broadcast | MatMul | [512, 128] | - | fc1_ones, fc1_b |
| fc1_out | Add | [512, 128] | - | fc1_xW, fc1_b_broadcast |
| fc1_act | SoftPlus | [512, 128] | - | fc1_out |
| fc2_W | Parameter | [128, 10] | 1,280 | - |
| fc2_b | Parameter | [1, 10] | 10 | - |
| fc2_ones | Input | [512, 1] | - | - |
| fc2_xW | MatMul | [512, 10] | - | fc1_act, fc2_W |
| fc2_b_broadcast | MatMul | [512, 10] | - | fc2_ones, fc2_b |
| fc2_out | Add | [512, 10] | - | fc2_xW, fc2_b_broadcast |
| loss | SoftmaxCE | [1, 1] | - | fc2_out, y |

**总参数量**: 101,770  
**可训练参数**: 101,770
