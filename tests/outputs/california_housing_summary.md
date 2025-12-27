# 模型摘要: default_graph

| 节点名称 | 类型 | 输出形状 | 参数量 | 父节点 |
|----------|------|----------|--------|--------|
| x | Input | [256, 8] | - | - |
| y_true | Input | [256, 1] | - | - |
| fc1_W | Parameter | [8, 128] | 1,024 | - |
| fc1_b | Parameter | [1, 128] | 128 | - |
| fc1_ones | Input | [256, 1] | - | - |
| fc1_xW | MatMul | [256, 128] | - | x, fc1_W |
| fc1_b_broadcast | MatMul | [256, 128] | - | fc1_ones, fc1_b |
| fc1_out | Add | [256, 128] | - | fc1_xW, fc1_b_broadcast |
| fc1_act | SoftPlus | [256, 128] | - | fc1_out |
| fc2_W | Parameter | [128, 64] | 8,192 | - |
| fc2_b | Parameter | [1, 64] | 64 | - |
| fc2_ones | Input | [256, 1] | - | - |
| fc2_xW | MatMul | [256, 64] | - | fc1_act, fc2_W |
| fc2_b_broadcast | MatMul | [256, 64] | - | fc2_ones, fc2_b |
| fc2_out | Add | [256, 64] | - | fc2_xW, fc2_b_broadcast |
| fc2_act | SoftPlus | [256, 64] | - | fc2_out |
| fc3_W | Parameter | [64, 32] | 2,048 | - |
| fc3_b | Parameter | [1, 32] | 32 | - |
| fc3_ones | Input | [256, 1] | - | - |
| fc3_xW | MatMul | [256, 32] | - | fc2_act, fc3_W |
| fc3_b_broadcast | MatMul | [256, 32] | - | fc3_ones, fc3_b |
| fc3_out | Add | [256, 32] | - | fc3_xW, fc3_b_broadcast |
| fc3_act | SoftPlus | [256, 32] | - | fc3_out |
| fc4_W | Parameter | [32, 1] | 32 | - |
| fc4_b | Parameter | [1, 1] | 1 | - |
| fc4_ones | Input | [256, 1] | - | - |
| fc4_xW | MatMul | [256, 1] | - | fc3_act, fc4_W |
| fc4_b_broadcast | MatMul | [256, 1] | - | fc4_ones, fc4_b |
| fc4_out | Add | [256, 1] | - | fc4_xW, fc4_b_broadcast |
| loss | MSELoss | [1, 1] | - | fc4_out, y_true |

**总参数量**: 11,521  
**可训练参数**: 11,521
