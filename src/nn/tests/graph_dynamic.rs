/*
 * @Author       : 老董
 * @Date         : 2025-12-21
 * @Description  : M4 测试 - 验证 Graph 的动态扩展能力（NEAT 友好性）
 *                 测试在 forward/backward 后动态添加节点的能力
 * @LastEditors  : 老董
 * @LastEditTime : 2025-12-21
 */

use crate::nn::Graph;
use crate::tensor::Tensor;

// ============================================================================
// 基础动态添加测试
// ============================================================================

/// 测试: 在 forward 后添加新节点并继续计算
#[test]
fn test_add_node_after_forward() {
    let mut graph = Graph::new();

    // 1. 创建初始图: add1 = a + b
    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add1 = graph.new_add_node(&[a, b], Some("add1")).unwrap();

    // 2. 第1次 forward
    graph.forward_node(add1).unwrap();
    let add1_value_before = graph.get_node_value(add1).unwrap().unwrap().clone();
    let first_forward_pass_id = graph.last_forward_pass_id();
    assert_eq!(first_forward_pass_id, 1);

    // 3. 动态添加新节点: add2 = add1 + c
    let c = graph
        .new_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let add2 = graph.new_add_node(&[add1, c], Some("add2")).unwrap();

    // 4. 验证新节点的 pass_id 是 0（还未参与计算）
    assert_eq!(graph.get_node(add2).unwrap().last_forward_pass_id(), 0);

    // 5. 对新节点进行 forward
    graph.forward_node(add2).unwrap();
    let second_forward_pass_id = graph.last_forward_pass_id();
    assert_eq!(second_forward_pass_id, 2);

    // 6. 验证计算结果正确
    let a_value = graph.get_node_value(a).unwrap().unwrap();
    let b_value = graph.get_node_value(b).unwrap().unwrap();
    let c_value = graph.get_node_value(c).unwrap().unwrap();
    let add1_value = graph.get_node_value(add1).unwrap().unwrap();
    let add2_value = graph.get_node_value(add2).unwrap().unwrap();

    // add1 = a + b
    let expected_add1 = a_value.clone() + b_value.clone();
    assert_eq!(add1_value, &expected_add1);

    // add2 = add1 + c
    let expected_add2 = expected_add1 + c_value.clone();
    assert_eq!(add2_value, &expected_add2);

    // 7. 验证原始节点的值没有被意外修改
    assert_eq!(&add1_value_before, add1_value);
}

/// 测试: 在 backward 后添加新节点并继续训练
#[test]
fn test_add_node_after_backward() {
    let mut graph = Graph::new();

    // 1. 创建初始图: y = w * x + b
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w = graph
        .new_parameter_node_seeded(&[1, 2], Some("w"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[1, 1], Some("b"), 43)
        .unwrap();
    let wx = graph.new_mat_mul_node(w, x, Some("wx")).unwrap();
    let y = graph.new_add_node(&[wx, b], Some("y")).unwrap();

    // 2. 设置输入并进行一轮训练
    let x_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.forward_node(y).unwrap();
    graph.backward_nodes(&[w, b], y).unwrap();

    // 3. 验证反向传播成功
    assert!(graph.get_node_jacobi(w).unwrap().is_some());
    assert!(graph.get_node_jacobi(b).unwrap().is_some());
    let w_jacobi_before = graph.get_node_jacobi(w).unwrap().unwrap().clone();

    // 4. 动态添加新节点: z = y + c
    let c = graph
        .new_parameter_node_seeded(&[1, 1], Some("c"), 44)
        .unwrap();
    let z = graph.new_add_node(&[y, c], Some("z")).unwrap();

    // 5. 通知图拓扑已变化
    graph.on_topology_changed();

    // 6. 验证 Jacobi 已被清除
    assert!(graph.get_node_jacobi(w).unwrap().is_none());
    assert!(graph.get_node_jacobi(b).unwrap().is_none());

    // 7. 验证值仍然保留
    assert!(graph.get_node_value(y).unwrap().is_some());

    // 8. 对扩展后的图进行新一轮训练
    graph.forward_node(z).unwrap();
    graph.backward_nodes(&[w, b, c], z).unwrap();

    // 9. 验证所有参数都有新的 Jacobi
    assert!(graph.get_node_jacobi(w).unwrap().is_some());
    assert!(graph.get_node_jacobi(b).unwrap().is_some());
    assert!(graph.get_node_jacobi(c).unwrap().is_some());

    // 10. 验证新旧 Jacobi 不同（因为结果节点变了）
    let w_jacobi_after = graph.get_node_jacobi(w).unwrap().unwrap();
    // 由于 z = y + c，dz/dw = dy/dw，所以实际上 Jacobi 应该相同
    // 但形状可能不同，这里主要验证能正常计算
    assert_eq!(w_jacobi_after, &w_jacobi_before);
}

// ============================================================================
// 多次拓扑变化测试
// ============================================================================

/// 测试: 连续多次添加节点
#[test]
fn test_multiple_topology_changes() {
    let mut graph = Graph::new();

    // 1. 初始图: node1 = param
    let param = graph
        .new_parameter_node_seeded(&[2, 1], Some("param"), 42)
        .unwrap();
    let input = graph.new_input_node(&[2, 1], Some("input")).unwrap();
    let node1 = graph.new_add_node(&[param, input], Some("node1")).unwrap();

    // 设置输入
    let input_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    // 2. 第1轮训练
    graph.forward_node(node1).unwrap();
    graph.backward_nodes(&[param], node1).unwrap();
    let value1 = graph.get_node_value(node1).unwrap().unwrap().clone();

    // 3. 第1次拓扑变化: 添加 node2 = tanh(node1)
    let node2 = graph.new_tanh_node(node1, Some("node2")).unwrap();
    graph.on_topology_changed();

    // 4. 第2轮训练
    graph.forward_node(node2).unwrap();
    graph.backward_nodes(&[param], node2).unwrap();
    let value2 = graph.get_node_value(node2).unwrap().unwrap().clone();

    // 5. 第2次拓扑变化: 添加 node3 = node2 + bias
    let bias = graph
        .new_parameter_node_seeded(&[2, 1], Some("bias"), 43)
        .unwrap();
    let node3 = graph.new_add_node(&[node2, bias], Some("node3")).unwrap();
    graph.on_topology_changed();

    // 6. 第3轮训练
    graph.forward_node(node3).unwrap();
    graph.backward_nodes(&[param, bias], node3).unwrap();
    let value3 = graph.get_node_value(node3).unwrap().unwrap().clone();

    // 7. 验证计算链正确
    // node1 的值应该没变
    assert_eq!(graph.get_node_value(node1).unwrap().unwrap(), &value1);

    // node2 = tanh(node1)
    let expected_node2 = value1.tanh();
    assert_eq!(&value2, &expected_node2);

    // node3 = node2 + bias
    let bias_value = graph.get_node_value(bias).unwrap().unwrap();
    let expected_node3 = expected_node2 + bias_value.clone();
    assert_eq!(&value3, &expected_node3);
}

/// 测试: 在同一个父节点上添加多个子节点（分支）
#[test]
fn test_add_multiple_branches() {
    let mut graph = Graph::new();

    // 1. 初始图
    let input = graph.new_input_node(&[2, 1], Some("input")).unwrap();
    let param = graph
        .new_parameter_node_seeded(&[2, 1], Some("param"), 42)
        .unwrap();
    let base = graph.new_add_node(&[input, param], Some("base")).unwrap();

    // 设置输入
    let input_value = Tensor::new(&[1.0, 2.0], &[2, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();

    // 2. 初始 forward
    graph.forward_node(base).unwrap();

    // 3. 添加分支1: branch1 = tanh(base)
    let branch1 = graph.new_tanh_node(base, Some("branch1")).unwrap();

    // 4. 添加分支2: branch2 = tanh(base) (使用另一个 tanh 以便测试)
    let branch2 = graph.new_tanh_node(base, Some("branch2")).unwrap();

    // 5. 合并分支: merged = branch1 + branch2
    let merged = graph
        .new_add_node(&[branch1, branch2], Some("merged"))
        .unwrap();

    // 6. 验证 base 现在有多个子节点
    let base_children = graph.get_node_children(base).unwrap();
    assert_eq!(base_children.len(), 2);
    assert!(base_children.contains(&branch1));
    assert!(base_children.contains(&branch2));

    // 7. Forward 合并节点
    graph.forward_node(merged).unwrap();

    // 8. 验证计算正确
    let base_value = graph.get_node_value(base).unwrap().unwrap();
    let branch1_value = graph.get_node_value(branch1).unwrap().unwrap();
    let branch2_value = graph.get_node_value(branch2).unwrap().unwrap();
    let merged_value = graph.get_node_value(merged).unwrap().unwrap();

    let expected_branch = base_value.tanh();
    // branch1 和 branch2 都是 tanh(base)，所以应该相等
    assert_eq!(branch1_value, &expected_branch);
    assert_eq!(branch2_value, &expected_branch);

    // merged = branch1 + branch2 = 2 * tanh(base)
    let expected_merged = expected_branch.clone() + expected_branch.clone();
    assert_eq!(merged_value, &expected_merged);
}

// ============================================================================
// 边界情况测试
// ============================================================================

/// 测试: 链式添加节点（A -> B -> C -> D）
#[test]
fn test_chain_node_addition() {
    let mut graph = Graph::new();

    // 1. 创建初始节点
    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();

    // 2. 逐步构建链: result = tanh(tanh(tanh(a + b)))
    let add = graph.new_add_node(&[a, b], Some("add")).unwrap();
    graph.forward_node(add).unwrap();

    let tanh1 = graph.new_tanh_node(add, Some("tanh1")).unwrap();
    graph.forward_node(tanh1).unwrap();

    let tanh2 = graph.new_tanh_node(tanh1, Some("tanh2")).unwrap();
    graph.forward_node(tanh2).unwrap();

    let tanh3 = graph.new_tanh_node(tanh2, Some("tanh3")).unwrap();
    graph.forward_node(tanh3).unwrap();

    // 3. 验证链式计算正确
    let a_value = graph.get_node_value(a).unwrap().unwrap();
    let b_value = graph.get_node_value(b).unwrap().unwrap();

    let expected = (a_value.clone() + b_value.clone()).tanh().tanh().tanh();
    let actual = graph.get_node_value(tanh3).unwrap().unwrap();

    assert_eq!(actual, &expected);

    // 4. 反向传播
    graph.backward_nodes(&[a, b], tanh3).unwrap();

    // 5. 验证梯度存在
    assert!(graph.get_node_jacobi(a).unwrap().is_some());
    assert!(graph.get_node_jacobi(b).unwrap().is_some());
}

/// 测试: 在已有复杂图上添加节点
#[test]
fn test_add_to_complex_graph() {
    let mut graph = Graph::new();

    // 1. 创建一个类似 XOR 的网络结构
    let x = graph.new_input_node(&[2, 1], Some("x")).unwrap();
    let w1 = graph
        .new_parameter_node_seeded(&[4, 2], Some("w1"), 42)
        .unwrap();
    let b1 = graph
        .new_parameter_node_seeded(&[4, 1], Some("b1"), 43)
        .unwrap();
    let w2 = graph
        .new_parameter_node_seeded(&[1, 4], Some("w2"), 44)
        .unwrap();
    let b2 = graph
        .new_parameter_node_seeded(&[1, 1], Some("b2"), 45)
        .unwrap();

    // 隐藏层
    let wx1 = graph.new_mat_mul_node(w1, x, None).unwrap();
    let z1 = graph.new_add_node(&[wx1, b1], None).unwrap();
    let h = graph.new_tanh_node(z1, Some("hidden")).unwrap();

    // 输出层
    let wx2 = graph.new_mat_mul_node(w2, h, None).unwrap();
    let output = graph.new_add_node(&[wx2, b2], Some("output")).unwrap();

    // 2. 进行一轮训练
    let x_value = Tensor::new(&[1.0, 0.0], &[2, 1]);
    graph.set_node_value(x, Some(&x_value)).unwrap();
    graph.forward_node(output).unwrap();
    graph.backward_nodes(&[w1, b1, w2, b2], output).unwrap();

    // 3. 动态添加一个新的隐藏层节点（NEAT 变异：添加节点）
    let w_new = graph
        .new_parameter_node_seeded(&[1, 4], Some("w_new"), 46)
        .unwrap();
    let new_hidden_out = graph
        .new_mat_mul_node(w_new, h, Some("new_hidden_out"))
        .unwrap();

    // 4. 创建新的输出（原输出 + 新隐藏层输出）
    // 注意：这需要形状匹配，这里简化为直接添加
    let combined = graph
        .new_add_node(&[output, new_hidden_out], Some("combined"))
        .unwrap();

    // 5. 通知拓扑变化
    graph.on_topology_changed();

    // 6. 对新图进行训练
    graph.forward_node(combined).unwrap();
    graph
        .backward_nodes(&[w1, b1, w2, b2, w_new], combined)
        .unwrap();

    // 7. 验证所有参数都有梯度
    assert!(graph.get_node_jacobi(w1).unwrap().is_some());
    assert!(graph.get_node_jacobi(b1).unwrap().is_some());
    assert!(graph.get_node_jacobi(w2).unwrap().is_some());
    assert!(graph.get_node_jacobi(b2).unwrap().is_some());
    assert!(graph.get_node_jacobi(w_new).unwrap().is_some());
}

/// 测试: on_topology_changed 多次调用
#[test]
fn test_multiple_on_topology_changed_calls() {
    let mut graph = Graph::new();

    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add = graph.new_add_node(&[a, b], None).unwrap();

    // Forward 和 backward
    graph.forward_node(add).unwrap();
    graph.backward_nodes(&[a, b], add).unwrap();

    // 多次调用 on_topology_changed 应该是安全的
    graph.on_topology_changed();
    graph.on_topology_changed();
    graph.on_topology_changed();

    // 验证状态正确
    assert!(graph.get_node_jacobi(a).unwrap().is_none());
    assert!(graph.get_node_jacobi(b).unwrap().is_none());

    // 验证值仍然保留
    assert!(graph.get_node_value(add).unwrap().is_some());

    // 可以继续训练
    graph.forward_node(add).unwrap();
    graph.backward_nodes(&[a, b], add).unwrap();

    assert!(graph.get_node_jacobi(a).unwrap().is_some());
    assert!(graph.get_node_jacobi(b).unwrap().is_some());
}

/// 测试: 添加节点后不调用 on_topology_changed 的情况
/// 验证即使不显式调用，pass_id 机制也能保证正确性
#[test]
fn test_add_node_without_explicit_topology_changed() {
    let mut graph = Graph::new();

    // 1. 创建初始图
    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add1 = graph.new_add_node(&[a, b], None).unwrap();

    // 2. 训练
    graph.forward_node(add1).unwrap();
    graph.backward_nodes(&[a, b], add1).unwrap();

    // 3. 添加新节点但不调用 on_topology_changed
    let c = graph
        .new_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let add2 = graph.new_add_node(&[add1, c], None).unwrap();

    // 4. 直接对新节点进行 forward/backward
    // 注意：旧的 Jacobi 仍然存在，但 pass_id 机制会确保重新计算
    graph.forward_node(add2).unwrap();

    // 旧的 Jacobi 仍然存在（因为没有显式清除）
    // 这是一个潜在的问题 - 用户应该调用 on_topology_changed
    // 但 forward 本身是正确的
    let add2_value = graph.get_node_value(add2).unwrap().unwrap();
    let a_value = graph.get_node_value(a).unwrap().unwrap();
    let b_value = graph.get_node_value(b).unwrap().unwrap();
    let c_value = graph.get_node_value(c).unwrap().unwrap();

    let expected = a_value.clone() + b_value.clone() + c_value.clone();
    assert_eq!(add2_value, &expected);

    // 5. 对新结果节点进行 backward
    // 这里清除 Jacobi 以确保正确计算
    graph.clear_jacobi().unwrap();
    graph.backward_nodes(&[a, b, c], add2).unwrap();

    // 验证所有参数都有正确的 Jacobi
    let a_jacobi = graph.get_node_jacobi(a).unwrap().unwrap();
    let b_jacobi = graph.get_node_jacobi(b).unwrap().unwrap();
    let c_jacobi = graph.get_node_jacobi(c).unwrap().unwrap();

    // 对于 add2 = a + b + c，d(add2)/da = d(add2)/db = d(add2)/dc = I
    let expected_jacobi = Tensor::eyes(2);
    assert_eq!(a_jacobi, &expected_jacobi);
    assert_eq!(b_jacobi, &expected_jacobi);
    assert_eq!(c_jacobi, &expected_jacobi);
}

// ============================================================================
// NEAT 典型场景测试
// ============================================================================

/// 测试: 模拟 NEAT 的"添加节点"变异
/// NEAT 中添加节点是在现有连接中间插入一个新节点
/// 原始: A -> B 变成 A -> NEW -> B
/// 注意: 当前 API 不直接支持"插入"，需要重建连接
#[test]
fn test_neat_add_node_mutation_simulation() {
    let mut graph = Graph::new();

    // 1. 初始网络: input -> hidden -> output
    let input = graph.new_input_node(&[2, 1], Some("input")).unwrap();
    let w1 = graph
        .new_parameter_node_seeded(&[3, 2], Some("w1"), 42)
        .unwrap();
    let hidden = graph.new_mat_mul_node(w1, input, Some("hidden")).unwrap();
    let w2 = graph
        .new_parameter_node_seeded(&[1, 3], Some("w2"), 43)
        .unwrap();
    let output = graph.new_mat_mul_node(w2, hidden, Some("output")).unwrap();

    // 2. 初始训练
    let input_value = Tensor::new(&[1.0, 0.5], &[2, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();
    graph.forward_node(output).unwrap();
    graph.backward_nodes(&[w1, w2], output).unwrap();

    // 3. NEAT 变异: 在 hidden 和 output 之间添加新节点
    // 由于不能修改现有连接，我们通过添加旁路来模拟:
    // hidden -> new_node -> new_output
    //        \-> w2 -> output (原路径) ->/
    let w_new = graph
        .new_parameter_node_seeded(&[3, 3], Some("w_new"), 44)
        .unwrap();
    let new_hidden = graph
        .new_mat_mul_node(w_new, hidden, Some("new_hidden"))
        .unwrap();
    let tanh_new = graph.new_tanh_node(new_hidden, Some("tanh_new")).unwrap();

    // 将新节点的输出与原输出合并
    // 注意: 需要调整形状，这里假设 tanh_new 是 [3,1]，output 是 [1,1]
    // 为简化测试，我们创建一个新的输出节点
    let w3 = graph
        .new_parameter_node_seeded(&[1, 3], Some("w3"), 45)
        .unwrap();
    let new_output = graph
        .new_mat_mul_node(w3, tanh_new, Some("new_output"))
        .unwrap();
    let final_output = graph
        .new_add_node(&[output, new_output], Some("final_output"))
        .unwrap();

    // 4. 通知拓扑变化
    graph.on_topology_changed();

    // 5. 对新图进行训练
    graph.forward_node(final_output).unwrap();
    graph
        .backward_nodes(&[w1, w2, w_new, w3], final_output)
        .unwrap();

    // 6. 验证所有参数都有梯度
    assert!(graph.get_node_jacobi(w1).unwrap().is_some());
    assert!(graph.get_node_jacobi(w2).unwrap().is_some());
    assert!(graph.get_node_jacobi(w_new).unwrap().is_some());
    assert!(graph.get_node_jacobi(w3).unwrap().is_some());

    // 7. 验证图的节点数量增加
    // input, w1, hidden, w2, output = 5
    // + w_new, new_hidden, tanh_new, w3, new_output, final_output = 6
    // 总计 11 个节点
    assert_eq!(graph.nodes_count(), 11);
}

/// 测试: 模拟 NEAT 的"添加连接"变异
/// 在两个已存在但未连接的节点之间添加连接
#[test]
fn test_neat_add_connection_mutation_simulation() {
    let mut graph = Graph::new();

    // 1. 创建一个有多个并行路径的网络
    let input = graph.new_input_node(&[2, 1], Some("input")).unwrap();

    // 路径1: input -> w1 -> h1
    let w1 = graph
        .new_parameter_node_seeded(&[2, 2], Some("w1"), 42)
        .unwrap();
    let h1 = graph.new_mat_mul_node(w1, input, Some("h1")).unwrap();
    let h1_tanh = graph.new_tanh_node(h1, Some("h1_tanh")).unwrap();

    // 路径2: input -> w2 -> h2
    let w2 = graph
        .new_parameter_node_seeded(&[2, 2], Some("w2"), 43)
        .unwrap();
    let h2 = graph.new_mat_mul_node(w2, input, Some("h2")).unwrap();
    let h2_tanh = graph.new_tanh_node(h2, Some("h2_tanh")).unwrap();

    // 输出: h1 + h2
    let output = graph
        .new_add_node(&[h1_tanh, h2_tanh], Some("output"))
        .unwrap();

    // 2. 初始训练
    let input_value = Tensor::new(&[1.0, 0.5], &[2, 1]);
    graph.set_node_value(input, Some(&input_value)).unwrap();
    graph.forward_node(output).unwrap();
    graph.backward_nodes(&[w1, w2], output).unwrap();

    // 3. NEAT 变异: 添加从 h1 到 h2 的连接（跨路径连接）
    // 新结构: h2_new = h2 + w3 * h1
    let w3 = graph
        .new_parameter_node_seeded(&[2, 2], Some("w3"), 44)
        .unwrap();
    let cross_conn = graph.new_mat_mul_node(w3, h1, Some("cross_conn")).unwrap();
    let h2_enhanced = graph
        .new_add_node(&[h2, cross_conn], Some("h2_enhanced"))
        .unwrap();
    let h2_enhanced_tanh = graph
        .new_tanh_node(h2_enhanced, Some("h2_enhanced_tanh"))
        .unwrap();

    // 新输出
    let new_output = graph
        .new_add_node(&[h1_tanh, h2_enhanced_tanh], Some("new_output"))
        .unwrap();

    // 4. 通知拓扑变化
    graph.on_topology_changed();

    // 5. 对新图进行训练
    graph.forward_node(new_output).unwrap();
    graph.backward_nodes(&[w1, w2, w3], new_output).unwrap();

    // 6. 验证所有参数都有梯度
    assert!(graph.get_node_jacobi(w1).unwrap().is_some());
    assert!(graph.get_node_jacobi(w2).unwrap().is_some());
    assert!(graph.get_node_jacobi(w3).unwrap().is_some());
}

// ============================================================================
// 稳定性和正确性测试
// ============================================================================

/// 测试: 验证动态添加后的梯度数值正确性
#[test]
fn test_gradient_correctness_after_dynamic_add() {
    let mut graph = Graph::new();

    // 1. 创建初始图: y = a + b
    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let y = graph.new_add_node(&[a, b], Some("y")).unwrap();

    // 2. Forward 和 backward
    graph.forward_node(y).unwrap();
    graph.backward_nodes(&[a, b], y).unwrap();

    // 保存原始 Jacobi
    let a_jacobi_original = graph.get_node_jacobi(a).unwrap().unwrap().clone();
    let b_jacobi_original = graph.get_node_jacobi(b).unwrap().unwrap().clone();

    // 3. 添加新节点: z = y + c = a + b + c
    let c = graph
        .new_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let z = graph.new_add_node(&[y, c], Some("z")).unwrap();

    // 4. 清除旧 Jacobi
    graph.on_topology_changed();

    // 5. 对新图进行 forward/backward
    graph.forward_node(z).unwrap();
    graph.backward_nodes(&[a, b, c], z).unwrap();

    // 6. 验证 Jacobi 的数值正确性
    // 对于 z = a + b + c，dz/da = dz/db = dz/dc = I
    let expected_jacobi = Tensor::eyes(2);

    let a_jacobi_new = graph.get_node_jacobi(a).unwrap().unwrap();
    let b_jacobi_new = graph.get_node_jacobi(b).unwrap().unwrap();
    let c_jacobi_new = graph.get_node_jacobi(c).unwrap().unwrap();

    assert_eq!(a_jacobi_new, &expected_jacobi);
    assert_eq!(b_jacobi_new, &expected_jacobi);
    assert_eq!(c_jacobi_new, &expected_jacobi);

    // 7. 验证原始参数的 Jacobi 与添加节点前相同
    // （因为 d(a+b+c)/da = d(a+b)/da = I）
    assert_eq!(a_jacobi_new, &a_jacobi_original);
    assert_eq!(b_jacobi_new, &b_jacobi_original);
}

/// 测试: 验证 forward pass ID 在动态添加后的行为
#[test]
fn test_pass_id_behavior_after_dynamic_add() {
    let mut graph = Graph::new();

    // 1. 创建初始图
    let a = graph
        .new_parameter_node_seeded(&[2, 1], Some("a"), 42)
        .unwrap();
    let b = graph
        .new_parameter_node_seeded(&[2, 1], Some("b"), 43)
        .unwrap();
    let add1 = graph.new_add_node(&[a, b], None).unwrap();

    // 2. Forward
    graph.forward_node(add1).unwrap();
    let pass_id_after_first_forward = graph.last_forward_pass_id();
    assert_eq!(pass_id_after_first_forward, 1);

    // 验证节点的 pass_id
    assert_eq!(graph.get_node(a).unwrap().last_forward_pass_id(), 1);
    assert_eq!(graph.get_node(b).unwrap().last_forward_pass_id(), 1);
    assert_eq!(graph.get_node(add1).unwrap().last_forward_pass_id(), 1);

    // 3. 添加新节点
    let c = graph
        .new_parameter_node_seeded(&[2, 1], Some("c"), 44)
        .unwrap();
    let add2 = graph.new_add_node(&[add1, c], None).unwrap();

    // 新节点的 pass_id 应该是 0
    assert_eq!(graph.get_node(c).unwrap().last_forward_pass_id(), 0);
    assert_eq!(graph.get_node(add2).unwrap().last_forward_pass_id(), 0);

    // 4. Forward 新节点
    graph.forward_node(add2).unwrap();
    let pass_id_after_second_forward = graph.last_forward_pass_id();
    assert_eq!(pass_id_after_second_forward, 2);

    // 所有相关节点的 pass_id 都应该更新
    assert_eq!(graph.get_node(a).unwrap().last_forward_pass_id(), 2);
    assert_eq!(graph.get_node(b).unwrap().last_forward_pass_id(), 2);
    assert_eq!(graph.get_node(c).unwrap().last_forward_pass_id(), 2);
    assert_eq!(graph.get_node(add1).unwrap().last_forward_pass_id(), 2);
    assert_eq!(graph.get_node(add2).unwrap().last_forward_pass_id(), 2);
}
