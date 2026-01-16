/*
 * @Description  : 循环/记忆机制基础测试（Phase 1）
 *
 * 测试 step()/reset()/connect_recurrent() API 的正确性
 */

use crate::assert_err;
use crate::nn::{GraphInner, GraphError, NodeId};
use crate::tensor::Tensor;

// ==================== 辅助函数 ====================

/// 创建标量张量
fn scalar(val: f64) -> Tensor {
    Tensor::new(&[val as f32], &[1, 1])
}

/// 创建一个简单的累加器网络
///
/// 结构：
/// ```text
///     input ──┐
///             ├──→ [Add] ──→ output
///  prev_out ──┘       │
///     ↑               │
///     └───────────────┘ (循环连接，延迟一步)
/// ```
///
/// 行为：output = input + prev_output
/// 如果 input 恒为 1，则 output 依次为 1, 2, 3, 4, ...
fn create_accumulator_graph() -> Result<(GraphInner, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new();

    // 输入节点（每步的新输入）
    let input = graph.new_input_node(&[1, 1], Some("input"))?;

    // 循环输入节点（接收上一步的输出）
    let prev_out = graph.new_input_node(&[1, 1], Some("prev_out"))?;

    // 初始化 prev_out 为 0（第一步时的初始状态）
    graph.set_node_value(prev_out, Some(&Tensor::zeros(&[1, 1])))?;

    // 加法节点：output = input + prev_out
    let output = graph.new_add_node(&[input, prev_out], Some("output"))?;

    // 声明循环连接：output 的值在下一步传给 prev_out
    graph.connect_recurrent(output, prev_out)?;

    Ok((graph, input, output))
}

// ==================== 基础功能测试 ====================

#[test]
fn test_accumulator_basic() {
    let (mut graph, input, output) = create_accumulator_graph().unwrap();

    // 验证初始状态
    assert_eq!(graph.current_time_step(), 0);
    assert!(graph.has_recurrent_edges());

    // Step 0: input=1, prev=0 → output=1
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!(
        (val.data_as_slice()[0] - 1.0).abs() < 1e-6,
        "第 0 步：期望 1.0，实际得到 {}",
        val.data_as_slice()[0]
    );
    assert_eq!(graph.current_time_step(), 1);

    // Step 1: input=1, prev=1 → output=2
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!(
        (val.data_as_slice()[0] - 2.0).abs() < 1e-6,
        "第 1 步：期望 2.0，实际得到 {}",
        val.data_as_slice()[0]
    );
    assert_eq!(graph.current_time_step(), 2);

    // Step 2: input=1, prev=2 → output=3
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!(
        (val.data_as_slice()[0] - 3.0).abs() < 1e-6,
        "第 2 步：期望 3.0，实际得到 {}",
        val.data_as_slice()[0]
    );
    assert_eq!(graph.current_time_step(), 3);

    // Step 3: input=1, prev=3 → output=4
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!(
        (val.data_as_slice()[0] - 4.0).abs() < 1e-6,
        "第 3 步：期望 4.0，实际得到 {}",
        val.data_as_slice()[0]
    );
    assert_eq!(graph.current_time_step(), 4);
}

#[test]
fn test_reset_clears_state() {
    let (mut graph, input, output) = create_accumulator_graph().unwrap();

    // 累加几步
    for _ in 0..5 {
        graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
        graph.step(output).unwrap();
    }

    // 验证累加到 5
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!((val.data_as_slice()[0] - 5.0).abs() < 1e-6);
    assert_eq!(graph.current_time_step(), 5);

    // 重置
    graph.reset();

    // 验证时间步归零
    assert_eq!(graph.current_time_step(), 0);

    // 重新开始累加，应该从 1 开始而不是 6
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert!(
        (val.data_as_slice()[0] - 1.0).abs() < 1e-6,
        "reset 后：期望 1.0，实际得到 {}",
        val.data_as_slice()[0]
    );
}

#[test]
fn test_variable_input() {
    let (mut graph, input, output) = create_accumulator_graph().unwrap();

    // 不同的输入值
    let inputs = [1.0, 2.0, 3.0, 0.0, -1.0];
    let expected = [1.0, 3.0, 6.0, 6.0, 5.0]; // 累加和

    for (i, (&inp, &exp)) in inputs.iter().zip(expected.iter()).enumerate() {
        graph.set_node_value(input, Some(&scalar(inp))).unwrap();
        graph.step(output).unwrap();
        let val = graph.get_node_value(output).unwrap().unwrap();
        assert!(
            (val.data_as_slice()[0] - exp as f32).abs() < 1e-6,
            "第 {} 步：期望 {}，实际得到 {}",
            i,
            exp,
            val.data_as_slice()[0]
        );
    }
}

// ==================== 衰减记忆测试 ====================

/// 创建一个带权重的循环网络（衰减记忆）
///
/// 结构：
/// ```text
///     input ─────────────────┐
///                            ├──→ [Add] ──→ output
///  prev_out ──→ [MatMul] ────┘        │
///     ↑          ↑                    │
///     │       weight                  │
///     └───────────────────────────────┘ (循环连接)
/// ```
///
/// 行为：output = input + weight * prev_output
fn create_weighted_recurrent_graph(weight: f64) -> Result<(GraphInner, NodeId, NodeId), GraphError> {
    let mut graph = GraphInner::new();

    // 输入节点
    let input = graph.new_input_node(&[1, 1], Some("input"))?;

    // 循环输入节点
    let prev_out = graph.new_input_node(&[1, 1], Some("prev_out"))?;
    graph.set_node_value(prev_out, Some(&Tensor::zeros(&[1, 1])))?;

    // 权重参数
    let w = graph.new_parameter_node(&[1, 1], Some("weight"))?;
    graph.set_node_value(w, Some(&scalar(weight)))?;

    // 加权：w * prev_out
    let weighted = graph.new_mat_mul_node(w, prev_out, Some("weighted"))?;

    // 相加：input + weighted
    let output = graph.new_add_node(&[input, weighted], Some("output"))?;

    // 循环连接
    graph.connect_recurrent(output, prev_out)?;

    Ok((graph, input, output))
}

#[test]
fn test_decaying_memory() {
    // 权重 0.5：每步衰减一半
    let (mut graph, input, output) = create_weighted_recurrent_graph(0.5).unwrap();

    // 脉冲输入：第一步为 1，后续为 0
    // 预期：1.0, 0.5, 0.25, 0.125, ...

    // Step 0: input=1 → output=1
    graph.set_node_value(input, Some(&scalar(1.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph
        .get_node_value(output)
        .unwrap()
        .unwrap()
        .data_as_slice()[0];
    assert!(
        (val - 1.0).abs() < 1e-6,
        "第 0 步：期望 1.0，实际得到 {}",
        val
    );

    // Step 1: input=0, prev=1 → output=0.5
    graph.set_node_value(input, Some(&scalar(0.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph
        .get_node_value(output)
        .unwrap()
        .unwrap()
        .data_as_slice()[0];
    assert!(
        (val - 0.5).abs() < 1e-6,
        "第 1 步：期望 0.5，实际得到 {}",
        val
    );

    // Step 2: input=0, prev=0.5 → output=0.25
    graph.set_node_value(input, Some(&scalar(0.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph
        .get_node_value(output)
        .unwrap()
        .unwrap()
        .data_as_slice()[0];
    assert!(
        (val - 0.25).abs() < 1e-6,
        "第 2 步：期望 0.25，实际得到 {}",
        val
    );

    // Step 3: input=0, prev=0.25 → output=0.125
    graph.set_node_value(input, Some(&scalar(0.0))).unwrap();
    graph.step(output).unwrap();
    let val = graph
        .get_node_value(output)
        .unwrap()
        .unwrap()
        .data_as_slice()[0];
    assert!(
        (val - 0.125).abs() < 1e-6,
        "第 3 步：期望 0.125，实际得到 {}",
        val
    );
}

// ==================== 错误处理测试 ====================

#[test]
fn test_duplicate_recurrent_connection() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let prev = graph.new_input_node(&[1, 1], Some("prev")).unwrap();
    let output = graph.new_add_node(&[input, prev], Some("output")).unwrap();
    let other = graph.new_input_node(&[1, 1], Some("other")).unwrap();

    // 第一次连接成功
    graph.connect_recurrent(output, prev).unwrap();

    // 第二次连接同一个 to_node 应该失败
    assert_err!(
        graph.connect_recurrent(other, prev),
        GraphError::InvalidOperation(msg) if msg.contains("已经有循环连接源")
    );
}

#[test]
fn test_recurrent_with_invalid_nodes() {
    let mut graph = GraphInner::new();
    let input = graph.new_input_node(&[1, 1], Some("input")).unwrap();
    let invalid_id = NodeId(999);

    // 不存在的 from_node
    let result = graph.connect_recurrent(invalid_id, input);
    assert!(matches!(result, Err(GraphError::NodeNotFound(_))));

    // 不存在的 to_node
    let result = graph.connect_recurrent(input, invalid_id);
    assert!(matches!(result, Err(GraphError::NodeNotFound(_))));
}

// ==================== 多循环连接测试 ====================

#[test]
fn test_multiple_recurrent_connections() {
    let mut graph = GraphInner::new();

    // 两个独立的循环
    let input1 = graph.new_input_node(&[1, 1], Some("input1")).unwrap();
    let prev1 = graph.new_input_node(&[1, 1], Some("prev1")).unwrap();
    graph
        .set_node_value(prev1, Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    let out1 = graph.new_add_node(&[input1, prev1], Some("out1")).unwrap();

    let input2 = graph.new_input_node(&[1, 1], Some("input2")).unwrap();
    let prev2 = graph.new_input_node(&[1, 1], Some("prev2")).unwrap();
    graph
        .set_node_value(prev2, Some(&Tensor::zeros(&[1, 1])))
        .unwrap();
    let out2 = graph.new_add_node(&[input2, prev2], Some("out2")).unwrap();

    graph.connect_recurrent(out1, prev1).unwrap();
    graph.connect_recurrent(out2, prev2).unwrap();

    // 最终输出：out1 + out2
    let final_out = graph.new_add_node(&[out1, out2], Some("final")).unwrap();

    // 验证两个独立累加器
    for i in 1..=3 {
        graph.set_node_value(input1, Some(&scalar(1.0))).unwrap();
        graph.set_node_value(input2, Some(&scalar(2.0))).unwrap();
        graph.step(final_out).unwrap();

        let v1 = graph.get_node_value(out1).unwrap().unwrap().data_as_slice()[0];
        let v2 = graph.get_node_value(out2).unwrap().unwrap().data_as_slice()[0];
        let vf = graph
            .get_node_value(final_out)
            .unwrap()
            .unwrap()
            .data_as_slice()[0];

        assert!(
            (v1 - i as f32).abs() < 1e-6,
            "out1 第 {} 步：期望 {}，实际得到 {}",
            i,
            i,
            v1
        );
        assert!(
            (v2 - (i * 2) as f32).abs() < 1e-6,
            "out2 第 {} 步：期望 {}，实际得到 {}",
            i,
            i * 2,
            v2
        );
        assert!(
            (vf - (i * 3) as f32).abs() < 1e-6,
            "final 第 {} 步：期望 {}，实际得到 {}",
            i,
            i * 3,
            vf
        );
    }
}

// ==================== 向量测试（非标量） ====================

#[test]
fn test_vector_accumulator() {
    let mut graph = GraphInner::new();

    // 3 维向量累加
    let input = graph.new_input_node(&[3, 1], Some("input")).unwrap();
    let prev = graph.new_input_node(&[3, 1], Some("prev")).unwrap();
    graph
        .set_node_value(prev, Some(&Tensor::zeros(&[3, 1])))
        .unwrap();
    let output = graph.new_add_node(&[input, prev], Some("output")).unwrap();
    graph.connect_recurrent(output, prev).unwrap();

    let input_vec = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);

    // Step 0: [1,2,3]
    graph.set_node_value(input, Some(&input_vec)).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(val.data_as_slice(), &[1.0, 2.0, 3.0]);

    // Step 1: [2,4,6]
    graph.set_node_value(input, Some(&input_vec)).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(val.data_as_slice(), &[2.0, 4.0, 6.0]);

    // Step 2: [3,6,9]
    graph.set_node_value(input, Some(&input_vec)).unwrap();
    graph.step(output).unwrap();
    let val = graph.get_node_value(output).unwrap().unwrap();
    assert_eq!(val.data_as_slice(), &[3.0, 6.0, 9.0]);
}
