/*
 * @Author       : 老董
 * @Description  : Stack 节点单元测试
 *
 * 测试策略：
 * 1. 基础功能测试（创建、形状验证、命名）
 * 2. 前向传播测试（concat 和 stack 模式）
 * 3. VJP 单元测试（直接调用 calc_grad_to_parent）
 * 4. 端到端反向传播测试（通过 graph.backward）
 */

use crate::assert_err;
use crate::nn::{GraphError, GraphInner};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 基础功能测试 ====================

/// 测试 Stack 节点创建（concat 模式，new_dim=false）
#[test]
fn test_stack_creation_concat_mode() {
    let mut graph = GraphInner::new();

    // 1. 两个形状兼容的节点拼接（axis=0）
    {
        let input1 = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
        let input2 = graph.new_basic_input_node(&[3, 3], Some("input2")).unwrap();
        let stack = graph
            .new_stack_node(&[input1, input2], 0, false, Some("concat_0"))
            .unwrap();

        assert_eq!(graph.get_node_name(stack).unwrap(), "concat_0");
        assert_eq!(graph.get_node_parents(stack).unwrap().len(), 2);
        // [2, 3] + [3, 3] 沿 axis=0 -> [5, 3]
        assert_eq!(graph.get_node_value_expected_shape(stack).unwrap(), &[5, 3]);
    }

    // 2. 沿 axis=1 拼接
    {
        let input1 = graph.new_basic_input_node(&[2, 3], Some("input3")).unwrap();
        let input2 = graph.new_basic_input_node(&[2, 4], Some("input4")).unwrap();
        let stack = graph
            .new_stack_node(&[input1, input2], 1, false, Some("concat_1"))
            .unwrap();

        // [2, 3] + [2, 4] 沿 axis=1 -> [2, 7]
        assert_eq!(graph.get_node_value_expected_shape(stack).unwrap(), &[2, 7]);
    }

    // 3. 三个节点拼接
    {
        let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
        let p2 = graph.new_parameter_node(&[3, 2], Some("p2")).unwrap();
        let p3 = graph.new_parameter_node(&[1, 2], Some("p3")).unwrap();
        let stack = graph
            .new_stack_node(&[p1, p2, p3], 0, false, Some("concat_three"))
            .unwrap();

        assert_eq!(graph.get_node_parents(stack).unwrap().len(), 3);
        // [2, 2] + [3, 2] + [1, 2] 沿 axis=0 -> [6, 2]
        assert_eq!(graph.get_node_value_expected_shape(stack).unwrap(), &[6, 2]);
    }
}

/// 测试 Stack 节点创建（stack 模式，new_dim=true）
#[test]
fn test_stack_creation_stack_mode() {
    let mut graph = GraphInner::new();

    // 1. 两个相同形状节点堆叠（axis=0）
    {
        let input1 = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
        let input2 = graph.new_basic_input_node(&[2, 3], Some("input2")).unwrap();
        let stack = graph
            .new_stack_node(&[input1, input2], 0, true, Some("stack_0"))
            .unwrap();

        assert_eq!(graph.get_node_name(stack).unwrap(), "stack_0");
        // [2, 3] 堆叠 2 个 -> [2, 2, 3]（在 axis=0 插入新维度）
        assert_eq!(
            graph.get_node_value_expected_shape(stack).unwrap(),
            &[2, 2, 3]
        );
    }

    // 2. 沿 axis=1 堆叠
    {
        let input1 = graph.new_basic_input_node(&[2, 3], Some("input3")).unwrap();
        let input2 = graph.new_basic_input_node(&[2, 3], Some("input4")).unwrap();
        let stack = graph
            .new_stack_node(&[input1, input2], 1, true, Some("stack_1"))
            .unwrap();

        // [2, 3] 堆叠 2 个，axis=1 -> [2, 2, 3]
        assert_eq!(
            graph.get_node_value_expected_shape(stack).unwrap(),
            &[2, 2, 3]
        );
    }

    // 3. 沿最后一个维度后堆叠（axis=ndim）
    {
        let input1 = graph.new_basic_input_node(&[2, 3], Some("input5")).unwrap();
        let input2 = graph.new_basic_input_node(&[2, 3], Some("input6")).unwrap();
        let stack = graph
            .new_stack_node(&[input1, input2], 2, true, Some("stack_last"))
            .unwrap();

        // [2, 3] 堆叠 2 个，axis=2 -> [2, 3, 2]
        assert_eq!(
            graph.get_node_value_expected_shape(stack).unwrap(),
            &[2, 3, 2]
        );
    }

    // 4. 三个节点堆叠
    {
        let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
        let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
        let p3 = graph.new_parameter_node(&[2, 2], Some("p3")).unwrap();
        let stack = graph
            .new_stack_node(&[p1, p2, p3], 0, true, Some("stack_three"))
            .unwrap();

        // 3 个 [2, 2] 堆叠 -> [3, 2, 2]
        assert_eq!(
            graph.get_node_value_expected_shape(stack).unwrap(),
            &[3, 2, 2]
        );
    }
}

/// 测试 Stack 创建时的形状校验（concat 模式）
#[test]
fn test_stack_creation_invalid_shape_concat() {
    let mut graph = GraphInner::new();

    // 除 axis 外其他维度不一致
    let input1 = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_basic_input_node(&[2, 4], Some("input2")).unwrap();

    // axis=0 拼接时，axis=1 维度必须相同 (3 != 4)
    let result = graph.new_stack_node(&[input1, input2], 0, false, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch(
            [2, 3],
            [2, 4],
            "Stack (new_dim=false): 父节点 1 在维度 1 大小不一致"
        )
    );
}

/// 测试 Stack 创建时的形状校验（stack 模式）
#[test]
fn test_stack_creation_invalid_shape_stack() {
    let mut graph = GraphInner::new();

    // stack 模式要求所有形状完全相同
    let input1 = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_basic_input_node(&[2, 4], Some("input2")).unwrap();

    let result = graph.new_stack_node(&[input1, input2], 0, true, None);
    assert_err!(
        result,
        GraphError::ShapeMismatch([2, 3], [2, 4], "Stack (new_dim=true): 父节点 1 形状不一致")
    );
}

/// 测试 Stack 创建时 axis 越界
#[test]
fn test_stack_creation_invalid_axis() {
    let mut graph = GraphInner::new();

    let input1 = graph.new_basic_input_node(&[2, 3], Some("input1")).unwrap();
    let input2 = graph.new_basic_input_node(&[2, 3], Some("input2")).unwrap();

    // concat 模式：axis 最大为 ndim-1 = 1
    let result = graph.new_stack_node(&[input1, input2], 2, false, None);
    assert_err!(
        result,
        GraphError::InvalidOperation("Stack: axis 2 超出有效范围 [0, 1]")
    );

    // stack 模式：axis 最大为 ndim = 2
    let result = graph.new_stack_node(&[input1, input2], 3, true, None);
    assert_err!(
        result,
        GraphError::InvalidOperation("Stack: axis 3 超出有效范围 [0, 2]")
    );
}

/// 测试 Stack 节点命名
#[test]
fn test_stack_name_generation() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 3], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2")).unwrap();

    // 1. 显式命名
    let stack1 = graph
        .new_stack_node(&[p1, p2], 0, true, Some("my_stack"))
        .unwrap();
    assert_eq!(graph.get_node_name(stack1).unwrap(), "my_stack");

    // 2. 自动命名
    let stack2 = graph.new_stack_node(&[p1, p2], 0, true, None).unwrap();
    assert_eq!(graph.get_node_name(stack2).unwrap(), "stack_1");

    // 3. 名称重复
    let result = graph.new_stack_node(&[p1, p2], 0, true, Some("my_stack"));
    assert_err!(
        result,
        GraphError::DuplicateNodeName("节点my_stack在图default_graph中重复")
    );
}

// ==================== 前向传播测试 ====================

/// 测试 Stack 前向传播（concat 模式，axis=0）
#[test]
fn test_stack_forward_concat_axis0() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[1, 2], Some("p2")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2], 0, false, Some("stack"))
        .unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6]]
    // concat -> [[1,2],[3,4],[5,6]]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_eq!(output, &expected);
}

/// 测试 Stack 前向传播（concat 模式，axis=1）
#[test]
fn test_stack_forward_concat_axis1() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2], 1, false, Some("stack"))
        .unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6,7],[8,9,10]]
    // concat axis=1 -> [[1,2,5,6,7],[3,4,8,9,10]]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(
            p2,
            Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3])),
        )
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    let expected = Tensor::new(
        &[1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );
    assert_eq!(output, &expected);
}

/// 测试 Stack 前向传播（stack 模式，axis=0）
#[test]
fn test_stack_forward_stack_axis0() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2], 0, true, Some("stack"))
        .unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]]
    // stack axis=0 -> [[[1,2],[3,4]], [[5,6],[7,8]]]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(output, &expected);
}

/// 测试 Stack 前向传播（stack 模式，axis=1）
#[test]
fn test_stack_forward_stack_axis1() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 3], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2], 1, true, Some("stack"))
        .unwrap();

    // p1=[[1,2,3],[4,5,6]], p2=[[7,8,9],[10,11,12]]
    // stack axis=1 -> 在 axis=1 插入新维度
    // 结果形状: [2, 2, 3]
    graph
        .set_node_value(
            p1,
            Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
        )
        .unwrap();
    graph
        .set_node_value(
            p2,
            Some(&Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3])),
        )
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 3]);
    // 期望: [[[1,2,3],[7,8,9]], [[4,5,6],[10,11,12]]]
    let expected = Tensor::new(
        &[
            1.0, 2.0, 3.0, 7.0, 8.0, 9.0, 4.0, 5.0, 6.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    );
    assert_eq!(output, &expected);
}

/// 测试 Stack 前向传播（stack 模式，axis=末尾）
#[test]
fn test_stack_forward_stack_axis_last() {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1")).unwrap();
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2], 2, true, Some("stack"))
        .unwrap();

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]]
    // stack axis=2 (末尾) -> 在最后插入新维度
    // 结果形状: [2, 2, 2]
    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 2, 2]);
    // 期望: [[[1,5],[2,6]], [[3,7],[4,8]]]
    let expected = Tensor::new(&[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0], &[2, 2, 2]);
    assert_eq!(output, &expected);
}

/// 测试 Stack 前向传播（三个父节点）
#[test]
fn test_stack_forward_three_parents() {
    let mut graph = GraphInner::new();

    // 使用 2D 形状避免维度限制
    let p1 = graph.new_basic_input_node(&[1, 2], Some("p1")).unwrap();
    let p2 = graph.new_basic_input_node(&[1, 2], Some("p2")).unwrap();
    let p3 = graph.new_basic_input_node(&[1, 2], Some("p3")).unwrap();
    let stack = graph
        .new_stack_node(&[p1, p2, p3], 0, true, Some("stack"))
        .unwrap();

    graph
        .set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))
        .unwrap();
    graph
        .set_node_value(p2, Some(&Tensor::new(&[3.0, 4.0], &[1, 2])))
        .unwrap();
    graph
        .set_node_value(p3, Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))
        .unwrap();

    graph.forward(stack).unwrap();

    let output = graph.get_node_value(stack).unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 1, 2]);
    let expected = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 1, 2]);
    assert_eq!(output, &expected);
}

// ==================== 节点级反向传播测试 ====================

/// 测试 Stack 对第一个父节点的梯度计算（stack 模式）
#[test]
fn test_stack_backward_to_first_parent_stack_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let stack = graph.new_stack_node(&[p1, p2], 0, true, Some("stack"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(stack)?;

    // 直接测试 VJP
    // upstream_grad shape: [2, 2, 2]
    let upstream_grad = Tensor::ones(&[2, 2, 2]);
    let stack_node = graph.get_node(stack)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 新签名：使用 parents 数组和索引
    let parents = [p1_node, p2_node];

    // stack 模式下，每个父节点的梯度是 upstream_grad 在 axis 维度的对应切片
    let grad = stack_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;

    // p1 对应 upstream_grad[0, :, :] = [[1,1],[1,1]]
    assert_eq!(grad.shape(), &[2, 2]);
    assert_eq!(&grad, &Tensor::ones(&[2, 2]));

    Ok(())
}

/// 测试 Stack 对第二个父节点的梯度计算（stack 模式）
#[test]
fn test_stack_backward_to_second_parent_stack_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let stack = graph.new_stack_node(&[p1, p2], 0, true, Some("stack"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.forward(stack)?;

    // upstream_grad = [[[1,2],[3,4]], [[5,6],[7,8]]]
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let stack_node = graph.get_node(stack)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 新签名：使用 parents 数组和索引
    let parents = [p1_node, p2_node];

    // p2 对应 upstream_grad[1, :, :] = [[5,6],[7,8]]
    let grad = stack_node.calc_grad_to_parent(1, &parents, &upstream_grad)?;

    assert_eq!(grad.shape(), &[2, 2]);
    let expected = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    assert_eq!(&grad, &expected);

    Ok(())
}

/// 测试 Stack 梯度计算（stack 模式，axis=1）
#[test]
fn test_stack_backward_stack_mode_axis1() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 3], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2"))?;
    let stack = graph.new_stack_node(&[p1, p2], 1, true, Some("stack"))?;

    // 设置值并前向传播
    graph.set_node_value(
        p1,
        Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])),
    )?;
    graph.set_node_value(
        p2,
        Some(&Tensor::new(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3])),
    )?;
    graph.forward(stack)?;

    // upstream_grad shape: [2, 2, 3]
    // 使用递增值便于验证切片正确性
    let upstream_grad = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // 第一个 [2, 3] 块
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // 第二个 [2, 3] 块
        ],
        &[2, 2, 3],
    );
    let stack_node = graph.get_node(stack)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 新签名：使用 parents 数组和索引
    let parents = [p1_node, p2_node];

    // p1 对应 upstream_grad[:, 0, :] = [[1,2,3],[7,8,9]]
    let grad_p1 = stack_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;
    assert_eq!(grad_p1.shape(), &[2, 3]);
    let expected_p1 = Tensor::new(&[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], &[2, 3]);
    assert_eq!(&grad_p1, &expected_p1);

    // p2 对应 upstream_grad[:, 1, :] = [[4,5,6],[10,11,12]]
    let grad_p2 = stack_node.calc_grad_to_parent(1, &parents, &upstream_grad)?;
    assert_eq!(grad_p2.shape(), &[2, 3]);
    let expected_p2 = Tensor::new(&[4.0, 5.0, 6.0, 10.0, 11.0, 12.0], &[2, 3]);
    assert_eq!(&grad_p2, &expected_p2);

    Ok(())
}

/// 测试 Stack 梯度计算（concat 模式，axis=1）
#[test]
fn test_stack_backward_concat_mode_axis1() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 3], Some("p2"))?;
    let stack = graph.new_stack_node(&[p1, p2], 1, false, Some("stack"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(
        p2,
        Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3])),
    )?;
    graph.forward(stack)?;

    // 输出形状: [2, 5]
    // upstream_grad 使用递增值
    let upstream_grad = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
    );
    let stack_node = graph.get_node(stack)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 新签名：使用 parents 数组和索引
    let parents = [p1_node, p2_node];

    // p1 对应 upstream_grad[:, 0:2] = [[1,2],[6,7]]
    let grad_p1 = stack_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;
    assert_eq!(grad_p1.shape(), &[2, 2]);
    let expected_p1 = Tensor::new(&[1.0, 2.0, 6.0, 7.0], &[2, 2]);
    assert_eq!(&grad_p1, &expected_p1);

    // p2 对应 upstream_grad[:, 2:5] = [[3,4,5],[8,9,10]]
    let grad_p2 = stack_node.calc_grad_to_parent(1, &parents, &upstream_grad)?;
    assert_eq!(grad_p2.shape(), &[2, 3]);
    let expected_p2 = Tensor::new(&[3.0, 4.0, 5.0, 8.0, 9.0, 10.0], &[2, 3]);
    assert_eq!(&grad_p2, &expected_p2);

    Ok(())
}

/// 测试 Stack 梯度计算（concat 模式，axis=0）
#[test]
fn test_stack_backward_concat_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[1, 2], Some("p2"))?;
    let stack = graph.new_stack_node(&[p1, p2], 0, false, Some("stack"))?;

    // 设置值并前向传播
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))?;
    graph.forward(stack)?;

    // upstream_grad shape: [3, 2]
    let upstream_grad = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let stack_node = graph.get_node(stack)?;
    let p1_node = graph.get_node(p1)?;
    let p2_node = graph.get_node(p2)?;

    // 新签名：使用 parents 数组和索引
    let parents = [p1_node, p2_node];

    // p1 对应 upstream_grad[0:2, :] = [[1,2],[3,4]]
    let grad_p1 = stack_node.calc_grad_to_parent(0, &parents, &upstream_grad)?;
    assert_eq!(grad_p1.shape(), &[2, 2]);
    let expected_p1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(&grad_p1, &expected_p1);

    // p2 对应 upstream_grad[2:3, :] = [[5,6]]
    let grad_p2 = stack_node.calc_grad_to_parent(1, &parents, &upstream_grad)?;
    assert_eq!(grad_p2.shape(), &[1, 2]);
    let expected_p2 = Tensor::new(&[5.0, 6.0], &[1, 2]);
    assert_eq!(&grad_p2, &expected_p2);

    Ok(())
}

// ==================== 端到端反向传播测试 ====================

/// 测试 Stack 通过 graph.backward() 的端到端反向传播（concat 模式，axis=0，相同形状父节点）
#[test]
fn test_stack_backward_e2e_concat_same_shape() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = concat([p1, p2], axis=0, new_dim=false)
    let p1 = graph.new_parameter_node(&[1, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[1, 2], Some("p2"))?;
    let result = graph.new_stack_node(&[p1, p2], 0, false, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：p1=[[1,2]], p2=[[3,4]], target=[[0,0],[0,0]]
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0], &[1, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[3.0, 4.0], &[1, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[1,2],[3,4]]
    // loss = mean((result - 0)^2) = mean([1,4,9,16]) = 30/4 = 7.5
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 7.5, epsilon = 1e-6);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = 2*(result - target)/n = result/2 = [[0.5,1],[1.5,2]]
    // ∂loss/∂p1 = ∂loss/∂result[0,:] = [[0.5, 1]]
    // ∂loss/∂p2 = ∂loss/∂result[1,:] = [[1.5, 2]]
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");

    assert_eq!(p1_grad.shape(), &[1, 2]);
    assert_eq!(p2_grad.shape(), &[1, 2]);

    let expected_p1_grad = Tensor::new(&[0.5, 1.0], &[1, 2]);
    let expected_p2_grad = Tensor::new(&[1.5, 2.0], &[1, 2]);
    assert_abs_diff_eq!(p1_grad, &expected_p1_grad, epsilon = 1e-6);
    assert_abs_diff_eq!(p2_grad, &expected_p2_grad, epsilon = 1e-6);

    Ok(())
}

/// 测试 Stack 通过 graph.backward() 的端到端反向传播（真正的 stack 模式，new_dim=true）
#[test]
fn test_stack_backward_e2e_stack_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = stack([p1, p2], axis=0, new_dim=true)
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let result = graph.new_stack_node(&[p1, p2], 0, true, Some("result"))?;

    // loss = MSE(result, target)
    // result 形状: [2, 2, 2]
    let target = graph.new_basic_input_node(&[2, 2, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：p1=[[1,2],[3,4]], p2=[[5,6],[7,8]], target=zeros
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 2, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[[1,2],[3,4]], [[5,6],[7,8]]]
    // loss = mean([1,4,9,16,25,36,49,64]) = 204/8 = 25.5
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 25.5, epsilon = 1e-6);

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = 2*(result - target)/n = result/4
    // ∂loss/∂p1 = ∂loss/∂result[0,:,:] = [[0.25,0.5],[0.75,1.0]]
    // ∂loss/∂p2 = ∂loss/∂result[1,:,:] = [[1.25,1.5],[1.75,2.0]]
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");

    assert_eq!(p1_grad.shape(), &[2, 2]);
    assert_eq!(p2_grad.shape(), &[2, 2]);

    let expected_p1_grad = Tensor::new(&[0.25, 0.5, 0.75, 1.0], &[2, 2]);
    let expected_p2_grad = Tensor::new(&[1.25, 1.5, 1.75, 2.0], &[2, 2]);
    assert_abs_diff_eq!(p1_grad, &expected_p1_grad, epsilon = 1e-6);
    assert_abs_diff_eq!(p2_grad, &expected_p2_grad, epsilon = 1e-6);

    Ok(())
}

/// 测试 Stack 通过 graph.backward() 的端到端反向传播（concat 模式，不同形状父节点）
#[test]
fn test_stack_backward_e2e_concat_mode() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 创建计算图：result = concat([p1, p2], axis=0)
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[1, 2], Some("p2"))?;
    let result = graph.new_stack_node(&[p1, p2], 0, false, Some("result"))?;

    // loss = MSE(result, target)
    let target = graph.new_basic_input_node(&[3, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // 设置值：p1=[[1,2],[3,4]], p2=[[5,6]], target=[[0,0],[0,0],[0,0]]
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0], &[1, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[3, 2])))?;

    // 前向传播
    graph.forward(loss)?;

    // result = [[1,2],[3,4],[5,6]]
    // loss = mean((result - 0)^2) = mean([1,4,9,16,25,36]) = 91/6 ≈ 15.167
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(
        loss_value.get_data_number().unwrap(),
        91.0 / 6.0,
        epsilon = 1e-4
    );

    // 反向传播
    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = 2*(result - target)/n = result/3
    //               = [[1/3, 2/3], [3/3, 4/3], [5/3, 6/3]]
    // ∂loss/∂p1 = ∂loss/∂result[0:2,:] = [[1/3, 2/3], [1, 4/3]]
    // ∂loss/∂p2 = ∂loss/∂result[2:3,:] = [[5/3, 2]]
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");

    assert_eq!(p1_grad.shape(), &[2, 2]);
    assert_eq!(p2_grad.shape(), &[1, 2]);

    let expected_p1_grad = Tensor::new(&[1.0 / 3.0, 2.0 / 3.0, 1.0, 4.0 / 3.0], &[2, 2]);
    let expected_p2_grad = Tensor::new(&[5.0 / 3.0, 2.0], &[1, 2]);
    assert_abs_diff_eq!(p1_grad, &expected_p1_grad, epsilon = 1e-4);
    assert_abs_diff_eq!(p2_grad, &expected_p2_grad, epsilon = 1e-4);

    Ok(())
}

/// 测试 Stack 端到端（三个父节点，concat 模式）
#[test]
fn test_stack_backward_e2e_three_parents() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 使用 2D Parameter 和 concat 模式避免维度问题
    let p1 = graph.new_parameter_node(&[1, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[1, 2], Some("p2"))?;
    let p3 = graph.new_parameter_node(&[1, 2], Some("p3"))?;
    let result = graph.new_stack_node(&[p1, p2, p3], 0, false, Some("result"))?;

    let target = graph.new_basic_input_node(&[3, 2], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // p1=[[1,1]], p2=[[2,2]], p3=[[3,3]], target=zeros
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0; 2], &[1, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[2.0; 2], &[1, 2])))?;
    graph.set_node_value(p3, Some(&Tensor::new(&[3.0; 2], &[1, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[3, 2])))?;

    graph.forward(loss)?;

    // result = [[1,1],[2,2],[3,3]]
    // loss = mean([1,1,4,4,9,9]) = 28/6 ≈ 4.667
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(
        loss_value.get_data_number().unwrap(),
        28.0 / 6.0,
        epsilon = 1e-4
    );

    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = result/3 = [[1/3,1/3],[2/3,2/3],[1,1]]
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");
    let p3_grad = graph.get_node(p3)?.grad().expect("p3 应有 grad");

    let expected_p1_grad = Tensor::new(&[1.0 / 3.0; 2], &[1, 2]);
    let expected_p2_grad = Tensor::new(&[2.0 / 3.0; 2], &[1, 2]);
    let expected_p3_grad = Tensor::new(&[1.0; 2], &[1, 2]);

    assert_abs_diff_eq!(p1_grad, &expected_p1_grad, epsilon = 1e-4);
    assert_abs_diff_eq!(p2_grad, &expected_p2_grad, epsilon = 1e-4);
    assert_abs_diff_eq!(p3_grad, &expected_p3_grad, epsilon = 1e-4);

    Ok(())
}

/// 测试 Stack 端到端（concat 模式，axis=1）
///
/// 使用相同形状的父节点以避免动态形状兼容性问题
#[test]
fn test_stack_backward_e2e_concat_axis1() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    // 使用相同形状的参数避免动态形状问题
    let p1 = graph.new_parameter_node(&[2, 2], Some("p1"))?;
    let p2 = graph.new_parameter_node(&[2, 2], Some("p2"))?;
    let result = graph.new_stack_node(&[p1, p2], 1, false, Some("result"))?;

    let target = graph.new_basic_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_mse_loss_node(result, target, Some("loss"))?;

    // p1=[[1,2],[3,4]], p2=[[5,6],[7,8]], target=zeros
    graph.set_node_value(p1, Some(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2])))?;
    graph.set_node_value(p2, Some(&Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 4])))?;

    graph.forward(loss)?;

    // result = [[1,2,5,6],[3,4,7,8]]
    // loss = mean([1,4,25,36,9,16,49,64]) = 204/8 = 25.5
    let loss_value = graph.get_node_value(loss)?.unwrap();
    assert_abs_diff_eq!(loss_value.get_data_number().unwrap(), 25.5, epsilon = 1e-4);

    graph.zero_grad()?;
    graph.backward(loss)?;

    // ∂loss/∂result = 2*result/n = result/4
    // ∂loss/∂p1 = [[0.25, 0.5], [0.75, 1.0]]
    // ∂loss/∂p2 = [[1.25, 1.5], [1.75, 2.0]]
    let p1_grad = graph.get_node(p1)?.grad().expect("p1 应有 grad");
    let p2_grad = graph.get_node(p2)?.grad().expect("p2 应有 grad");

    let expected_p1_grad = Tensor::new(&[0.25, 0.5, 0.75, 1.0], &[2, 2]);
    let expected_p2_grad = Tensor::new(&[1.25, 1.5, 1.75, 2.0], &[2, 2]);
    assert_abs_diff_eq!(p1_grad, &expected_p1_grad, epsilon = 1e-4);
    assert_abs_diff_eq!(p2_grad, &expected_p2_grad, epsilon = 1e-4);

    Ok(())
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
use std::rc::Rc;

#[test]
fn test_create_stack_node_concat_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // concat 模式：[2, 3] + [1, 3] -> [3, 3]
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("p1"))
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("p2"))
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1.clone(), p2.clone()], 0, false, Some("stack"))
        .unwrap();

    assert_eq!(stack.shape(), vec![3, 3]);
    assert_eq!(stack.name(), Some("stack"));
    assert!(!stack.is_leaf());
    assert_eq!(stack.parents().len(), 2);
}

#[test]
fn test_create_stack_node_stack_axis0() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stack 模式：在 axis=0 插入新维度
    // [2, 3] + [2, 3] -> [2, 2, 3]
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2], 0, true, None)
        .unwrap();

    assert_eq!(stack.shape(), vec![2, 2, 3]);
}

#[test]
fn test_create_stack_node_three_parents() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();
    let p3 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 2], None)
        .unwrap();

    let stack = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2, p3], 0, true, None)
        .unwrap();

    // [2, 2] * 3 -> [3, 2, 2]
    assert_eq!(stack.shape(), vec![3, 2, 2]);
}

#[test]
fn test_create_stack_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    // stack 模式要求形状完全相同
    let p1 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let p2 = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不同
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_stack_node(vec![p1, p2], 0, true, None);
    assert!(result.is_err());
}

#[test]
fn test_create_stack_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_stack;
    let weak_p1;
    let weak_p2;
    {
        let p1 = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        let p2 = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_p1 = Rc::downgrade(&p1);
        weak_p2 = Rc::downgrade(&p2);

        let stack = inner
            .borrow_mut()
            .create_stack_node(vec![p1, p2], 0, true, None)
            .unwrap();
        weak_stack = Rc::downgrade(&stack);

        assert!(weak_stack.upgrade().is_some());
        assert!(weak_p1.upgrade().is_some());
        assert!(weak_p2.upgrade().is_some());
    }
    assert!(weak_stack.upgrade().is_none());
    assert!(weak_p1.upgrade().is_none());
    assert!(weak_p2.upgrade().is_none());
}
