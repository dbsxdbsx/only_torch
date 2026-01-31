//! gather 相关测试

use crate::tensor::Tensor;

// ============================================================================
// 基本功能测试
// ============================================================================

#[test]
fn test_gather_2d_dim1_basic() {
    // SAC/DQN 核心场景：按动作索引选择 Q 值
    // Q 值：[[1.0, 2.0, 3.0],
    //        [4.0, 5.0, 6.0]]
    let q_values = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // 动作索引：[[1], [2]]
    let actions = Tensor::new(&[1.0, 2.0], &[2, 1]);

    let selected = q_values.gather(1, &actions);

    assert_eq!(selected.shape(), &[2, 1]);
    assert_eq!(selected[[0, 0]], 2.0); // q_values[0, 1]
    assert_eq!(selected[[1, 0]], 6.0); // q_values[1, 2]
}

#[test]
fn test_gather_2d_dim0_basic() {
    // dim=0 场景：沿行方向 gather
    // input: [[1.0, 2.0],
    //         [3.0, 4.0],
    //         [5.0, 6.0]]
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    // index: [[0, 2],
    //         [1, 0]]
    let index = Tensor::new(&[0.0, 2.0, 1.0, 0.0], &[2, 2]);

    let result = input.gather(0, &index);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // input[0, 0]
    assert_eq!(result[[0, 1]], 6.0); // input[2, 1]
    assert_eq!(result[[1, 0]], 3.0); // input[1, 0]
    assert_eq!(result[[1, 1]], 2.0); // input[0, 1]
}

#[test]
fn test_gather_2d_multiple_indices_per_row() {
    // 每行选择多个元素
    // Q 值：[[1.0, 2.0, 3.0, 4.0],
    //        [5.0, 6.0, 7.0, 8.0]]
    let q_values = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]);

    // 每行选 2 个
    // index: [[0, 3],
    //         [1, 2]]
    let index = Tensor::new(&[0.0, 3.0, 1.0, 2.0], &[2, 2]);

    let result = q_values.gather(1, &index);

    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 1.0); // q_values[0, 0]
    assert_eq!(result[[0, 1]], 4.0); // q_values[0, 3]
    assert_eq!(result[[1, 0]], 6.0); // q_values[1, 1]
    assert_eq!(result[[1, 1]], 7.0); // q_values[1, 2]
}

// ============================================================================
// 1D 张量测试
// ============================================================================

#[test]
fn test_gather_1d() {
    // 1D 张量 gather
    let input = Tensor::new(&[10.0, 20.0, 30.0, 40.0, 50.0], &[5]);
    let index = Tensor::new(&[4.0, 0.0, 2.0], &[3]);

    let result = input.gather(0, &index);

    assert_eq!(result.shape(), &[3]);
    assert_eq!(result[[0]], 50.0); // input[4]
    assert_eq!(result[[1]], 10.0); // input[0]
    assert_eq!(result[[2]], 30.0); // input[2]
}

// ============================================================================
// 3D 张量测试
// ============================================================================

#[test]
fn test_gather_3d_dim2() {
    // 3D 张量 gather（模拟 batch + seq + features）
    // input: [2, 2, 3]
    #[rustfmt::skip]
    let input = Tensor::new(&[
        // batch 0
        1.0, 2.0, 3.0,   // seq 0
        4.0, 5.0, 6.0,   // seq 1
        // batch 1
        7.0, 8.0, 9.0,   // seq 0
        10.0, 11.0, 12.0 // seq 1
    ], &[2, 2, 3]);

    // index: [2, 2, 1]
    let index = Tensor::new(&[0.0, 2.0, 1.0, 0.0], &[2, 2, 1]);

    let result = input.gather(2, &index);

    assert_eq!(result.shape(), &[2, 2, 1]);
    assert_eq!(result[[0, 0, 0]], 1.0);  // input[0, 0, 0]
    assert_eq!(result[[0, 1, 0]], 6.0);  // input[0, 1, 2]
    assert_eq!(result[[1, 0, 0]], 8.0);  // input[1, 0, 1]
    assert_eq!(result[[1, 1, 0]], 10.0); // input[1, 1, 0]
}

// ============================================================================
// 边界情况测试
// ============================================================================

#[test]
fn test_gather_same_index_repeated() {
    // 同一个索引被多次选择
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let index = Tensor::new(&[1.0, 1.0, 1.0], &[1, 3]);

    let result = input.gather(1, &index);

    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result[[0, 0]], 2.0);
    assert_eq!(result[[0, 1]], 2.0);
    assert_eq!(result[[0, 2]], 2.0);
}

#[test]
fn test_gather_single_element() {
    // 单元素张量
    let input = Tensor::new(&[42.0], &[1, 1]);
    let index = Tensor::new(&[0.0], &[1, 1]);

    let result = input.gather(1, &index);

    assert_eq!(result.shape(), &[1, 1]);
    assert_eq!(result[[0, 0]], 42.0);
}

#[test]
fn test_gather_all_indices() {
    // 选择所有元素（相当于复制）
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let index = Tensor::new(&[0.0, 1.0, 2.0], &[1, 3]);

    let result = input.gather(1, &index);

    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[0, 1]], 2.0);
    assert_eq!(result[[0, 2]], 3.0);
}

#[test]
fn test_gather_reverse_order() {
    // 逆序选择
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let index = Tensor::new(&[3.0, 2.0, 1.0, 0.0], &[1, 4]);

    let result = input.gather(1, &index);

    assert_eq!(result.shape(), &[1, 4]);
    assert_eq!(result[[0, 0]], 4.0);
    assert_eq!(result[[0, 1]], 3.0);
    assert_eq!(result[[0, 2]], 2.0);
    assert_eq!(result[[0, 3]], 1.0);
}

// ============================================================================
// 强化学习实际场景测试
// ============================================================================

#[test]
fn test_gather_rl_batch_q_selection() {
    // 模拟 SAC/DQN 中的完整场景
    // batch_size=4, action_dim=3
    #[rustfmt::skip]
    let q_values = Tensor::new(&[
        1.5, 2.3, 0.8,  // sample 0
        3.1, 1.2, 2.5,  // sample 1
        0.9, 4.1, 1.7,  // sample 2
        2.2, 3.3, 4.4,  // sample 3
    ], &[4, 3]);

    // 每个样本选择的动作
    let actions = Tensor::new(&[1.0, 0.0, 1.0, 2.0], &[4, 1]);

    let selected_q = q_values.gather(1, &actions);

    assert_eq!(selected_q.shape(), &[4, 1]);
    assert_eq!(selected_q[[0, 0]], 2.3); // q[0, 1]
    assert_eq!(selected_q[[1, 0]], 3.1); // q[1, 0]
    assert_eq!(selected_q[[2, 0]], 4.1); // q[2, 1]
    assert_eq!(selected_q[[3, 0]], 4.4); // q[3, 2]
}

#[test]
fn test_gather_with_float_indices() {
    // 验证 f32 索引被正确转换为 usize
    let input = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    // 使用 0.9 和 1.1，应该被截断为 0 和 1
    let index = Tensor::new(&[0.9, 1.1], &[2, 1]);

    let result = input.gather(1, &index);

    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result[[0, 0]], 10.0); // input[0, 0] (0.9 -> 0)
    assert_eq!(result[[1, 0]], 40.0); // input[1, 1] (1.1 -> 1)
}

// ============================================================================
// 错误情况测试
// ============================================================================

#[test]
#[should_panic(expected = "gather: dim 2 超出张量维度 2")]
fn test_gather_dim_out_of_range() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = Tensor::new(&[0.0, 1.0], &[2, 1]);
    let _ = input.gather(2, &index); // dim=2 超出范围
}

#[test]
#[should_panic(expected = "gather: index 维度数 1 必须与输入张量维度数 2 相同")]
fn test_gather_index_dim_mismatch() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = Tensor::new(&[0.0, 1.0], &[2]); // 1D 而非 2D
    let _ = input.gather(1, &index);
}

#[test]
#[should_panic(expected = "gather: 维度 0 上 index 大小 3 与输入张量大小 2 不匹配")]
fn test_gather_shape_mismatch() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]); // batch 大小不匹配
    let _ = input.gather(1, &index);
}

#[test]
#[should_panic(expected = "gather: 索引 3 超出维度 1 的范围 [0, 2)")]
fn test_gather_index_out_of_range() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let index = Tensor::new(&[0.0, 3.0], &[2, 1]); // 3 超出 dim=1 的范围 [0, 2)
    let _ = input.gather(1, &index);
}
