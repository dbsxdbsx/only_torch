use crate::assert_panic;
use crate::tensor::Tensor;
use crate::tensor::property::broadcast_shape;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照/view(_mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_view() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let tensor = Tensor::new(&data, &shape);
    let view = tensor.view();
    // 检查可否正常打印
    println!("{:?}", view);
    // 检查view的索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(view[[0]]);
    // 修改view通过索引的元素是否和原始张量保持一致
    assert_eq!(view[[0, 0]], 1.0);
    assert_eq!(view[[0, 1]], 2.0);
    assert_eq!(view[[1, 0]], 3.0);
    assert_eq!(view[[1, 1]], 4.0);
}

#[test]
fn test_view_mut() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2, 2];
    let mut tensor = Tensor::new(&data, &shape);
    let mut view_mut = tensor.view_mut();
    // 检查可否正常打印
    println!("{:?}", view_mut);
    // 检查view_mut的索引是否正确，若非指向具体的某个元素，则会panic
    assert_panic!(view_mut[[0]]);
    // 修改view_mut中的值，并检查原始张量是否也发生了改变
    view_mut[[0, 0]] = 5.0;
    view_mut[[0, 1]] = 6.0;
    view_mut[[1, 0]] = 7.0;
    view_mut[[1, 1]] = 8.0;
    assert_eq!(tensor.data[[0, 0]], 5.0);
    assert_eq!(tensor.data[[0, 1]], 6.0);
    assert_eq!(tensor.data[[1, 0]], 7.0);
    assert_eq!(tensor.data[[1, 1]], 8.0);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照/view(_mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓shape↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_compare_shapes_with_same_shapes() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    assert!(tensor1.is_same_shape(&tensor2));
}

#[test]
fn test_compare_shapes_with_diff_shapes() {
    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[4]);
    assert!(!tensor1.is_same_shape(&tensor2));

    let tensor1 = Tensor::new(&[1., 2., 3., 4.], &[1, 4]);
    let tensor2 = Tensor::new(&[1., 2., 3., 4.], &[4, 1]);
    assert!(!tensor1.is_same_shape(&tensor2));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑shape↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

#[test]
fn test_dimension() {
    let tensor = Tensor::new(&[1.], &[]);
    assert_eq!(tensor.dimension(), 0);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    assert_eq!(tensor.dimension(), 1);

    let tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert_eq!(tensor.dimension(), 2);

    let tensor = Tensor::new(&[1.], &[1, 1, 1]);
    assert_eq!(tensor.dimension(), 3);
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓判断张量是否为标量、向量、矩阵↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_is_scalar() {
    let scalar_tensor = Tensor::new(&[1.], &[]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.], &[1]);
    assert!(scalar_tensor.is_scalar());

    let scalar_tensor = Tensor::new(&[1.], &[1, 1]);
    assert!(scalar_tensor.is_scalar());

    let non_scalar_tensor = Tensor::new(&[1., 2.], &[2]);
    assert!(!non_scalar_tensor.is_scalar());
}

#[test]
fn test_is_vector() {
    let vector_tensor = Tensor::new(&[1., 2., 3.], &[3]);
    assert!(vector_tensor.is_vector());

    let vector_tensor = Tensor::new(&[1., 2., 3.], &[1, 3]);
    assert!(vector_tensor.is_vector());

    let vector_tensor = Tensor::new(&[1., 2., 3.], &[3, 1]);
    assert!(vector_tensor.is_vector());

    let non_vector_tensor = Tensor::new(&[1.], &[]);
    assert!(!non_vector_tensor.is_vector());

    let non_vector_tensor = Tensor::new(&[1.], &[1]);
    assert!(!non_vector_tensor.is_vector());

    let non_vector_tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert!(!non_vector_tensor.is_vector());

    let non_vector_tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2, 1]);
    assert!(!non_vector_tensor.is_vector());
}

#[test]
fn test_is_matrix() {
    let matrix_tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    assert!(matrix_tensor.is_matrix());

    let non_matrix_tensor = Tensor::new(&[1., 2., 3.], &[3]);
    assert!(!non_matrix_tensor.is_matrix());

    let non_matrix_tensor = Tensor::new(&[1.], &[]);
    assert!(!non_matrix_tensor.is_matrix());

    let non_matrix_tensor = Tensor::new(&[1.], &[1, 1]);
    assert!(!non_matrix_tensor.is_matrix());

    let non_matrix_tensor = Tensor::new(&[1., 2.], &[2, 1]);
    assert!(!non_matrix_tensor.is_matrix());

    let non_matrix_tensor = Tensor::new(&[1.], &[1, 1, 1]);
    assert!(!non_matrix_tensor.is_matrix());
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑判断张量是否为标量、向量、矩阵↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

#[test]
fn test_has_zero_value() {
    // 测试不包含零值的张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(!tensor.has_zero_value());

    // 测试包含零值的张量
    let tensor = Tensor::new(&[1.0, 0.0, 3.0, 4.0], &[2, 2]);
    assert!(tensor.has_zero_value());

    // 测试全为零的张量
    let tensor = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
    assert!(tensor.has_zero_value());

    // 测试标量张量
    let tensor = Tensor::new(&[0.0], &[]);
    assert!(tensor.has_zero_value());

    let tensor = Tensor::new(&[1.0], &[]);
    assert!(!tensor.has_zero_value());

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 0.0, 5.0, 6.0], &[2, 3, 1]);
    assert!(tensor.has_zero_value());
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓size↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_size() {
    // 测试标量
    let tensor = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor.size(), 1);

    // 测试1维向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor.size(), 3);

    // 测试2维矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(tensor.size(), 4);

    // 测试3维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    assert_eq!(tensor.size(), 6);

    // 测试高维张量
    let tensor = Tensor::new(&[1.0; 24], &[2, 3, 2, 2]);
    assert_eq!(tensor.size(), 24);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑size↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓广播工具函数↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// 测试 broadcast_shape 函数 - 成功场景
#[test]
fn test_broadcast_shape_success() {
    // Python 参考: tests/python/tensor_reference/broadcast_utils_reference.py
    let success_cases: &[(&[usize], &[usize], &[usize])] = &[
        // 相同形状
        (&[3, 4], &[3, 4], &[3, 4]),
        (&[2, 3, 4], &[2, 3, 4], &[2, 3, 4]),
        (&[], &[], &[]),
        // 标量广播
        (&[], &[3], &[3]),
        (&[3], &[], &[3]),
        (&[], &[2, 3], &[2, 3]),
        (&[2, 3], &[], &[2, 3]),
        // 低维广播到高维
        (&[3, 4], &[4], &[3, 4]),
        (&[4], &[3, 4], &[3, 4]),
        (&[2, 3, 4], &[4], &[2, 3, 4]),
        (&[2, 3, 4], &[3, 4], &[2, 3, 4]),
        // 带 1 的广播
        (&[3, 4], &[1, 4], &[3, 4]),
        (&[3, 4], &[3, 1], &[3, 4]),
        (&[3, 1], &[1, 4], &[3, 4]),
        (&[1, 4], &[3, 1], &[3, 4]),
        // 高维
        (&[2, 3, 4], &[1, 3, 1], &[2, 3, 4]),
        (&[2, 1, 4], &[1, 3, 1], &[2, 3, 4]),
        (&[1, 1, 4], &[2, 3, 1], &[2, 3, 4]),
    ];

    for (shape_a, shape_b, expected) in success_cases {
        let result = broadcast_shape(shape_a, shape_b);
        assert_eq!(
            result,
            Some(expected.to_vec()),
            "broadcast_shape({:?}, {:?}) 失败",
            shape_a,
            shape_b
        );
    }
}

/// 测试 broadcast_shape 函数 - 失败场景
#[test]
fn test_broadcast_shape_failure() {
    let failure_cases: &[(&[usize], &[usize])] = &[
        (&[3], &[4]),
        (&[2, 3], &[3, 2]),
        (&[2, 3], &[4]),
        (&[2, 3, 4], &[2, 5, 4]),
    ];

    for (shape_a, shape_b) in failure_cases {
        let result = broadcast_shape(shape_a, shape_b);
        assert_eq!(
            result, None,
            "broadcast_shape({:?}, {:?}) 应返回 None",
            shape_a, shape_b
        );
    }
}

/// 测试 sum_to_shape 方法
#[test]
fn test_sum_to_shape() {
    // Python 参考: tests/python/tensor_reference/broadcast_utils_reference.py
    let test_cases: &[(&[usize], &[usize], &[f32], &[f32])] = &[
        // [2, 3] -> [2, 3] (no-op)
        (
            &[2, 3],
            &[2, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        // [3, 4] -> [1, 4] (sum axis 0)
        (
            &[3, 4],
            &[1, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[15.0, 18.0, 21.0, 24.0],
        ),
        // [3, 4] -> [3, 1] (sum axis 1)
        (
            &[3, 4],
            &[3, 1],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[10.0, 26.0, 42.0],
        ),
        // [3, 4] -> [1, 1] (sum all)
        (
            &[3, 4],
            &[1, 1],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[78.0],
        ),
        // [3, 4] -> [4] (reduce dimension)
        (
            &[3, 4],
            &[4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[15.0, 18.0, 21.0, 24.0],
        ),
        // [2, 3, 4] -> [1, 3, 1] (sum axes 0 and 2)
        (
            &[2, 3, 4],
            &[1, 3, 1],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[68.0, 100.0, 132.0],
        ),
        // [2, 3, 4] -> [4] (reduce to vector)
        (
            &[2, 3, 4],
            &[4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[66.0, 72.0, 78.0, 84.0],
        ),
        // [2, 3, 4] -> [3, 4] (remove first dimension)
        (
            &[2, 3, 4],
            &[3, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ],
            &[
                14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0,
            ],
        ),
    ];

    for (input_shape, target_shape, input_data, expected_data) in test_cases {
        let tensor = Tensor::new(*input_data, *input_shape);
        let result = tensor.sum_to_shape(*target_shape);
        let expected = Tensor::new(*expected_data, *target_shape);

        assert_eq!(
            result, expected,
            "sum_to_shape 失败: {:?} -> {:?}",
            input_shape, target_shape
        );
    }
}

/// 测试 sum_axis_keepdims 方法
#[test]
fn test_sum_axis_keepdims() {
    // [2, 3] sum axis 0 -> [1, 3]
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let result = t.sum_axis_keepdims(0);
    assert_eq!(result.shape(), &[1, 3]);
    assert_eq!(result.data_as_slice(), &[5.0, 7.0, 9.0]);

    // [2, 3] sum axis 1 -> [2, 1]
    let result = t.sum_axis_keepdims(1);
    assert_eq!(result.shape(), &[2, 1]);
    assert_eq!(result.data_as_slice(), &[6.0, 15.0]);

    // [2, 3, 4] sum axis 1 -> [2, 1, 4]
    let t = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
        ],
        &[2, 3, 4],
    );
    let result = t.sum_axis_keepdims(1);
    assert_eq!(result.shape(), &[2, 1, 4]);
    assert_eq!(
        result.data_as_slice(),
        &[15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0]
    );
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑广播工具函数↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓内存连续性↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// 测试 is_contiguous 正确识别连续/非连续内存
#[test]
fn test_is_contiguous() {
    // 普通创建的 Tensor 应该是连续的
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(t.is_contiguous());

    // stack 沿 axis=0 应该是连续的
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let stacked = Tensor::stack(&[&a, &b], 0, false);
    assert!(stacked.is_contiguous());

    // stack 沿 axis=1 也应该是连续的（因为我们自动转换了）
    let c = Tensor::new(&[1.0, 2.0], &[2, 1]);
    let d = Tensor::new(&[3.0, 4.0], &[2, 1]);
    let concat_axis1 = Tensor::stack(&[&c, &d], 1, false);
    assert!(concat_axis1.is_contiguous());
}

/// 测试 to_vec 总是能获取数据（无论内存布局）
#[test]
fn test_to_vec() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let vec = t.to_vec();
    assert_eq!(vec, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// 测试 stack 结果总是可以安全调用 data_as_slice
#[test]
fn test_stack_always_contiguous() {
    // 沿 axis=1 拼接（这是之前会产生非连续结果的操作）
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
    let concat = Tensor::stack(&[&a, &b], 1, false);

    // 应该可以安全调用 data_as_slice 而不 panic
    assert_eq!(concat.shape(), &[2, 5]);
    let slice = concat.data_as_slice();
    assert_eq!(slice.len(), 10);
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑内存连续性↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓broadcast_to↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// 测试 broadcast_to 基本功能
#[test]
fn test_broadcast_to_basic() {
    // [3, 1] -> [3, 4]：沿 axis=1 广播
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let b = a.broadcast_to(&[3, 4]);
    assert_eq!(b.shape(), &[3, 4]);
    // 每一行都被复制 4 次
    assert_eq!(
        b.data_as_slice(),
        &[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    );

    // [1, 4] -> [3, 4]：沿 axis=0 广播
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let b = a.broadcast_to(&[3, 4]);
    assert_eq!(b.shape(), &[3, 4]);
    assert_eq!(
        b.data_as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );
}

/// 测试 broadcast_to 维度扩展（低维到高维）
#[test]
fn test_broadcast_to_expand_dims() {
    // [4] -> [3, 4]：增加 axis=0
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = a.broadcast_to(&[3, 4]);
    assert_eq!(b.shape(), &[3, 4]);
    assert_eq!(
        b.data_as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]
    );

    // [4] -> [2, 3, 4]：增加两个维度
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let b = a.broadcast_to(&[2, 3, 4]);
    assert_eq!(b.shape(), &[2, 3, 4]);
    assert_eq!(b.size(), 24);
    // 前 12 个和后 12 个应该相同
    let slice = b.data_as_slice();
    assert_eq!(&slice[0..12], &slice[12..24]);
}

/// 测试 broadcast_to 形状相同（快速路径）
#[test]
fn test_broadcast_to_same_shape() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = a.broadcast_to(&[2, 2]);
    assert_eq!(b.shape(), &[2, 2]);
    assert_eq!(b.data_as_slice(), a.data_as_slice());
}

/// 测试 broadcast_to 高维场景
#[test]
fn test_broadcast_to_high_dim() {
    // [1, 3, 1] -> [2, 3, 4]
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3, 1]);
    let b = a.broadcast_to(&[2, 3, 4]);
    assert_eq!(b.shape(), &[2, 3, 4]);

    // 验证数据：每个 [3, 4] 切片的列应该相同
    let slice = b.data_as_slice();
    // 第一个 [3, 4] 块
    assert_eq!(slice[0], 1.0); // [0, 0, 0]
    assert_eq!(slice[1], 1.0); // [0, 0, 1]
    assert_eq!(slice[4], 2.0); // [0, 1, 0]
    assert_eq!(slice[8], 3.0); // [0, 2, 0]
}

/// 测试 broadcast_to 与 sum_to_shape 的互逆性
#[test]
fn test_broadcast_to_inverse_of_sum_to_shape() {
    // 原始张量 [1, 4]
    let original = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    // 广播到 [3, 4]
    let broadcast = original.broadcast_to(&[3, 4]);
    assert_eq!(broadcast.shape(), &[3, 4]);

    // 用 sum_to_shape 收缩回去
    let summed = broadcast.sum_to_shape(&[1, 4]);
    assert_eq!(summed.shape(), &[1, 4]);

    // 结果应该是原始值的 3 倍（因为沿 axis=0 求和了 3 个相同的行）
    assert_eq!(summed.data_as_slice(), &[3.0, 6.0, 9.0, 12.0]);
}

/// 测试 broadcast_to 不兼容形状应 panic
#[test]
#[should_panic(expected = "broadcast_to")]
fn test_broadcast_to_incompatible_shape() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    a.broadcast_to(&[4]); // 3 无法广播到 4
}

/// 测试 broadcast_to 维度数超过目标应 panic
#[test]
#[should_panic(expected = "broadcast_to")]
fn test_broadcast_to_dim_exceed() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    a.broadcast_to(&[4]); // 2D 无法广播到 1D
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑broadcast_to↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
