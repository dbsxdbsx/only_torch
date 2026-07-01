use crate::assert_panic;
use crate::errors::TensorError;
use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓reshape↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_reshape() {
    // 1.标量reshape
    let data = &[5.];
    let shape = &[];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[1, 1, 1];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2];
    assert_panic!(tensor.reshape(incompatible_shape));

    // 2.向量reshape
    let data = &[1., 2., 3., 4.];
    let shape = &[4, 1];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[2, 2];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 3];
    assert_panic!(tensor.reshape(incompatible_shape));

    // 3.矩阵reshape
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[2, 3];
    let tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[3, 2];
    assert_eq!(tensor.reshape(new_shape).shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 2];
    assert_panic!(tensor.reshape(incompatible_shape));
}

#[test]
fn test_reshape_mut() {
    // 1.标量reshape
    let data = &[5.];
    let shape = &[];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[1, 1, 1];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2];
    assert_panic!(tensor.reshape_mut(incompatible_shape));

    // 2.向量reshape
    let data = &[1., 2., 3., 4.];
    let shape = &[4, 1];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[2, 2];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 3];
    assert_panic!(tensor.reshape_mut(incompatible_shape));

    // 3.矩阵reshape
    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[2, 3];
    let mut tensor = Tensor::new(data, shape);
    // 成功情况
    let new_shape = &[3, 2];
    tensor.reshape_mut(new_shape);
    assert_eq!(tensor.shape(), new_shape);
    // 应当失败情况
    let incompatible_shape = &[2, 2];
    assert_panic!(tensor.reshape_mut(incompatible_shape));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑reshape↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

// stack 和 concat 测试已移至独立文件

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓split↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
/// 测试 split 方法（Tensor::concat 的逆操作）
#[test]
fn test_split_basic() {
    // 1. 沿 axis=0 分割 1D 张量
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let parts = t.split(0, &[2, 3]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], Tensor::new(&[1.0, 2.0], &[2]));
    assert_eq!(parts[1], Tensor::new(&[3.0, 4.0, 5.0], &[3]));

    // 2. 沿 axis=0 分割 2D 张量
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let parts = t.split(0, &[1, 2]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], Tensor::new(&[1.0, 2.0], &[1, 2]));
    assert_eq!(parts[1], Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]));

    // 3. 沿 axis=1 分割 2D 张量
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let parts = t.split(1, &[1, 2]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], Tensor::new(&[1.0, 4.0], &[2, 1]));
    assert_eq!(parts[1], Tensor::new(&[2.0, 3.0, 5.0, 6.0], &[2, 2]));

    // 4. 分割成多个部分
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    let parts = t.split(0, &[1, 2, 3]);
    assert_eq!(parts.len(), 3);
    assert_eq!(parts[0], Tensor::new(&[1.0], &[1]));
    assert_eq!(parts[1], Tensor::new(&[2.0, 3.0], &[2]));
    assert_eq!(parts[2], Tensor::new(&[4.0, 5.0, 6.0], &[3]));

    // 5. 分割成等大小的部分
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let parts = t.split(0, &[2, 2]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0], Tensor::new(&[1.0, 2.0], &[2]));
    assert_eq!(parts[1], Tensor::new(&[3.0, 4.0], &[2]));
}

#[test]
fn test_split_3d() {
    // 沿 axis=1 分割 3D 张量
    let t = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 3, 2],
    );
    let parts = t.split(1, &[1, 2]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].shape(), &[2, 1, 2]);
    assert_eq!(parts[1].shape(), &[2, 2, 2]);
    assert_eq!(parts[0], Tensor::new(&[1.0, 2.0, 7.0, 8.0], &[2, 1, 2]));
    assert_eq!(
        parts[1],
        Tensor::new(&[3.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0], &[2, 2, 2])
    );
}

#[test]
fn test_split_errors() {
    // 1. axis 超出维度
    let t = Tensor::new(&[1.0, 2.0], &[2]);
    assert_panic!(t.split(1, &[1, 1]), "split: axis 1 超出张量维度 1");

    // 2. sizes 之和不等于轴大小
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    assert_panic!(
        t.split(0, &[1, 2]),
        "split: sizes 之和 3 不等于轴 0 的大小 4"
    );

    // 3. sizes 之和超过轴大小
    let t = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_panic!(
        t.split(0, &[2, 3]),
        "split: sizes 之和 5 不等于轴 0 的大小 3"
    );
}

#[test]
fn test_split_stack_roundtrip() {
    // 验证 split 是 concat 的逆操作

    // 1. axis=0 (concat)
    let t1 = Tensor::new(&[1.0, 2.0], &[2]);
    let t2 = Tensor::new(&[3.0, 4.0, 5.0], &[3]);
    let stacked = Tensor::concat(&[&t1, &t2], 0);
    let parts = stacked.split(0, &[2, 3]);
    assert_eq!(parts[0], t1);
    assert_eq!(parts[1], t2);

    // 2. axis=1 (concat)
    let t1 = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let t2 = Tensor::new(&[3.0, 4.0, 5.0], &[1, 3]);
    let stacked = Tensor::concat(&[&t1, &t2], 1);
    let parts = stacked.split(1, &[2, 3]);
    assert_eq!(parts[0], t1);
    assert_eq!(parts[1], t2);

    // 3. 更复杂的 2D 情况
    let t1 = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
    let stacked = Tensor::concat(&[&t1, &t2], 1);
    let parts = stacked.split(1, &[2, 3]);
    assert_eq!(parts[0], t1);
    assert_eq!(parts[1], t2);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑split↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓(un)squeeze↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_squeeze() {
    // 测试标量
    let data = &[1.];
    let shape = &[];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[] as &[usize]);

    let data = &[1.];
    let shape = &[1];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[] as &[usize]);

    // 测试向量
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[4]);

    // 测试矩阵
    let data = &[1., 2., 3., 4.];
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    for shape in shapes {
        let squeezed_tensor = Tensor::new(data, shape).squeeze();
        assert_eq!(squeezed_tensor.shape(), &[4]);
    }

    // 测试高维张量
    let data = &[1., 2., 3., 4.];
    let shape = &[1, 1, 1, 4];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[4]);

    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[1, 2, 1, 3];
    let squeezed_tensor = Tensor::new(data, shape).squeeze();
    assert_eq!(squeezed_tensor.shape(), &[2, 3]);
}
#[test]
fn test_squeeze_mut() {
    // 测试标量
    let data = &[1.];
    let shape = &[];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[] as &[usize]);

    let data = &[1.];
    let shape = &[1];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[] as &[usize]);

    // 测试向量
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[4]);

    // 测试矩阵
    let data = &[1., 2., 3., 4.];
    let shapes: &[&[usize]] = &[&[4], &[1, 4], &[4, 1]];
    for shape in shapes {
        let mut tensor = Tensor::new(data, shape);
        tensor.squeeze_mut();
        assert_eq!(tensor.shape(), &[4]);
    }

    // 测试高维张量
    let data = &[1., 2., 3., 4.];
    let shape = &[1, 1, 1, 4];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[4]);

    let data = &[1., 2., 3., 4., 5., 6.];
    let shape = &[1, 2, 1, 3];
    let mut tensor = Tensor::new(data, shape);
    tensor.squeeze_mut();
    assert_eq!(tensor.shape(), &[2, 3]);
}

/// **回归测试**：`squeeze` / `squeeze_mut` 对**非连续**张量（`permute` 产物）必须先按
/// 逻辑序物化再 `into_shape`、不得 panic（与 `reshape` 一致）。以"物化连续副本"为参考。
#[test]
fn test_squeeze_noncontiguous() {
    // base [1,2,3]（含 1 维）→ permute[2,1,0] → [3,2,1] 非连续；squeeze 应得 [3,2]
    let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let nc = base.permute(&[2, 1, 0]);
    assert!(!nc.is_contiguous(), "permute 结果应为非连续");
    let contig = nc.clone().into_contiguous();

    let sq = nc.squeeze();
    assert_eq!(sq.shape(), &[3, 2]);
    assert_eq!(
        sq,
        contig.squeeze(),
        "squeeze 非连续应等于 squeeze 连续副本"
    );

    let mut nc_mut = base.permute(&[2, 1, 0]);
    nc_mut.squeeze_mut();
    assert_eq!(nc_mut.shape(), &[3, 2]);
    assert_eq!(
        nc_mut,
        contig.squeeze(),
        "squeeze_mut 非连续应等于 squeeze 连续副本"
    );
}

#[test]
fn test_unsqueeze() {
    // 测试在最前面增加一个维度
    let data = &[1., 2., 3., 4.];
    let shape = &[4];
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(0);
    assert_eq!(unsqueezed_tensor.shape(), &[1, 4]);
    // 测试在最后面增加一个维度
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-1);
    assert_eq!(unsqueezed_tensor.shape(), &[4, 1]);
    // 测试在中间增加一个维度
    let shape = &[2, 2];
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(1);
    assert_eq!(unsqueezed_tensor.shape(), &[2, 1, 2]);
    // 测试负索引
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-2);
    assert_eq!(unsqueezed_tensor.shape(), &[2, 1, 2]);
    let unsqueezed_tensor = Tensor::new(data, shape).unsqueeze(-3);
    assert_eq!(unsqueezed_tensor.shape(), &[1, 2, 2]);
    // 测试超出范围的索引
    assert_panic!(Tensor::new(data, shape).unsqueeze(3));
    assert_panic!(Tensor::new(data, shape).unsqueeze(-4));
}
#[test]
fn test_unsqueeze_mut() {
    // 测试在最前面增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    tensor.unsqueeze_mut(0);
    assert_eq!(tensor.shape(), &[1, 4]);
    // 测试在最后面增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[4]);
    tensor.unsqueeze_mut(-1);
    assert_eq!(tensor.shape(), &[4, 1]);
    // 测试在中间增加一个维度
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(1);
    assert_eq!(tensor.shape(), &[2, 1, 2]);
    // 测试负索引
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(-2);
    assert_eq!(tensor.shape(), &[2, 1, 2]);
    let mut tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    tensor.unsqueeze_mut(-3);
    assert_eq!(tensor.shape(), &[1, 2, 2]);
    // 测试超出范围的索引
    assert_panic!(Tensor::new(&[1., 2., 3., 4.], &[2, 2]).unsqueeze_mut(3));
    assert_panic!(Tensor::new(&[1., 2., 3., 4.], &[2, 2]).unsqueeze_mut(-4));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑(un)squeeze↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓permute↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_permute() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 应该成功的情况
    let permuted_tensor = tensor.permute(&[1, 0]);
    let expected_tensor = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
    assert_eq!(permuted_tensor, expected_tensor);
    // 应该失败的情况
    assert_panic!(tensor.permute(&[]), TensorError::PermuteNeedAtLeast2Dims);
    assert_panic!(tensor.permute(&[1]), TensorError::PermuteNeedAtLeast2Dims);
    assert_panic!(
        tensor.permute(&[1, 1]),
        TensorError::PermuteNeedUniqueAndInRange
    );
}

#[test]
fn test_permute_mut() {
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // 应该成功的情况
    tensor.permute_mut(&[1, 0]);
    let expected_tensor = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]);
    assert_eq!(tensor, expected_tensor);
    // 应该失败的情况
    assert_panic!(
        tensor.permute_mut(&[]),
        TensorError::PermuteNeedAtLeast2Dims
    );
    assert_panic!(
        tensor.permute_mut(&[1]),
        TensorError::PermuteNeedAtLeast2Dims
    );
    assert_panic!(
        tensor.permute_mut(&[1, 1]),
        TensorError::PermuteNeedUniqueAndInRange
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑permute↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓transpose↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_transpose() {
    // 测试标量
    let tensor = Tensor::new(&[1.0], &[]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[] as &[usize]);

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[3]); // 1维张量的转置仍然是1维的

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[2, 2]);
    assert_eq!(transposed, Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2]));

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let transposed = tensor.transpose();
    assert_eq!(transposed.shape(), &[3, 2, 1]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );
}

#[test]
fn test_transpose_mut() {
    // 测试标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[] as &[usize]);

    // 测试向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[3]); // 1维张量的转置仍然是1维的

    // 测试矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 3.0, 2.0, 4.0], &[2, 2]));

    // 测试高维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_mut();
    assert_eq!(tensor.shape(), &[3, 2, 1]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );
}

#[test]
fn test_transpose_dims() {
    // 1. 交换第0和第1维
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let transposed = tensor.transpose_dims(0, 1);
    assert_eq!(transposed.shape(), &[3, 2, 1]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );

    // 2. 交换第1和第2维
    let transposed = tensor.transpose_dims(1, 2);
    assert_eq!(transposed.shape(), &[2, 1, 3]);
    assert_eq!(
        transposed,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3])
    );

    // 3. 测试维度超出范围的情况
    assert_panic!(tensor.transpose_dims(0, 3));
}

#[test]
fn test_transpose_dims_mut() {
    // 1. 交换第0和第1维
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_dims_mut(0, 1);
    assert_eq!(tensor.shape(), &[3, 2, 1]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2, 1])
    );

    // 2. 交换第1和第2维
    tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.transpose_dims_mut(1, 2);
    assert_eq!(tensor.shape(), &[2, 1, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3])
    );

    // 3. 测试维度超出范围的情况
    assert_panic!(tensor.transpose_dims_mut(0, 3));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑transpose↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓flatten↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_flatten() {
    // 测试标量
    let tensor = Tensor::new(&[5.0], &[]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[1]);
    assert_eq!(flattened, Tensor::new(&[5.0], &[1]));

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[3]);
    assert_eq!(flattened, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[4]);
    assert_eq!(flattened, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let flattened = tensor.flatten();
    assert_eq!(flattened.shape(), &[6]);
    assert_eq!(
        flattened,
        Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6])
    );
}

#[test]
fn test_flatten_mut() {
    // 测试标量
    let mut tensor = Tensor::new(&[5.0], &[]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[1]);
    assert_eq!(tensor, Tensor::new(&[5.0], &[1]));

    // 测试向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 测试矩阵
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]));

    // 测试高维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    tensor.flatten_mut();
    assert_eq!(tensor.shape(), &[6]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]));
}

/// **回归测试**：`flatten` / `flatten_mut` 对**非连续**张量（`permute`/`transpose` 产物）
/// 必须按逻辑行主序展平、不得 panic（与 `reshape` 行为一致）。
#[test]
fn test_flatten_noncontiguous() {
    // [2,3] → permute → [3,2] 非连续视图；逻辑行主序应为转置后的顺序
    let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let permuted = base.permute(&[1, 0]); // 形状 [3,2]，非连续
    assert!(!permuted.is_contiguous(), "permute 结果应为非连续");

    // 逻辑序：行优先遍历 [3,2] → [1,4,2,5,3,6]
    let expected = Tensor::new(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[6]);

    let flattened = permuted.flatten();
    assert_eq!(flattened.shape(), &[6]);
    assert_eq!(flattened, expected, "flatten 应按逻辑行主序展平非连续张量");

    let mut permuted_mut = base.permute(&[1, 0]);
    permuted_mut.flatten_mut();
    assert_eq!(permuted_mut.shape(), &[6]);
    assert_eq!(
        permuted_mut, expected,
        "flatten_mut 应按逻辑行主序展平非连续张量"
    );
}

/// **回归测试**：`pad` / `slice_ranges` / `repeat` 对**非连续**输入（`permute` 视图）必须按
/// 逻辑行主序处理、不得 panic/静默算错。以"物化连续副本"为参考，两者结果应逐比特一致。
#[test]
fn test_pad_repeat_slice_noncontiguous() {
    // base [2,3] → permute[1,0] → [3,2] 非连续
    let base = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let nc = base.permute(&[1, 0]); // [3,2] 非连续
    let contig = nc.clone().into_contiguous(); // 逻辑等价的连续副本
    assert!(!nc.is_contiguous());
    assert!(contig.is_contiguous());

    // pad：四周各 pad 1
    let p_nc = nc.pad(&[(1, 1), (1, 1)], 0.0);
    let p_ref = contig.pad(&[(1, 1), (1, 1)], 0.0);
    assert_eq!(p_nc.shape(), &[5, 4]);
    assert_eq!(p_nc, p_ref, "pad 非连续应等于 pad 连续副本");

    // slice_ranges：取回中间区域（pad 的逆），应还原
    let s_nc = p_nc.slice_ranges(&[(1, 4), (1, 3)]);
    assert_eq!(s_nc, contig, "slice_ranges 非连续应还原逻辑张量");

    // slice_ranges 直接作用在**非连续**张量上（上游 permute）：取子块应等于连续副本同操作
    let sub_nc = nc.slice_ranges(&[(0, 2), (0, 2)]);
    let sub_ref = contig.slice_ranges(&[(0, 2), (0, 2)]);
    assert_eq!(sub_nc, sub_ref, "slice_ranges 非连续输入应等于连续副本");

    // repeat：沿两维重复
    let r_nc = nc.repeat(&[2, 3]);
    let r_ref = contig.repeat(&[2, 3]);
    assert_eq!(r_nc.shape(), &[6, 6]);
    assert_eq!(r_nc, r_ref, "repeat 非连续应等于 repeat 连续副本");
}

#[test]
fn test_flatten_view() {
    // 测试标量
    let tensor = Tensor::new(&[5.0], &[]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 1);
    assert_eq!(flattened[0], 5.0);

    // 测试向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 3);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0]);

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 4);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);

    // 测试高维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let flattened = tensor.flatten_view();
    assert_eq!(flattened.len(), 6);
    assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑flatten↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓diag↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_diag() {
    // 1. 测试标量 -> 标量 (保持形状不变)
    // 1维标量
    let tensor = Tensor::new(&[1.0], &[1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[1]);
    assert_eq!(diag, Tensor::new(&[1.0], &[1]));

    // 2维标量
    let tensor = Tensor::new(&[1.0], &[1, 1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[1, 1]);
    assert_eq!(diag, Tensor::new(&[1.0], &[1, 1]));

    // 2. 测试向量 -> 对角方阵
    // 1维向量
    let tensor = Tensor::new(&[1.0, 2.0], &[2]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[2, 2]);
    assert_eq!(diag, Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]));

    // 列向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3, 3]);
    assert_eq!(
        diag,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 行向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3, 3]);
    assert_eq!(
        diag,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 3. 测试方阵 -> 对角向量
    // 2x2方阵
    let tensor = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[2]);
    assert_eq!(diag, Tensor::new(&[1.0, 2.0], &[2]));

    // 3x3方阵
    let tensor = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    let diag = tensor.diag();
    assert_eq!(diag.shape(), &[3]);
    assert_eq!(diag, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 4. 测试非法输入
    // 0维标量
    let tensor = Tensor::new(&[1.0], &[]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");

    // 非方阵 (2x3)
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_panic!(tensor.diag(), "张量必须是标量、向量或方阵");

    // 非方阵 (3x2)
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_panic!(tensor.diag(), "张量必须是标量、向量或方阵");

    // 3维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");

    // 4维张量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    assert_panic!(tensor.diag(), "张量维度必须为1或2");
}

#[test]
fn test_diag_mut() {
    // 1. 测试标量 -> 标量 (保持形状不变)
    // 1维标量
    let mut tensor = Tensor::new(&[1.0], &[1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[1]);
    assert_eq!(tensor, Tensor::new(&[1.0], &[1]));

    // 2维标量
    let mut tensor = Tensor::new(&[1.0], &[1, 1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[1, 1]);
    assert_eq!(tensor, Tensor::new(&[1.0], &[1, 1]));

    // 2. 测试向量 -> 对角方阵
    // 1维向量
    let mut tensor = Tensor::new(&[1.0, 2.0], &[2]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]));

    // 列向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 行向量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3, 3]);
    assert_eq!(
        tensor,
        Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
    );

    // 3. 测试方阵 -> 对角向量
    // 2x2方阵
    let mut tensor = Tensor::new(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[2]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0], &[2]));

    // 3x3方阵
    let mut tensor = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3]);
    tensor.diag_mut();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor, Tensor::new(&[1.0, 2.0, 3.0], &[3]));

    // 4. 测试非法输入
    // 0维标量
    let mut tensor = Tensor::new(&[1.0], &[]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");

    // 非方阵 (2x3)
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_panic!(tensor.diag_mut(), "张量必须是标量、向量或方阵");

    // 非方阵 (3x2)
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    assert_panic!(tensor.diag_mut(), "张量必须是标量、向量或方阵");

    // 3维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2, 1]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");

    // 4维张量
    let mut tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2, 1]);
    assert_panic!(tensor.diag_mut(), "张量维度必须为1或2");
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑diag↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓jacobi_diag↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_jacobi_diag() {
    // 1. 标量情况：始终返回 [1, 1] 矩阵（与 diag() 不同）
    // 1D 标量
    let tensor = Tensor::new(&[0.25], &[1]);
    let jacobi = tensor.jacobi_diag();
    assert_eq!(jacobi.shape(), &[1, 1]);
    assert_eq!(jacobi, Tensor::new(&[0.25], &[1, 1]));

    // 2D 标量
    let tensor = Tensor::new(&[0.5], &[1, 1]);
    let jacobi = tensor.jacobi_diag();
    assert_eq!(jacobi.shape(), &[1, 1]);
    assert_eq!(jacobi, Tensor::new(&[0.5], &[1, 1]));

    // 2. 向量情况：与 diag() 行为一致
    let tensor = Tensor::new(&[0.1, 0.2, 0.3], &[3]);
    let jacobi = tensor.jacobi_diag();
    assert_eq!(jacobi.shape(), &[3, 3]);
    assert_eq!(
        jacobi,
        Tensor::new(&[0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3], &[3, 3])
    );

    // 3. 2D 张量情况：先 flatten 再转对角矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let jacobi = tensor.jacobi_diag();
    assert_eq!(jacobi.shape(), &[4, 4]);
    #[rustfmt::skip]
    let expected = Tensor::new(
        &[1.0, 0.0, 0.0, 0.0,
          0.0, 2.0, 0.0, 0.0,
          0.0, 0.0, 3.0, 0.0,
          0.0, 0.0, 0.0, 4.0],
        &[4, 4]
    );
    assert_eq!(jacobi, expected);

    // 4. 高维张量：flatten 后转对角矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let jacobi = tensor.jacobi_diag();
    assert_eq!(jacobi.shape(), &[6, 6]);

    // 5. 验证与 mat_mul 兼容性（核心用途）
    let derivative = Tensor::new(&[0.19661193], &[1]); // sigmoid'(0) ≈ 0.25
    let jacobi = derivative.jacobi_diag();
    assert_eq!(jacobi.shape(), &[1, 1]);
    // 可以进行 mat_mul 操作
    let upstream = Tensor::new(&[1.0], &[1, 1]);
    let result = upstream.mat_mul(&jacobi);
    assert_eq!(result.shape(), &[1, 1]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑jacobi_diag↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓narrow↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_narrow_basic() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let n = t.narrow(1, 1, 2);
    assert_eq!(n.shape(), &[2, 2]);
    assert!((n[[0, 0]] - 2.0).abs() < 1e-6);
    assert!((n[[0, 1]] - 3.0).abs() < 1e-6);
    assert!((n[[1, 0]] - 5.0).abs() < 1e-6);
    assert!((n[[1, 1]] - 6.0).abs() < 1e-6);
}

#[test]
fn test_narrow_axis0() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let n = t.narrow(0, 0, 2);
    assert_eq!(n.shape(), &[2, 2]);
    assert!((n[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((n[[1, 1]] - 4.0).abs() < 1e-6);
}

#[test]
fn test_narrow_full_length() {
    let t = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let n = t.narrow(1, 0, 3);
    assert_eq!(n.shape(), &[1, 3]);
    assert!((n[[0, 0]] - 1.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "narrow")]
fn test_narrow_out_of_bounds() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    t.narrow(1, 1, 3); // start(1) + length(3) > axis_size(2)
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑narrow↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
