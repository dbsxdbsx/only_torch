use crate::tensor::Tensor;
use ndarray::Axis;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓order↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_order() {
    // 1. 2维张量
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let tensor2 = Tensor::new(&[3., 4., 1., 2., 5., 6.], &[2, 3]);
    let ordered_tensor = tensor2.order();
    assert_eq!(tensor1, ordered_tensor);

    // 2. 3维张量
    let tensor1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 2, 3],
    );
    let tensor2 = Tensor::new(
        &[7., 8., 9., 10., 11., 12., 3., 4., 1., 2., 5., 6.],
        &[2, 2, 3],
    );
    let ordered_tensor = tensor2.order();
    assert_eq!(tensor1, ordered_tensor);
}

/// **回归测试**：`order` / `shuffle_mut_seeded` 对**非连续**输入（`permute` 视图）不得 panic，
/// 且按逻辑序处理（与物化连续副本结果一致）。
#[test]
fn test_order_shuffle_noncontiguous() {
    let base = Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0], &[2, 3]);
    let nc = base.permute(&[1, 0]); // [3,2] 非连续
    let contig = nc.clone().into_contiguous();
    assert!(!nc.is_contiguous());

    // order：排序结果与连续副本一致
    assert_eq!(nc.order(), contig.order(), "order 非连续应等于连续副本");

    // shuffle_mut_seeded：同 seed + 同逻辑值 → 同结果
    let mut a = nc.clone();
    let mut b = contig.clone();
    a.shuffle_mut_seeded(None, 123);
    b.shuffle_mut_seeded(None, 123);
    assert_eq!(a, b, "shuffle_mut_seeded 非连续应与连续副本同结果");
}

#[test]
fn test_order_mut() {
    // 1. 2维张量
    let tensor1 = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let mut tensor2 = Tensor::new(&[3., 4., 1., 2., 5., 6.], &[2, 3]);
    tensor2.order_mut();
    assert_eq!(tensor1, tensor2);

    // 2. 3维张量
    let tensor1 = Tensor::new(
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[2, 2, 3],
    );
    let mut tensor2 = Tensor::new(
        &[7., 8., 9., 10., 11., 12., 3., 4., 1., 2., 5., 6.],
        &[2, 2, 3],
    );
    tensor2.order_mut();
    assert_eq!(tensor1, tensor2);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑order↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓shuffle↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// **契约金测试**：`shuffled_row_indices_seeded` + `select_rows` 必须与
/// 「整集 clone + `shuffle_mut_seeded(Some(0), seed)` + `narrow` 切片」逐 bit 等价
/// （演化 mini-batch 索引 shuffle 路径的等价性前提，覆盖 2D/3D、多种子、尾批）。
#[test]
fn test_shuffled_row_indices_select_rows_逐bit等价旧路径() {
    for &(rows, cols) in &[(7usize, 3usize), (32, 5)] {
        let data: Vec<f32> = (0..rows * cols).map(|v| v as f32 * 1.5 - 3.0).collect();
        for seed in [0u64, 42, 20260704] {
            // 2D
            let base = Tensor::new(&data, &[rows, cols]);
            let mut old_path = base.clone();
            old_path.shuffle_mut_seeded(Some(0), seed);
            let indices = Tensor::shuffled_row_indices_seeded(rows, seed);
            let new_path = base.select_rows(&indices);
            assert_eq!(old_path, new_path, "2D 全集置换应逐 bit 相同");

            // narrow 批切片 vs 索引子段 gather（含尾批）
            let bs = 5;
            let mut offset = 0;
            while offset < rows {
                let end = (offset + bs).min(rows);
                let old_batch = old_path.narrow(0, offset, end - offset);
                let new_batch = base.select_rows(&indices[offset..end]);
                assert_eq!(
                    old_batch, new_batch,
                    "batch [{offset}, {end}) 应逐 bit 相同"
                );
                offset = end;
            }

            // 3D（序列型标签场景）
            let base3 = Tensor::new(&data, &[rows, cols, 1]);
            let mut old3 = base3.clone();
            old3.shuffle_mut_seeded(Some(0), seed);
            assert_eq!(
                old3,
                base3.select_rows(&indices),
                "3D 应复用同一置换且逐 bit 相同"
            );
        }
    }
}

#[test]
fn test_select_rows_重复与乱序索引() {
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let picked = t.select_rows(&[2, 0, 2]);
    assert_eq!(
        picked,
        Tensor::new(&[5.0, 6.0, 1.0, 2.0, 5.0, 6.0], &[3, 2])
    );
}

#[test]
fn test_shuffle() {
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0,
    ];
    let shape = &[6, 6];
    let tensor = Tensor::new(data, shape);

    // 1. 仅打乱第1个维度（打乱后的形状仍一致，但数据不一致）
    let shuffled_tensor_row = tensor.shuffle(Some(0));
    assert_eq!(tensor.shape(), shuffled_tensor_row.shape());
    assert_ne!(tensor.data, shuffled_tensor_row.data);
    // 1.1 虽然打乱后整体数据是不一致的，但是该张量每行的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in shuffled_tensor_row.data.axis_iter(Axis(0)) {
        assert!(tensor.data.axis_iter(Axis(0)).any(|r| r == row));
    }

    // 2. 仅打乱第2个维度（打乱后的形状仍一致，但数据不一致）
    let shuffled_tensor_col = tensor.shuffle(Some(1));
    assert_eq!(tensor.shape(), shuffled_tensor_col.shape());
    assert_ne!(tensor.data, shuffled_tensor_col.data);
    // 2.1 虽然打乱后整体数据是不一致的，但是该张量每列的数据总是能在另一个张量中的某列找到完全一致的数据
    for col in shuffled_tensor_col.data.axis_iter(Axis(1)) {
        assert!(tensor.data.axis_iter(Axis(1)).any(|c| c == col));
    }

    // 3. 全局打乱（打乱后的形状仍一致，但数据不一致）
    let tensor_shuffle = tensor.shuffle(None);
    assert_eq!(tensor.shape(), tensor_shuffle.shape());
    assert_ne!(tensor.data, tensor_shuffle.data);
    // 3.1 确保没有一行或一列和原来一样的
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(0))
            .all(|row| { tensor.data.axis_iter(Axis(0)).all(|r| r != row) })
    );
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(1))
            .all(|col| { tensor.data.axis_iter(Axis(1)).all(|r| r != col) })
    );
    // 3.2 重新排序后则应完全一致
    let ordered_tensor = tensor_shuffle.order();
    assert_eq!(tensor, ordered_tensor);
}

#[test]
fn test_shuffle_mut() {
    let data = &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
        32.0, 33.0, 34.0, 35.0, 36.0,
    ];
    let shape = &[6, 6];
    let tensor = Tensor::new(data, shape);

    // 1. 仅打乱第1个维度（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle_row = Tensor::new(data, shape);
    tensor_shuffle_row.shuffle_mut(Some(0));
    assert_eq!(tensor.shape(), tensor_shuffle_row.shape());
    assert_ne!(tensor.data, tensor_shuffle_row.data);
    // 1.1 虽然打乱后整体数据是不一致的，但是该张量每行的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in tensor_shuffle_row.data.axis_iter(Axis(0)) {
        assert!(tensor.data.axis_iter(Axis(0)).any(|r| r == row));
    }

    // 2. 仅打乱第2个维度（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle_col = Tensor::new(data, shape);
    tensor_shuffle_col.shuffle_mut(Some(1));
    assert_eq!(tensor.shape(), tensor_shuffle_col.shape());
    assert_ne!(tensor.data, tensor_shuffle_col.data);
    // 2.1 虽然打乱后整体数据是不一致的，但是该张量每列的数据总是能在另一个张量中的某行找到完全一致的数据
    for row in tensor_shuffle_col.data.axis_iter(Axis(1)) {
        assert!(tensor.data.axis_iter(Axis(1)).any(|r| r == row));
    }

    // 3. 全局打乱（打乱后的形状仍一致，但数据不一致）
    let mut tensor_shuffle = Tensor::new(data, shape);
    tensor_shuffle.shuffle_mut(None);
    assert_eq!(tensor.shape(), tensor_shuffle.shape());
    assert_ne!(tensor.data, tensor_shuffle.data);
    // 3.1 确保没有一行或一列和原来一样的
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(0))
            .all(|row| { tensor.data.axis_iter(Axis(0)).all(|r| r != row) })
    );
    assert!(
        tensor_shuffle
            .data
            .axis_iter(Axis(1))
            .all(|col| { tensor.data.axis_iter(Axis(1)).all(|r| r != col) })
    );
    let ordered_tensor = tensor_shuffle.order();
    // 3.2 重新排序后则应完全一致
    assert_eq!(tensor, ordered_tensor);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑shuffle↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓soft_update↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_soft_update_basic() {
    let mut target = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let source = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    target.soft_update(&source, 0.1);

    // target = 0.1 * source + 0.9 * target
    // = 0.1 * [10, 20, 30, 40] + 0.9 * [1, 2, 3, 4]
    // = [1, 2, 3, 4] + [0.9, 1.8, 2.7, 3.6]
    // = [1.9, 3.8, 5.7, 7.6]
    let expected = Tensor::new(&[1.9, 3.8, 5.7, 7.6], &[2, 2]);
    assert_eq!(target, expected);
}

#[test]
fn test_soft_update_tau_zero() {
    // tau=0: target 完全不变
    let mut target = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let source = Tensor::new(&[10.0, 20.0], &[1, 2]);

    target.soft_update(&source, 0.0);

    assert_eq!(target, Tensor::new(&[1.0, 2.0], &[1, 2]));
}

#[test]
fn test_soft_update_tau_one() {
    // tau=1: target 完全变为 source
    let mut target = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let source = Tensor::new(&[10.0, 20.0], &[1, 2]);

    target.soft_update(&source, 1.0);

    assert_eq!(target, Tensor::new(&[10.0, 20.0], &[1, 2]));
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑soft_update↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sort_along_axis↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_sort_along_axis_ascending() {
    // 1D 升序
    let t = Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
    let (sorted, indices) = t.sort_along_axis(0, false);
    assert_eq!(sorted.data.as_slice().unwrap(), &[1.0, 1.0, 3.0, 4.0, 5.0]);
    // 索引指向原始位置
    assert_eq!(indices[[0]], 1.0); // 值 1.0 原在 index 1
    assert_eq!(indices[[1]], 3.0); // 值 1.0 原在 index 3
    assert_eq!(indices[[2]], 0.0); // 值 3.0 原在 index 0
}

#[test]
fn test_sort_along_axis_descending() {
    let t = Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]);
    let (sorted, indices) = t.sort_along_axis(0, true);
    assert_eq!(sorted.data.as_slice().unwrap(), &[5.0, 4.0, 3.0, 1.0, 1.0]);
    assert_eq!(indices[[0]], 4.0); // 值 5.0 原在 index 4
    assert_eq!(indices[[1]], 2.0); // 值 4.0 原在 index 2
}

#[test]
fn test_sort_along_axis_2d_axis1() {
    // 2D 沿 axis=1（行内排序）
    let t = Tensor::new(&[3.0, 1.0, 2.0, 6.0, 4.0, 5.0], &[2, 3]);
    let (sorted, indices) = t.sort_along_axis(1, false);
    assert_eq!(sorted.shape(), &[2, 3]);
    // 第一行: [3,1,2] -> [1,2,3]
    assert_eq!(sorted[[0, 0]], 1.0);
    assert_eq!(sorted[[0, 1]], 2.0);
    assert_eq!(sorted[[0, 2]], 3.0);
    assert_eq!(indices[[0, 0]], 1.0);
    assert_eq!(indices[[0, 1]], 2.0);
    assert_eq!(indices[[0, 2]], 0.0);
    // 第二行: [6,4,5] -> [4,5,6]
    assert_eq!(sorted[[1, 0]], 4.0);
    assert_eq!(sorted[[1, 1]], 5.0);
    assert_eq!(sorted[[1, 2]], 6.0);
}

#[test]
fn test_sort_along_axis_2d_axis0() {
    // 2D 沿 axis=0（列内排序）
    let t = Tensor::new(&[3.0, 1.0, 6.0, 4.0], &[2, 2]);
    let (sorted, idx) = t.sort_along_axis(0, false);
    // 第一列: [3,6] -> [3,6]（已排序），索引 [0,1]
    assert_eq!(sorted[[0, 0]], 3.0);
    assert_eq!(sorted[[1, 0]], 6.0);
    assert_eq!(idx[[0, 0]], 0.0);
    assert_eq!(idx[[1, 0]], 1.0);
    // 第二列: [1,4] -> [1,4]（已排序），索引 [0,1]
    assert_eq!(sorted[[0, 1]], 1.0);
    assert_eq!(sorted[[1, 1]], 4.0);
    assert_eq!(idx[[0, 1]], 0.0);
    assert_eq!(idx[[1, 1]], 1.0);
}

#[test]
fn test_sort_along_axis_already_sorted() {
    // 已排序输入，索引应为 0,1,2,...
    let t = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let (sorted, indices) = t.sort_along_axis(0, false);
    assert_eq!(sorted, t);
    for i in 0..4 {
        assert_eq!(indices[[i]], i as f32);
    }
}

#[test]
fn test_sort_along_axis_shape_preserved() {
    let t = Tensor::new(&[2.0, 1.0, 4.0, 3.0, 6.0, 5.0], &[2, 3]);
    let (sorted, idx) = t.sort_along_axis(1, false);
    assert_eq!(sorted.shape(), &[2, 3]);
    assert_eq!(idx.shape(), &[2, 3]);
}

#[test]
#[should_panic(expected = "sort_along_axis: axis")]
fn test_sort_along_axis_invalid_axis() {
    let t = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let _ = t.sort_along_axis(1, false); // 1D 张量只有 axis=0
}

#[test]
fn test_sort_along_axis_negative_values() {
    // 含负数排序
    let t = Tensor::new(&[-3.0, 2.0, -1.0, 0.0, 5.0, -4.0], &[6]);
    let (sorted, _) = t.sort_along_axis(0, false);
    assert_eq!(
        sorted.data.as_slice().unwrap(),
        &[-4.0, -3.0, -1.0, 0.0, 2.0, 5.0]
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sort_along_axis↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
