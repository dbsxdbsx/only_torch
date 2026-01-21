use crate::assert_panic;
use crate::tensor::Tensor;
use crate::{tensor_slice, tensor_slice_view};

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓slice↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_slice() {
    // 创建一个4维张量 [2, 3, 1, 4] 形状
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // [0,0,0,:]
        5.0, 6.0, 7.0, 8.0, // [0,1,0,:]
        9.0, 10.0, 11.0, 12.0, // [0,2,0,:]
        13.0, 14.0, 15.0, 16.0, // [1,0,0,:]
        17.0, 18.0, 19.0, 20.0, // [1,1,0,:]
        21.0, 22.0, 23.0, 24.0, // [1,2,0,:]
    ];
    let tensor = Tensor::new(&data, &[2, 3, 1, 4]);

    // 测试混合切片: [:, 0:2, 0, 1:3]
    // - 第1维: 选择全部
    // - 第2维: 选择范围0..2
    // - 第3维: 选择索引0
    // - 第4维: 选择范围1..3
    let result1 = tensor.slice(&[&(..), &(0..2), &0, &(1..3)]);
    let result2 = tensor.slice(&[&(..), &(0..2), &(..), &(1..3)]);

    // 验证切片后的数据
    // 原始数据中对应的切片应该选择:
    // [0,:,:,:] -> [[2.0, 3.0], [6.0, 7.0]]
    // [1,:,:,:] -> [[14.0, 15.0], [18.0, 19.0]]
    // NOTE: 本实现中，单个索引时保持维度为1，而 NumPy 会自动压缩掉该维度
    // 例如：对形状为 [2,3,1,4] 的张量，切片 [:, 0:2, 0, 1:3]
    // 本实现输出形状为 [2,2,1,2]
    // NumPy 输出形状为 [2,2,2]（自动压缩了第3维）
    let expected = Tensor::new(&[2.0, 3.0, 6.0, 7.0, 14.0, 15.0, 18.0, 19.0], &[2, 2, 1, 2]);
    assert_eq!(result1, expected);
    assert_eq!(result2, expected);

    // 测试完整范围: 用`(..)`表示
    let full_slice = tensor.slice(&[&(..), &(..), &(..), &(..)]);
    assert_eq!(full_slice, tensor);

    // 测试完整范围：用具体数字范围表示
    let full_slice = tensor.slice(&[&(0..2), &(0..3), &(0..1), &(0..4)]);
    assert_eq!(full_slice, tensor);
    let full_slice = tensor.slice(&[&(0..=1), &(0..=2), &(0..1), &(0..=3)]);
    assert_eq!(full_slice, tensor);
}

#[test]
fn test_slice_panic_cases() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // [0,0,0,:]
        5.0, 6.0, 7.0, 8.0, // [0,1,0,:]
        9.0, 10.0, 11.0, 12.0, // [0,2,0,:]
        13.0, 14.0, 15.0, 16.0, // [1,0,0,:]
    ];
    let tensor = Tensor::new(&data, &[2, 2, 1, 4]);

    // 测试空索引列表
    assert_panic!(tensor.slice(&[]), "slice(_view)无法接受空索引");

    // 测试维度不匹配（索引数量少于张量维度）
    assert_panic!(
        tensor.slice(&[&1, &2, &3]),
        "slice(_view)仅提供了3个维度的索引，但目标张量是4维"
    );

    // 测试维度不匹配（索引数量多于张量维度）
    assert_panic!(
        tensor.slice(&[&1, &2, &3, &4, &5]),
        "slice(_view)提供了5个维度的索引，但目标张量只有4维"
    );

    // 测试包含空范围的切片（0..0 会导致该维度大小为0）
    assert_panic!(
        tensor.slice(&[&(0..0), &1, &0, &(0..4)]),
        "slice(_view)无法接受某个维度为零数据范围的索引"
    );

    // 测试索引超出范围
    assert_panic!(
        tensor.slice(&[&5, &1, &0, &3]), // 第1维的索引5超出范围
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
    assert_panic!(
        tensor.slice(&[&(0..3), &1, &0, &3]), // 第1维的范围0..3超出实际大小2
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑slice↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓slice_view↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_slice_view() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // [0,0,0,:]
        5.0, 6.0, 7.0, 8.0, // [0,1,0,:]
        9.0, 10.0, 11.0, 12.0, // [0,2,0,:]
        13.0, 14.0, 15.0, 16.0, // [1,0,0,:]
        17.0, 18.0, 19.0, 20.0, // [1,1,0,:]
        21.0, 22.0, 23.0, 24.0, // [1,2,0,:]
    ];
    let tensor = Tensor::new(&data, &[2, 3, 1, 4]);

    // 测试基本切片功能
    let slice = tensor.slice_view(&[&1, &(0..2), &0, &(1..3)]);
    assert_eq!(slice.shape(), &[1, 2, 1, 2]);
    let expected = vec![
        14.0, 15.0, // [1,0,0,1:3]
        18.0, 19.0, // [1,1,0,1:3]
    ];
    let expected_tensor = Tensor::new(&expected, &[1, 2, 1, 2]);
    assert_eq!(slice, expected_tensor.data.view());

    // 测试完整切片
    let full_slice = tensor.slice_view(&[&(0..2), &(0..3), &(0..1), &(0..4)]);
    assert_eq!(full_slice, tensor.data.view());
    let full_slice = tensor.slice_view(&[&(0..=1), &(0..=2), &(0..1), &(0..=3)]);
    assert_eq!(full_slice, tensor.data.view());
}

#[test]
fn test_slice_view_panic_cases() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // [0,0,0,:]
        5.0, 6.0, 7.0, 8.0, // [0,1,0,:]
        9.0, 10.0, 11.0, 12.0, // [0,2,0,:]
        13.0, 14.0, 15.0, 16.0, // [1,0,0,:]
    ];
    let tensor = Tensor::new(&data, &[2, 2, 1, 4]);

    // 测试空索引列表
    assert_panic!(tensor.slice_view(&[]), "slice(_view)无法接受空索引");

    // 测试维度不匹配（索引数量少于张量维度）
    assert_panic!(
        tensor.slice_view(&[&1, &2, &3]),
        "slice(_view)仅提供了3个维度的索引，但目标张量是4维"
    );

    // 测试维度不匹配（索引数量多于张量维度）
    assert_panic!(
        tensor.slice_view(&[&1, &2, &3, &4, &5]),
        "slice(_view)提供了5个维度的索引，但目标张量只有4维"
    );

    // 测试包含空范围的切片（0..0 会导致该维度大小为0）
    assert_panic!(
        tensor.slice_view(&[&(0..0), &1, &0, &(0..4)]),
        "slice(_view)无法接受某个维度为零数据范围的索引"
    );

    // 测试索引超出范围
    assert_panic!(
        tensor.slice_view(&[&5, &1, &0, &3]), // 第1维的索引5超出范围
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
    assert_panic!(
        tensor.slice_view(&[&(0..3), &1, &0, &3]), // 第1维的范围0..3超出实际大小2
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑slice_view↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

#[test]
fn test_slice_macro() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // [0,0,0,:]
        5.0, 6.0, 7.0, 8.0, // [0,1,0,:]
        9.0, 10.0, 11.0, 12.0, // [0,2,0,:]
        13.0, 14.0, 15.0, 16.0, // [1,0,0,:]
    ];
    let tensor = Tensor::new(&data, &[2, 2, 1, 4]);

    // 测试宏的基本切片功能
    let slice1 = tensor_slice!(tensor, 0usize, 0..2usize, .., 1..3usize);
    let slice2 = tensor.slice(&[&0, &(0..2), &(..), &(1..3)]);
    assert_eq!(slice1, slice2);

    // 测试宏的视图版本
    let view1 = tensor_slice_view!(tensor, 1usize, .., 0usize, ..);
    let view2 = tensor.slice_view(&[&1, &(..), &0, &(..)]);
    assert_eq!(view1, view2);

    // 测试宏对RangeInclusive的支持
    let slice3 = tensor_slice!(tensor, 0usize..=1, 0usize..=1, 0usize..=0, 0usize..=3);
    let slice4 = tensor.slice(&[&(0..=1), &(0..=1), &(0..=0), &(0..=3)]);
    assert_eq!(slice3, slice4);

    // 测试宏的尾部逗号支持
    let slice5 = tensor_slice!(tensor, 0usize, 1usize, 0usize, 1..3usize,);
    let slice6 = tensor.slice(&[&0, &1, &0, &(1..3)]);
    assert_eq!(slice5, slice6);
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓select↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

#[test]
fn test_select_basic() {
    // 2D 张量: [3, 4]
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // row 0
        5.0, 6.0, 7.0, 8.0, // row 1
        9.0, 10.0, 11.0, 12.0, // row 2
    ];
    let tensor = Tensor::new(&data, &[3, 4]);

    // 选择第 1 行 (axis=0, index=1) → 形状 [4]
    let row1 = tensor.select(0, 1);
    assert_eq!(row1.shape(), &[4]);
    assert_eq!(row1, Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[4]));

    // 选择第 2 列 (axis=1, index=2) → 形状 [3]
    let col2 = tensor.select(1, 2);
    assert_eq!(col2.shape(), &[3]);
    assert_eq!(col2, Tensor::new(&[3.0, 7.0, 11.0], &[3]));
}

#[test]
fn test_select_3d_rnn_usecase() {
    // 3D 张量: [batch=2, seq_len=3, input=4]（RNN 典型输入）
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2, 3, 4]);

    // 选择时间步 t=1 (axis=1, index=1) → 形状 [2, 4]
    let x_t1 = tensor.select(1, 1);
    assert_eq!(x_t1.shape(), &[2, 4]);
    // batch 0, t=1: [5, 6, 7, 8]
    // batch 1, t=1: [17, 18, 19, 20]
    assert_eq!(
        x_t1,
        Tensor::new(&[5.0, 6.0, 7.0, 8.0, 17.0, 18.0, 19.0, 20.0], &[2, 4])
    );

    // 选择 batch=0 (axis=0, index=0) → 形状 [3, 4]
    let batch0 = tensor.select(0, 0);
    assert_eq!(batch0.shape(), &[3, 4]);
    assert_eq!(
        batch0,
        Tensor::new(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4]
        )
    );

    // 选择最后一个特征 (axis=2, index=3) → 形状 [2, 3]
    let last_feat = tensor.select(2, 3);
    assert_eq!(last_feat.shape(), &[2, 3]);
    assert_eq!(
        last_feat,
        Tensor::new(&[4.0, 8.0, 12.0, 16.0, 20.0, 24.0], &[2, 3])
    );
}

#[test]
fn test_select_4d_cnn_usecase() {
    // 4D 张量: [batch=2, channels=3, height=2, width=2]（CNN 典型输入）
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let tensor = Tensor::new(&data, &[2, 3, 2, 2]);

    // 选择通道 c=1 (axis=1, index=1) → 形状 [2, 2, 2]
    let channel1 = tensor.select(1, 1);
    assert_eq!(channel1.shape(), &[2, 2, 2]);
    // batch 0, c=1: [[5,6],[7,8]]
    // batch 1, c=1: [[17,18],[19,20]]
    assert_eq!(
        channel1,
        Tensor::new(&[5.0, 6.0, 7.0, 8.0, 17.0, 18.0, 19.0, 20.0], &[2, 2, 2])
    );
}

#[test]
fn test_select_1d() {
    // 1D 张量: [5]
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);

    // 选择 index=2 → 形状 [] (标量)
    let scalar = tensor.select(0, 2);
    let empty_shape: &[usize] = &[];
    assert_eq!(scalar.shape(), empty_shape); // 0 维标量
    assert_eq!(scalar, Tensor::new(&[3.0], &[]));
}

#[test]
fn test_select_vs_slice_comparison() {
    // 对比 select 和 slice 的区别
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // select: 降维
    let selected = tensor.select(0, 1); // 选择第 1 行
    assert_eq!(selected.shape(), &[3]); // 维度从 2D 变为 1D

    // slice: 保持维度
    let sliced = tensor_slice!(tensor, 1usize, ..);
    assert_eq!(sliced.shape(), &[1, 3]); // 维度仍为 2D，第一维为 1
}

#[test]
fn test_select_panic_axis_out_of_bounds() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_panic!(
        tensor.select(2, 0), // axis=2 超出 2D 张量
        "select: axis 2 超出张量维度 2"
    );
}

#[test]
fn test_select_panic_index_out_of_bounds() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_panic!(
        tensor.select(0, 5), // index=5 超出轴 0 的大小 2
        "select: index 5 超出轴 0 的大小 2"
    );
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑select↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓scatter_at↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/

/// 测试基本的 scatter_at 功能：2D 张量
#[test]
fn test_scatter_at_basic_2d() {
    // 目标张量: [3, 4]
    let mut target = Tensor::zeros(&[3, 4]);

    // 源张量: [1, 4]（要放入 axis=0, index=1）
    let source = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    target.scatter_at(0, 1, &source);

    // 验证：只有第 1 行被填充
    for row in 0..3 {
        for col in 0..4 {
            let expected = if row == 1 { (col + 1) as f32 } else { 0.0 };
            assert_eq!(target[[row, col]], expected);
        }
    }
}

/// 测试 3D 张量的 scatter_at（RNN 场景）
#[test]
fn test_scatter_at_3d_rnn_gradient() {
    // 模拟 RNN 反向传播：将梯度放回特定时间步
    // 目标张量: [batch=2, seq_len=3, input=4]
    let mut grad_input = Tensor::zeros(&[2, 3, 4]);

    // 源张量（某个时间步的梯度）: [2, 1, 4]
    let grad_t1 = Tensor::ones(&[2, 1, 4]);

    // 放入 axis=1, index=1
    grad_input.scatter_at(1, 1, &grad_t1);

    // 验证：只有 [:, 1, :] 被填充为 1.0
    for b in 0..2 {
        for t in 0..3 {
            for i in 0..4 {
                let expected = if t == 1 { 1.0 } else { 0.0 };
                assert_eq!(grad_input[[b, t, i]], expected);
            }
        }
    }
}

/// 测试 scatter_at 是 select 的逆操作
#[test]
fn test_scatter_at_inverse_of_select() {
    // 原始张量
    let original = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    // 使用 select 提取第 0 行
    let row0 = original.select(0, 0); // 形状 [3]

    // 将其放回一个全零张量的相同位置
    let mut restored = Tensor::zeros(&[2, 3]);
    // scatter_at 需要的 source 形状应该是 [1, 3]
    let row0_expanded = row0.reshape(&[1, 3]);
    restored.scatter_at(0, 0, &row0_expanded);

    // 验证第 0 行被正确恢复
    assert_eq!(restored[[0, 0]], 1.0);
    assert_eq!(restored[[0, 1]], 2.0);
    assert_eq!(restored[[0, 2]], 3.0);
    // 其他位置仍为 0
    assert_eq!(restored[[1, 0]], 0.0);
    assert_eq!(restored[[1, 1]], 0.0);
    assert_eq!(restored[[1, 2]], 0.0);
}

/// 测试多次 scatter_at（累积效果）
#[test]
fn test_scatter_at_multiple_times() {
    let mut target = Tensor::zeros(&[3, 4]);

    // 放入不同行
    let row0 = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]);
    let row1 = Tensor::new(&[2.0, 2.0, 2.0, 2.0], &[1, 4]);
    let row2 = Tensor::new(&[3.0, 3.0, 3.0, 3.0], &[1, 4]);

    target.scatter_at(0, 0, &row0);
    target.scatter_at(0, 1, &row1);
    target.scatter_at(0, 2, &row2);

    // 验证整个张量
    assert_eq!(target, Tensor::new(&[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.], &[3, 4]));
}

/// 测试 scatter_at 的 axis=1 场景
#[test]
fn test_scatter_at_axis_1() {
    let mut target = Tensor::zeros(&[2, 3, 4]);

    // 在 axis=1, index=2 处放入数据
    let source = Tensor::new(&[1.0; 8], &[2, 1, 4]); // [batch=2, 1, input=4]

    target.scatter_at(1, 2, &source);

    // 验证只有 [:, 2, :] 被填充
    for b in 0..2 {
        for t in 0..3 {
            for i in 0..4 {
                let expected = if t == 2 { 1.0 } else { 0.0 };
                assert_eq!(target[[b, t, i]], expected);
            }
        }
    }
}

/// 测试 scatter_at panic：axis 越界
#[test]
fn test_scatter_at_panic_axis_out_of_bounds() {
    let mut target = Tensor::zeros(&[2, 3]);
    let source = Tensor::zeros(&[2, 1]);

    assert_panic!(
        target.scatter_at(2, 0, &source),
        "scatter_at: axis 2 超出张量维度 2"
    );
}

/// 测试 scatter_at panic：index 越界
#[test]
fn test_scatter_at_panic_index_out_of_bounds() {
    let mut target = Tensor::zeros(&[2, 3]);
    let source = Tensor::zeros(&[1, 3]);

    assert_panic!(
        target.scatter_at(0, 5, &source),
        "scatter_at: index 5 超出轴 0 的大小 2"
    );
}

/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑scatter_at↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
