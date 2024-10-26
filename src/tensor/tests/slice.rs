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
        tensor.slice(&[&5, &1, &0, &3]), // 第一维的索引5超出范围
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
    assert_panic!(
        tensor.slice(&[&(0..3), &1, &0, &3]), // 第一维的范围0..3超出实际大小2
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
        tensor.slice_view(&[&5, &1, &0, &3]), // 第一维的索引5超出范围
        "slice(_view)无法接受某个维度的索引超出目标张量在该维度范围"
    );
    assert_panic!(
        tensor.slice_view(&[&(0..3), &1, &0, &3]), // 第一维的范围0..3超出实际大小2
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
    let slice1 = tensor_slice!(tensor, 0, 0..2, .., 1..3);
    let slice2 = tensor.slice(&[&0, &(0..2), &(..), &(1..3)]);
    assert_eq!(slice1, slice2);

    // 测试宏的视图版本
    let view1 = tensor_slice_view!(tensor, 1, .., 0, ..);
    let view2 = tensor.slice_view(&[&1, &(..), &0, &(..)]);
    assert_eq!(view1, view2);

    // 测试宏对RangeInclusive的支持
    let slice3 = tensor_slice!(tensor, 0..=1, 0..=1, 0..=0, 0..=3);
    let slice4 = tensor.slice(&[&(0..=1), &(0..=1), &(0..=0), &(0..=3)]);
    assert_eq!(slice3, slice4);

    // 测试宏的尾部逗号支持
    let slice5 = tensor_slice!(tensor, 0, 1, 0, 1..3,);
    let slice6 = tensor.slice(&[&0, &1, &0, &(1..3)]);
    assert_eq!(slice5, slice6);
}
