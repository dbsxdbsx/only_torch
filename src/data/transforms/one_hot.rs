//! one-hot 编码（迁移自原 transforms.rs）

use crate::tensor::Tensor;

/// 将类别索引转换为 one-hot 编码
///
/// # 参数
/// - `labels`: 类别索引 Tensor，形状 [N] 或 [N, 1]，值为 `0..num_classes`
/// - `num_classes`: 类别总数
///
/// # 返回
/// one-hot 编码 Tensor，形状 [N, `num_classes`]
///
/// # 示例
/// ```ignore
/// let labels = Tensor::new(&[0.0, 2.0, 1.0], &[3]);
/// let one_hot = one_hot(&labels, 3);
/// // 结果: [[1,0,0], [0,0,1], [0,1,0]]
/// ```
pub fn one_hot(labels: &Tensor, num_classes: usize) -> Tensor {
    let flat = labels.flatten();
    let n = flat.size();

    let mut data = vec![0.0; n * num_classes];
    for i in 0..n {
        let class_idx = flat[[i]] as usize;
        if class_idx < num_classes {
            data[i * num_classes + class_idx] = 1.0;
        }
    }

    Tensor::new(&data, &[n, num_classes])
}
