use crate::tensor::Tensor;

impl Tensor {
    /// 实现矩阵乘法。只接受2阶张量，否则会触发panic。
    /// 需要保证前一个张量的列数（col）等于后一个张量的行数（row），否则也会触发panic。
    pub fn mat_mul(&self, other: &Tensor) -> Tensor {
        // 检查输入的张量维度
        let self_dims = self.dimension();
        let other_dims = other.dimension();
        assert!(self_dims == 2, "输入的张量维度必须为2");
        assert!(other_dims == 2, "输入的张量维度必须为2");
        // 检查前一个张量的列数是否等于后一个张量的行数
        assert!(
            self.shape()[1] == other.shape()[0],
            "前一个张量的列数必须等于后一个张量的行数"
        );
        // 将动态维度数组转换为常量维度数组
        let self_data = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let other_data = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        // 执行矩阵乘法
        let result_data = self_data.dot(&other_data);
        // 创建并返回新的张量
        Tensor {
            data: result_data.into_dyn(),
        }
    }
}
