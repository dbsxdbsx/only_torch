use crate::tensor::Tensor;

impl Tensor {
    /// 实现矩阵乘法。只接受1维或2维的张量，否则会触发panic。
    pub fn mat_mul(&self, other: &Tensor) -> Tensor {
        // 检查输入的张量维度
        let self_dims = self.dims();
        let other_dims = other.dims();
        assert!(self_dims == 1 || self_dims == 2, "输入的张量维度必须为1或2");
        assert!(
            other_dims == 1 || other_dims == 2,
            "输入的张量维度必须为1或2"
        );

        // if self_dims
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
