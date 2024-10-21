use super::Tensor;

impl Tensor {
    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 张量的维（dim）数、阶（rank）数
    /// 即`shape()`的元素个数--如：形状为`[]`的标量阶数为0，向量阶数为1，矩阵阶数为2，以此类推
    /// NOTE: 这里用`dimension`是参照了大多数库的命名规范，如PyTorch、NumPy等
    /// `但和MatrixSlow中的``dimension`不同（https://github.com/zc911/MatrixSlow/blob/a6db0d38802004449941e6644e609a2455b26327/matrixslow/core/node.py#L106），
    /// 后者是张量中所有元素的数量，在本库中请使用`size()`方法来获取
    pub fn dimension(&self) -> usize {
        self.data.ndim()
    }

    /// 计算张量中所有元素的数量
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }

    /// 判断张量是否为标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().all(|x| *x == 1)
    }

    /// 转化为纯数（number）。若为标量，则返回Some(number)，否则返回None
    pub fn number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }
}
