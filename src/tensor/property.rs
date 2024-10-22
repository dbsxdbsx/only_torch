/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @Description  : 本类仅包含一些属性方法，不包含任何运算方法，所以不会需要用到mut
 * @LastEditors  : 老董
 * @LastEditTime : 2024-10-21 14:54:48
 */

use super::Tensor;
use ndarray::{ArrayViewD, ArrayViewMutD};

impl Tensor {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓快照/view(_mut)↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    pub fn view(&self) -> ArrayViewD<'_, f32> {
        ArrayViewD::from_shape(self.shape(), self.data.as_slice().unwrap()).unwrap()
    }
    pub fn view_mut(&mut self) -> ArrayViewMutD<'_, f32> {
        let shape = self.shape().to_owned();
        let slice_mut = self.data.as_slice_mut();
        ArrayViewMutD::from_shape(shape, slice_mut.unwrap()).unwrap()
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑快照/view(_mut)↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

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

    /// 计算张量中元素的数量
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 检查张量是否所有元素为`NaN`。
    /// 因为即使是仅含有1个`NaN`元素的也可能已经初始化，所以这里采用“所有”元素作为判断依据：
    /// 若所有元素为`NaN`，则判定为即未初始化，反之则已判定为已初始化。（单个元素的张量作为特例暂不特殊照顾）
    pub fn is_inited(&self) -> bool {
        !self.data.iter().all(|&x| x.is_nan())
    }

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Self) -> bool {
        self.shape() == other.shape()
    }

    /// 判断张量是否为标量
    /// 这里的判断标准是：若形状为空或形状各维数乘积为1，则认为是标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().product::<usize>() == 1
    }

    /// 判断张量中是否存在0值
    pub fn has_zero_value(&self) -> bool {
        self.data.iter().any(|&x| x == 0.)
    }

    /// 将张量转化为纯数（number）。若为标量，则返回Some(number)，否则返回None
    pub fn get_data_number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }
}
