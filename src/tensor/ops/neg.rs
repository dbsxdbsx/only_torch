/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量的取反（Negate），对每个元素取反并返回新张量。
 */

use crate::tensor::Tensor;
use std::ops::Neg;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓（不）带引用的张量取反↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
impl Neg for Tensor {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            data: -&self.data,
        }
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        Tensor {
            data: -&self.data,
        }
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑（不）带引用的张量取反↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
