/*
 * @Author       : 老董
 * @Date         : 2026-07-02
 * @Description  : 优化器融合原地更新原语（SGD / Adam）
 *
 * 设计要点：
 * - 单趟 Zip 遍历融合全部逐元素运算，消除优化器 step 中的临时张量链
 *   （旧路径每参数每步约 6~8 次全尺寸分配 + 2 次整块 clone）。
 * - 所有遍历用 ndarray `Zip`（stride 感知），对非连续布局（如上游 permute 反向
 *   产生的梯度）也按逻辑序正确，不依赖底层内存连续。
 * - 逐元素公式与旧张量表达式**逐 bit 一致**（无跨元素归约，融合不改变浮点顺序），
 *   由 `nn/tests/optimizer.rs` 的 fused-vs-reference 金测试钉死。
 */

use crate::tensor::Tensor;
use ndarray::Zip;

impl Tensor {
    /// SGD 原地更新：`self -= lr * grad`（单趟 Zip，零临时分配）
    ///
    /// 逐元素语义与 `self = &*self - lr * grad` 完全一致（含浮点顺序：
    /// 先 `lr * g` 再相减）。
    ///
    /// # Panics
    /// `self` 与 `grad` 形状不一致时 panic。
    pub fn sgd_step_inplace(&mut self, grad: &Self, lr: f32) {
        assert_eq!(
            self.shape(),
            grad.shape(),
            "sgd_step_inplace: 参数形状 {:?} 与梯度形状 {:?} 不一致",
            self.shape(),
            grad.shape()
        );
        Zip::from(&mut self.data).and(&grad.data).for_each(|p, &g| {
            *p -= lr * g;
        });
    }

    /// Adam 原地更新：单趟融合一阶矩 m、二阶矩 v 与参数三者的逐元素更新
    ///
    /// 逐元素执行（与旧张量表达式链同序，逐 bit 一致）：
    /// ```text
    /// m = β1·m + (1-β1)·g
    /// v = β2·v + (g·g)·(1-β2)
    /// denom = sqrt(v / bc2) + ε
    /// p -= step_size · (m / denom)          // step_size = lr / bc1
    /// ```
    ///
    /// # 参数
    /// - `grad`: 本步梯度
    /// - `m` / `v`: 优化器持有的一阶/二阶矩状态（原地更新）
    /// - `step_size`: `lr / (1 - β1^t)`（偏差修正已并入）
    /// - `bc2`: `1 - β2^t`（二阶矩偏差修正因子）
    ///
    /// # Panics
    /// 四个张量形状不一致时 panic。
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step_inplace(
        &mut self,
        grad: &Self,
        m: &mut Self,
        v: &mut Self,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        step_size: f32,
        bc2: f32,
    ) {
        assert!(
            self.shape() == grad.shape() && self.shape() == m.shape() && self.shape() == v.shape(),
            "adam_step_inplace: 形状不一致 param={:?} grad={:?} m={:?} v={:?}",
            self.shape(),
            grad.shape(),
            m.shape(),
            v.shape()
        );
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        Zip::from(&mut self.data)
            .and(&grad.data)
            .and(&mut m.data)
            .and(&mut v.data)
            .for_each(|p, &g, m_i, v_i| {
                *m_i = *m_i * beta1 + g * one_minus_beta1;
                *v_i = *v_i * beta2 + (g * g) * one_minus_beta2;
                let denom = (*v_i / bc2).sqrt() + epsilon;
                *p -= step_size * (*m_i / denom);
            });
    }
}
