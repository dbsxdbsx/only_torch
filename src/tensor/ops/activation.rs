/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量激活与数学函数
 */

use super::super::next_source_id;
use crate::tensor::Tensor;

impl Tensor {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓relu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 ReLU 激活函数
    ///
    /// `relu(x) = max(0, x)`
    pub fn relu(&self) -> Self {
        let data = self.data.mapv(|x| if x > 0.0 { x } else { 0.0 });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑relu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓leaky_relu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Leaky ReLU 激活函数
    ///
    /// `leaky_relu(x, alpha) = x if x > 0, else alpha * x`
    ///
    /// 当 `alpha = 0` 时等价于标准 ReLU。
    ///
    /// # 参数
    /// - `alpha`: 负半轴斜率（非负数）
    pub fn leaky_relu(&self, alpha: f32) -> Self {
        let data = self.data.mapv(|x| if x > 0.0 { x } else { alpha * x });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑leaky_relu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softplus↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用数值稳定的 SoftPlus 激活函数
    ///
    /// `softplus(x) = ln(1 + e^x)`
    ///
    /// 数值稳定策略：
    /// - `x > 20`: 直接返回 x（避免 e^x 溢出）
    /// - `0 < x <= 20`: `x + ln(1 + e^(-x))`（恒等变换）
    /// - `x <= 0`: `ln(1 + e^x)`（标准公式）
    pub fn softplus(&self) -> Self {
        const THRESHOLD: f32 = 20.0;
        let data = self.data.mapv(|val| {
            if val > THRESHOLD {
                val
            } else if val > 0.0 {
                val + (-val).exp().ln_1p()
            } else {
                val.exp().ln_1p()
            }
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softplus↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓step_fn↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Step（阶跃）函数
    ///
    /// `step(x) = 1 if x >= 0, else 0`
    ///
    /// Step 函数不可微，梯度恒为 0（在 Node 层处理）。
    pub fn step_fn(&self) -> Self {
        let data = self.data.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑step_fn↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓gelu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 GELU 激活函数（tanh 近似版，GPT-2 风格）
    ///
    /// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
    pub fn gelu(&self) -> Self {
        const SQRT_2_OVER_PI: f32 = 0.7978845608; // sqrt(2/pi)
        const COEFF: f32 = 0.044715;
        let data = self.data.mapv(|x| {
            let z = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
            0.5 * x * (1.0 + z.tanh())
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑gelu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓swish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Swish/SiLU 激活函数
    ///
    /// `swish(x) = x * sigmoid(x)`
    pub fn swish(&self) -> Self {
        let data = self.data.mapv(|x| {
            let sig = 1.0 / (1.0 + (-x).exp());
            x * sig
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑swish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓elu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 ELU 激活函数
    ///
    /// `elu(x, alpha) = x if x > 0, else alpha * (exp(x) - 1)`
    pub fn elu(&self, alpha: f32) -> Self {
        let data = self.data.mapv(|x| {
            if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑elu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓selu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 SELU 激活函数
    ///
    /// `selu(x) = LAMBDA * elu(x, ALPHA)`，使用固定常数
    pub fn selu(&self) -> Self {
        const LAMBDA: f32 = 1.0507009873554805;
        const ALPHA: f32 = 1.6732632423543772;
        let data = self.data.mapv(|x| {
            if x > 0.0 { LAMBDA * x } else { LAMBDA * ALPHA * (x.exp() - 1.0) }
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑selu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓mish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Mish 激活函数
    ///
    /// `mish(x) = x * tanh(softplus(x))`
    ///
    /// 内部使用数值稳定的 softplus 计算。
    pub fn mish(&self) -> Self {
        const THRESHOLD: f32 = 20.0;
        let data = self.data.mapv(|x| {
            // 数值稳定的 softplus
            let sp = if x > THRESHOLD {
                x
            } else if x > 0.0 {
                x + (-x).exp().ln_1p()
            } else {
                x.exp().ln_1p()
            };
            x * sp.tanh()
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑mish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_swish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 HardSwish 激活函数
    ///
    /// 分段定义：`0 if x <= -3, x if x >= 3, x*(x+3)/6 otherwise`
    pub fn hard_swish(&self) -> Self {
        let data = self.data.mapv(|x| {
            if x <= -3.0 { 0.0 } else if x >= 3.0 { x } else { x * (x + 3.0) / 6.0 }
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_swish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_sigmoid↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 HardSigmoid 激活函数
    ///
    /// 分段定义：`0 if x <= -3, 1 if x >= 3, (x+3)/6 otherwise`
    pub fn hard_sigmoid(&self) -> Self {
        let data = self.data.mapv(|x| {
            if x <= -3.0 { 0.0 } else if x >= 3.0 { 1.0 } else { (x + 3.0) / 6.0 }
        });
        Self { data, source_id: next_source_id() }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_sigmoid↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /// 计算张量每个元素的平方根
    pub fn sqrt(&self) -> Self {
        let sqrt_data = self.data.mapv(f32::sqrt);
        Self { data: sqrt_data, source_id: next_source_id() }
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓tanh↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用双曲正切函数(tanh)
    ///
    /// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.tanh();
    /// // y ≈ [0.0, 0.7616, -0.7616]
    /// ```
    pub fn tanh(&self) -> Self {
        let data = self.data.mapv(f32::tanh);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素应用双曲正切函数(tanh)
    pub fn tanh_mut(&mut self) {
        self.data.mapv_inplace(f32::tanh);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑tanh↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sigmoid↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Sigmoid 函数
    ///
    /// sigmoid(x) = 1 / (1 + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.sigmoid();
    /// // y ≈ [0.5, 0.7311, 0.2689]
    /// ```
    pub fn sigmoid(&self) -> Self {
        let data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素应用 Sigmoid 函数
    pub fn sigmoid_mut(&mut self) {
        self.data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sigmoid↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓exp↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算指数函数 e^x
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, 2.0], &[3]);
    /// let y = x.exp();
    /// // y ≈ [1.0, 2.7183, 7.3891]
    /// ```
    pub fn exp(&self) -> Self {
        let data = self.data.mapv(f32::exp);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素计算指数函数
    pub fn exp_mut(&mut self) {
        self.data.mapv_inplace(f32::exp);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑exp↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ln↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算自然对数 ln(x)
    ///
    /// # 注意
    /// 对于 x <= 0 的元素，结果为 NaN 或 -inf
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.7183, 7.3891], &[3]);
    /// let y = x.ln();
    /// // y ≈ [0.0, 1.0, 2.0]
    /// ```
    pub fn ln(&self) -> Self {
        let data = self.data.mapv(f32::ln);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素计算自然对数
    pub fn ln_mut(&mut self) {
        self.data.mapv_inplace(f32::ln);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ln↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓pow↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算幂运算
    ///
    /// # 参数
    /// - `exponent`: 指数（f32 常量）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    /// let y = x.powf(2.0);
    /// // y = [1.0, 4.0, 9.0]
    /// ```
    pub fn powf(&self, exponent: f32) -> Self {
        let data = self.data.mapv(|x| x.powf(exponent));
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑pow↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓square↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算平方
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    /// let y = x.square();
    /// // y = [1.0, 4.0, 9.0, 16.0]
    /// ```
    pub fn square(&self) -> Self {
        let data = self.data.mapv(|x| x * x);
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑square↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓reciprocal↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算倒数: 1/x
    ///
    /// 注意：输入 x 不应包含 0，否则结果为 Inf。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 4.0, 5.0], &[4]);
    /// let y = x.reciprocal();
    /// // y = [1.0, 0.5, 0.25, 0.2]
    /// ```
    pub fn reciprocal(&self) -> Self {
        let data = self.data.mapv(|x| 1.0 / x);
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑reciprocal↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log10↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算以 10 为底的对数
    ///
    /// 注意：输入 x 必须为正数。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 10.0, 100.0, 1000.0], &[4]);
    /// let y = x.log10();
    /// // y = [0.0, 1.0, 2.0, 3.0]
    /// ```
    pub fn log10(&self) -> Self {
        let data = self.data.mapv(f32::log10);
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log10↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log2↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算以 2 为底的对数
    ///
    /// 注意：输入 x 必须为正数。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 4.0, 8.0], &[4]);
    /// let y = x.log2();
    /// // y = [0.0, 1.0, 2.0, 3.0]
    /// ```
    pub fn log2(&self) -> Self {
        let data = self.data.mapv(f32::log2);
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log2↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓relu6↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// ReLU6 激活：min(max(0, x), 6)
    ///
    /// 移动端和量化网络常用。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-1.0, 0.0, 3.0, 7.0], &[4]);
    /// let y = x.relu6();
    /// // y = [0.0, 0.0, 3.0, 6.0]
    /// ```
    pub fn relu6(&self) -> Self {
        let data = self.data.mapv(|x| x.max(0.0).min(6.0));
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑relu6↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_tanh↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// HardTanh 激活：min(max(min_val, x), max_val)
    ///
    /// # 参数
    /// - `min_val`: 最小值下限
    /// - `max_val`: 最大值上限
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-2.0, -0.5, 0.5, 2.0], &[4]);
    /// let y = x.hard_tanh(-1.0, 1.0);
    /// // y = [-1.0, -0.5, 0.5, 1.0]
    /// ```
    pub fn hard_tanh(&self, min_val: f32, max_val: f32) -> Self {
        let data = self.data.mapv(|x| x.max(min_val).min(max_val));
        Self {
            data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_tanh↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓one_hot↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 将整数索引张量转换为 one-hot 编码张量
    ///
    /// 输入张量中的值被视为整数索引（向下取整），输出在对应位置为 1.0，其余为 0.0。
    ///
    /// # 参数
    /// - `num_classes`: 类别总数（one-hot 编码的维度）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let indices = Tensor::new(&[0.0, 2.0, 1.0], &[3]);
    /// let encoded = indices.one_hot(3);
    /// // encoded = [[1,0,0], [0,0,1], [0,1,0]], shape [3, 3]
    /// ```
    pub fn one_hot(&self, num_classes: usize) -> Self {
        let flat = self.flatten_view();
        let n = flat.len();
        let mut data = vec![0.0f32; n * num_classes];
        for i in 0..n {
            let idx = flat[i] as usize;
            assert!(idx < num_classes, "one_hot: 索引 {} 超出类别数 {}", idx, num_classes);
            data[i * num_classes + idx] = 1.0;
        }
        // 输出形状：原形状 + [num_classes]
        let mut new_shape = self.shape().to_vec();
        new_shape.push(num_classes);
        Self::new(&data, &new_shape)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑one_hot↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 softmax（数值稳定版本）
    ///
    /// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    ///
    /// 通过先减去最大值再计算 exp，避免数值溢出。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴计算 softmax
    ///
    /// # 返回
    /// 新张量，形状与输入相同，沿指定轴的元素和为 1
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let probs = x.softmax(1);  // 沿最后一维计算
    ///
    /// // 每行和为 1
    /// assert!((probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]] - 1.0).abs() < 1e-6);
    /// assert!((probs[[1, 0]] + probs[[1, 1]] + probs[[1, 2]] - 1.0).abs() < 1e-6);
    ///
    /// // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    /// assert!((probs[[0, 2]] - 0.6652).abs() < 0.001);
    /// ```
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis); // 沿 axis 取最大值
        let max_broadcast = max_vals.unsqueeze(axis as i8); // 恢复维度以便广播

        // x - max(x)
        let shifted = self - &max_broadcast;

        // exp(x - max)
        let exp_vals = shifted.exp();

        // sum(exp)
        let sum_exp = exp_vals.sum_axis_keepdims(axis);

        // exp / sum
        &exp_vals / &sum_exp
    }

    /// 沿最后一维计算 softmax（数值稳定版本）
    ///
    /// 等价于 `self.softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let probs = logits.softmax_last_dim();
    /// // probs ≈ [[0.0900, 0.2447, 0.6652]]
    /// ```
    pub fn softmax_last_dim(&self) -> Self {
        assert!(self.dimension() > 0, "softmax_last_dim: 张量维度必须大于 0");
        self.softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log_softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 log_softmax（数值稳定版本）
    ///
    /// `log_softmax(x) = log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))`
    ///
    /// 比直接计算 `softmax(x).ln()` 更数值稳定，避免 softmax 输出接近 0 时的精度问题。
    ///
    /// # 参数
    /// - `axis`: 计算 softmax 的轴
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let log_probs = logits.log_softmax(1);
    ///
    /// // 检查形状
    /// assert_eq!(log_probs.shape(), &[2, 3]);
    ///
    /// // log_softmax 输出应该都是负数（因为 softmax 输出 < 1）
    /// assert!(log_probs[[0, 0]] < 0.0);
    /// assert!(log_probs[[0, 1]] < 0.0);
    /// assert!(log_probs[[0, 2]] < 0.0);
    ///
    /// // exp(log_softmax) 应该等于 softmax
    /// let probs = log_probs.exp();
    /// let sum = probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]];
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn log_softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "log_softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis);
        let max_broadcast = max_vals.unsqueeze(axis as i8);

        // shifted = x - max(x)
        let shifted = self - &max_broadcast;

        // log_sum_exp = log(sum(exp(shifted)))
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis_keepdims(axis);
        let log_sum_exp = sum_exp.ln();

        // log_softmax = shifted - log_sum_exp
        &shifted - &log_sum_exp
    }

    /// 沿最后一维计算 log_softmax（数值稳定版本）
    ///
    /// 等价于 `self.log_softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let log_probs = logits.log_softmax_last_dim();
    ///
    /// // log_softmax([1,2,3]) ≈ [-2.407, -1.407, -0.407]
    /// assert!((log_probs[[0, 0]] - (-2.407)).abs() < 0.01);
    /// ```
    pub fn log_softmax_last_dim(&self) -> Self {
        assert!(
            self.dimension() > 0,
            "log_softmax_last_dim: 张量维度必须大于 0"
        );
        self.log_softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log_softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
