//! 随机仿射变换
//!
//! 对图像应用随机的旋转、平移、缩放、剪切组合变换，使用双线性插值。
//! 对应 PyTorch `torchvision.transforms.RandomAffine`。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机仿射变换
///
/// 对输入张量 [C, H, W] 或 [H, W] 应用随机仿射变换（旋转+平移+缩放+剪切）。
/// 超出边界的像素用 `fill_value` 填充。
///
/// # 参数
/// - `degrees`: 旋转角度范围 [-degrees, +degrees]（度）
/// - `translate`: 平移范围 (max_dx, max_dy)，以图像尺寸的比例表示
/// - `scale`: 缩放范围 (min, max)
/// - `shear`: 剪切角度范围 [-shear, +shear]（度）
///
/// # 示例
///
/// ```ignore
/// // 仅旋转 ±10°
/// let t = RandomAffine::new(10.0);
///
/// // 旋转 ±5° + 平移 5% + 缩放 0.9~1.1
/// let t = RandomAffine::new(5.0)
///     .translate(0.05, 0.05)
///     .scale(0.9, 1.1);
///
/// // 完整参数
/// let t = RandomAffine::new(15.0)
///     .translate(0.1, 0.1)
///     .scale(0.8, 1.2)
///     .shear(10.0);
/// ```
pub struct RandomAffine {
    degrees: f64,
    translate: Option<(f64, f64)>,
    scale_range: Option<(f64, f64)>,
    shear: Option<f64>,
    fill_value: f32,
}

impl RandomAffine {
    /// 创建随机仿射变换
    ///
    /// # 参数
    /// - `degrees`: 最大旋转角度（绝对值），单位为度
    pub fn new(degrees: f64) -> Self {
        assert!(degrees >= 0.0, "RandomAffine: degrees 必须 >= 0");
        Self {
            degrees,
            translate: None,
            scale_range: None,
            shear: None,
            fill_value: 0.0,
        }
    }

    /// 设置最大平移量（以图像尺寸的比例表示）
    ///
    /// # 参数
    /// - `max_dx`: 水平最大平移比例，范围 [0, 1]
    /// - `max_dy`: 垂直最大平移比例，范围 [0, 1]
    pub fn translate(mut self, max_dx: f64, max_dy: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&max_dx) && (0.0..=1.0).contains(&max_dy),
            "RandomAffine: translate 范围必须在 [0, 1]"
        );
        self.translate = Some((max_dx, max_dy));
        self
    }

    /// 设置缩放范围
    pub fn scale(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max, "RandomAffine: scale 范围无效");
        self.scale_range = Some((min, max));
        self
    }

    /// 设置最大剪切角度（度）
    pub fn shear(mut self, degrees: f64) -> Self {
        assert!(degrees >= 0.0, "RandomAffine: shear 必须 >= 0");
        self.shear = Some(degrees);
        self
    }

    /// 设置超出边界的填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl Transform for RandomAffine {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        let ndim = shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "RandomAffine: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );

        let (c, h, w) = if ndim == 2 {
            (1, shape[0], shape[1])
        } else {
            (shape[0], shape[1], shape[2])
        };

        // 采样随机参数
        let mut rng = rand::thread_rng();

        let angle = if self.degrees > 0.0 {
            rng.gen_range(-self.degrees..=self.degrees)
        } else {
            0.0
        };

        let (tx, ty) = if let Some((max_dx, max_dy)) = self.translate {
            let tx = rng.gen_range(-max_dx..=max_dx) * w as f64;
            let ty = rng.gen_range(-max_dy..=max_dy) * h as f64;
            (tx, ty)
        } else {
            (0.0, 0.0)
        };

        let s = if let Some((min_s, max_s)) = self.scale_range {
            rng.gen_range(min_s..=max_s)
        } else {
            1.0
        };

        let shear_rad = if let Some(max_shear) = self.shear {
            let shear_deg = rng.gen_range(-max_shear..=max_shear);
            shear_deg.to_radians()
        } else {
            0.0
        };

        // 构建仿射矩阵（围绕图像中心）
        // 变换顺序: 剪切 → 缩放 → 旋转 → 平移
        //
        // 正向变换矩阵 M = T * R * S * Sh * C^(-1)
        // 其中 C 是中心化平移
        //
        // 我们需要逆变换（从输出坐标反推输入坐标）用于双线性插值
        let cx = (w as f64 - 1.0) / 2.0;
        let cy = (h as f64 - 1.0) / 2.0;

        let angle_rad = angle.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let tan_sh = shear_rad.tan();

        // 逆变换: input_coord = inv_M * (output_coord - center - translate) + center
        // inv_M = inv(R * S * Sh)
        //
        // R * S * Sh = s * [cos_a, cos_a*tan_sh - sin_a]
        //                  [sin_a, sin_a*tan_sh + cos_a]
        //
        // det = s^2 * (cos_a*(sin_a*tan_sh + cos_a) - sin_a*(cos_a*tan_sh - sin_a))
        //     = s^2 * (cos_a*sin_a*tan_sh + cos_a^2 - sin_a*cos_a*tan_sh + sin_a^2)
        //     = s^2
        let a00 = s * cos_a;
        let a01 = s * (cos_a * tan_sh - sin_a);
        let a10 = s * sin_a;
        let a11 = s * (sin_a * tan_sh + cos_a);
        let det = s * s; // det of the 2x2 part

        // inv of 2x2: [a11, -a01; -a10, a00] / det
        let inv00 = a11 / det;
        let inv01 = -a01 / det;
        let inv10 = -a10 / det;
        let inv11 = a00 / det;

        let flat: Vec<f32> = tensor.flatten_view().to_vec();
        let mut out = vec![self.fill_value; c * h * w];

        for ch in 0..c {
            let ch_offset = ch * h * w;
            for out_y in 0..h {
                for out_x in 0..w {
                    // 输出坐标相对于中心+平移
                    let dx = out_x as f64 - cx - tx;
                    let dy = out_y as f64 - cy - ty;

                    // 逆变换得到输入坐标
                    let in_x = inv00 * dx + inv01 * dy + cx;
                    let in_y = inv10 * dx + inv11 * dy + cy;

                    // 边界检查 + 双线性插值
                    if in_x >= -0.5 && in_x <= w as f64 - 0.5
                        && in_y >= -0.5 && in_y <= h as f64 - 0.5
                    {
                        out[ch_offset + out_y * w + out_x] =
                            bilinear_sample(&flat, ch_offset, h, w, in_y, in_x);
                    }
                }
            }
        }

        Tensor::new(&out, shape)
    }
}

/// 双线性插值采样（clamp 边界）
fn bilinear_sample(flat: &[f32], ch_offset: usize, h: usize, w: usize, y: f64, x: f64) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let dx = (x - x0 as f64) as f32;
    let dy = (y - y0 as f64) as f32;

    let v00 = flat[ch_offset + y0 * w + x0];
    let v01 = flat[ch_offset + y0 * w + x1];
    let v10 = flat[ch_offset + y1 * w + x0];
    let v11 = flat[ch_offset + y1 * w + x1];

    v00 * (1.0 - dx) * (1.0 - dy)
        + v01 * dx * (1.0 - dy)
        + v10 * (1.0 - dx) * dy
        + v11 * dx * dy
}
