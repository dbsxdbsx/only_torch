//! 随机旋转
//!
//! 在 [-degrees, +degrees] 范围内随机旋转图像，使用双线性插值。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机旋转
///
/// 对输入张量 [C, H, W] 或 [H, W] 在 `[-degrees, +degrees]` 范围内随机旋转，
/// 使用双线性插值填充，超出边界的像素用 `fill_value` 填充。
///
/// # 示例
///
/// ```ignore
/// let rot = RandomRotation::new(15.0); // ±15°
/// let output = rot.apply(&image_tensor);
/// ```
pub struct RandomRotation {
    degrees: f64,
    fill_value: f32,
}

impl RandomRotation {
    /// 创建随机旋转变换
    ///
    /// # 参数
    /// - `degrees`: 最大旋转角度（绝对值），单位为度
    pub fn new(degrees: f64) -> Self {
        assert!(
            degrees >= 0.0,
            "RandomRotation: degrees 必须 >= 0，得到 {degrees}"
        );
        Self {
            degrees,
            fill_value: 0.0,
        }
    }

    /// 设置超出边界的填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl Transform for RandomRotation {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        if self.degrees == 0.0 {
            return tensor.clone();
        }

        let mut rng = rand::thread_rng();
        let angle = rng.gen_range(-self.degrees..=self.degrees);

        rotate(tensor, angle, self.fill_value)
    }
}

/// 旋转图像（确定性版本，供内部和测试使用）
///
/// # 参数
/// - `tensor`: [C, H, W] 或 [H, W]
/// - `angle_deg`: 旋转角度（度），正值逆时针
/// - `fill_value`: 超出边界的填充值
pub(crate) fn rotate(tensor: &Tensor, angle_deg: f64, fill_value: f32) -> Tensor {
    let shape = tensor.shape();
    let ndim = shape.len();
    assert!(
        ndim == 2 || ndim == 3,
        "RandomRotation: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
    );

    let (c, h, w) = if ndim == 2 {
        (1, shape[0], shape[1])
    } else {
        (shape[0], shape[1], shape[2])
    };

    let angle_rad = angle_deg.to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // 中心点
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;

    let flat: Vec<f32> = tensor.flatten_view().to_vec();
    let mut data = vec![fill_value; c * h * w];

    // 对每个输出像素，反向映射到输入像素并做双线性插值
    for ch in 0..c {
        let ch_offset = ch * h * w;
        for out_y in 0..h {
            for out_x in 0..w {
                // 反向旋转：从输出坐标求输入坐标
                let dx = out_x as f64 - cx;
                let dy = out_y as f64 - cy;
                let in_x = cos_a * dx + sin_a * dy + cx;
                let in_y = -sin_a * dx + cos_a * dy + cy;

                // 双线性插值
                if let Some(val) = bilinear_interpolate(&flat, ch_offset, h, w, in_y, in_x) {
                    data[ch_offset + out_y * w + out_x] = val;
                }
            }
        }
    }

    Tensor::new(&data, shape)
}

/// 双线性插值
///
/// 在 flat 数据中，从 `ch_offset` 开始的 h×w 区域中插值
fn bilinear_interpolate(
    flat: &[f32],
    ch_offset: usize,
    h: usize,
    w: usize,
    y: f64,
    x: f64,
) -> Option<f32> {
    // 边界检查（含浮点容差）
    let eps = 1e-6;
    if x < -eps || x > (w - 1) as f64 + eps || y < -eps || y > (h - 1) as f64 + eps {
        return None;
    }

    // 钳制到有效范围
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    let v00 = flat[ch_offset + y0 * w + x0] as f64;
    let v01 = flat[ch_offset + y0 * w + x1] as f64;
    let v10 = flat[ch_offset + y1 * w + x0] as f64;
    let v11 = flat[ch_offset + y1 * w + x1] as f64;

    let val = v00 * (1.0 - dx) * (1.0 - dy)
        + v01 * dx * (1.0 - dy)
        + v10 * (1.0 - dx) * dy
        + v11 * dx * dy;

    Some(val as f32)
}
