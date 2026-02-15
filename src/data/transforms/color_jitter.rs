//! 随机色彩扰动
//!
//! 随机调整亮度 (brightness)、对比度 (contrast)、饱和度 (saturation)。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机色彩扰动
///
/// 对输入张量 [C, H, W] 随机调整亮度、对比度、饱和度。
/// 每个参数指定扰动范围 [1-val, 1+val]。
///
/// # 处理逻辑
/// - **亮度 (brightness)**: 所有像素乘以随机系数
/// - **对比度 (contrast)**: 向通道均值收缩/扩展
/// - **饱和度 (saturation)**: 向灰度图收缩/扩展（仅 3 通道 RGB）
///
/// # 示例
///
/// ```ignore
/// let jitter = ColorJitter::new(0.2, 0.2, 0.2);
/// let output = jitter.apply(&image_tensor);
/// ```
pub struct ColorJitter {
    brightness: f64,
    contrast: f64,
    saturation: f64,
}

impl ColorJitter {
    /// 创建色彩扰动变换
    ///
    /// # 参数
    /// - `brightness`: 亮度扰动范围（>= 0），实际系数在 [1-b, 1+b]
    /// - `contrast`: 对比度扰动范围（>= 0），实际系数在 [1-c, 1+c]
    /// - `saturation`: 饱和度扰动范围（>= 0），实际系数在 [1-s, 1+s]
    pub fn new(brightness: f64, contrast: f64, saturation: f64) -> Self {
        assert!(brightness >= 0.0, "brightness 必须 >= 0");
        assert!(contrast >= 0.0, "contrast 必须 >= 0");
        assert!(saturation >= 0.0, "saturation 必须 >= 0");
        Self {
            brightness,
            contrast,
            saturation,
        }
    }
}

impl Transform for ColorJitter {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        assert!(
            shape.len() == 3,
            "ColorJitter: 输入应为 3D [C, H, W]，得到 {}D",
            shape.len()
        );

        let mut rng = rand::thread_rng();
        let mut result = tensor.clone();

        // 随机打乱变换顺序
        let mut ops: Vec<u8> = vec![0, 1, 2]; // brightness, contrast, saturation
        // Fisher-Yates shuffle
        for i in (1..ops.len()).rev() {
            let j = rng.gen_range(0..=i);
            ops.swap(i, j);
        }

        for &op in &ops {
            match op {
                0 if self.brightness > 0.0 => {
                    let factor =
                        rng.gen_range((1.0 - self.brightness)..=(1.0 + self.brightness)) as f32;
                    result = &result * factor;
                }
                1 if self.contrast > 0.0 => {
                    let factor =
                        rng.gen_range((1.0 - self.contrast)..=(1.0 + self.contrast)) as f32;
                    result = adjust_contrast(&result, factor);
                }
                2 if self.saturation > 0.0 && shape[0] == 3 => {
                    let factor =
                        rng.gen_range((1.0 - self.saturation)..=(1.0 + self.saturation)) as f32;
                    result = adjust_saturation(&result, factor);
                }
                _ => {}
            }
        }

        // 裁剪到 [0, 1]，防止亮度/对比度/饱和度调整后溢出
        result.clip(0.0, 1.0)
    }
}

/// 调整对比度
///
/// `output = factor * input + (1 - factor) * mean`
fn adjust_contrast(tensor: &Tensor, factor: f32) -> Tensor {
    let shape = tensor.shape();
    let c = shape[0];
    let spatial_size: usize = shape[1..].iter().product();
    let flat = tensor.flatten_view();
    let mut data = vec![0.0f32; tensor.size()];

    for ch in 0..c {
        let offset = ch * spatial_size;
        // 计算通道均值
        let mean: f32 = (0..spatial_size)
            .map(|i| flat[offset + i])
            .sum::<f32>()
            / spatial_size as f32;

        for i in 0..spatial_size {
            data[offset + i] = factor * flat[offset + i] + (1.0 - factor) * mean;
        }
    }

    Tensor::new(&data, shape)
}

/// 调整饱和度（仅限 3 通道 RGB）
///
/// 将图像向灰度图收缩/扩展：
/// `output = factor * input + (1 - factor) * grayscale`
fn adjust_saturation(tensor: &Tensor, factor: f32) -> Tensor {
    let shape = tensor.shape();
    assert_eq!(shape[0], 3, "adjust_saturation: 仅支持 3 通道 RGB");
    let h = shape[1];
    let w = shape[2];
    let hw = h * w;
    let flat = tensor.flatten_view();
    let mut data = vec![0.0f32; tensor.size()];

    // ITU-R BT.601 灰度系数
    let wr = 0.2989_f32;
    let wg = 0.5870_f32;
    let wb = 0.1140_f32;

    for i in 0..hw {
        let r = flat[i];
        let g = flat[hw + i];
        let b = flat[2 * hw + i];
        let gray = wr * r + wg * g + wb * b;

        data[i] = factor * r + (1.0 - factor) * gray;
        data[hw + i] = factor * g + (1.0 - factor) * gray;
        data[2 * hw + i] = factor * b + (1.0 - factor) * gray;
    }

    Tensor::new(&data, shape)
}
