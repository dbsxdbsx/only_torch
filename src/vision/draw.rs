use super::{ImageBufferEnum, Vision};
use crate::tensor::Tensor;
use crate::utils::traits::image::ForImageBuffer;
use image::{ColorType, Luma, Rgb};

impl Vision {
    /// 在图像上绘制一个圆形。
    /// `tensor`：输入的图像张量；
    /// `center`：圆心坐标；
    /// `radius`：圆的半径；
    /// `rgb_color`：圆的颜色(若是灰色)；
    /// `thickness`：圆的轮廓线宽度。
    ///
    /// 注：若`thickness`为0或超过`radius`，则填充整个圆。
    pub fn draw_circle(
        tensor: &Tensor,
        center: (usize, usize),
        radius: usize,
        rgb_color: [u8; 3],
        thickness: usize,
    ) -> Result<Tensor, String> {
        // 检查是否为图像
        let image_type = tensor.is_image()?;
        let mut buf = match image_type {
            ColorType::L8 => ImageBufferEnum::Luma(tensor.to_image_buff_for_luma8()),
            ColorType::Rgb8 => ImageBufferEnum::Rgb(tensor.to_image_buff_for_rgb8()),
            _ => todo!(),
        };
        let (x, y) = center;
        let inner_radius = radius.saturating_sub(thickness);

        for dy in 0..radius * 2 {
            for dx in 0..radius * 2 {
                let xt = x as i32 + dx as i32 - radius as i32;
                let yt = y as i32 + dy as i32 - radius as i32;
                let distance_squared = (xt - x as i32).pow(2) + (yt - y as i32).pow(2);

                if distance_squared <= radius.pow(2) as i32
                    && (thickness == 0 || distance_squared > inner_radius.pow(2) as i32)
                {
                    match &mut buf {
                        ImageBufferEnum::Rgb(buf) => {
                            buf.put_pixel(xt as u32, yt as u32, Rgb(rgb_color));
                        }
                        ImageBufferEnum::Luma(buf) => {
                            buf.put_pixel(xt as u32, yt as u32, Luma([rgb_color[0]]));
                        }
                    }
                }
            }
        }

        // 将ImageBuffer转回Tensor
        let new_tensor = match buf {
            ImageBufferEnum::Rgb(buf) => buf.to_tensor()?,
            ImageBufferEnum::Luma(buf) => buf.to_tensor()?,
        };
        Ok(new_tensor)
    }

    /// 在图像上绘制一个矩形。
    /// `tensor`：输入的图像张量；
    /// `center`：矩形中心坐标；
    /// `height`：矩形的高度；
    /// `width`：矩形的宽度；
    /// `rgb_color`：矩形的颜色；
    /// `thickness`：矩形的轮廓线宽度。
    ///
    /// 注：若`thickness`为0或超过`radius`，则填充整个矩形。
    pub fn draw_rectangle(
        tensor: &Tensor,
        center: (usize, usize),
        height: usize,
        width: usize,
        rgb_color: [u8; 3],
        thickness: usize,
    ) -> Result<Tensor, String> {
        // 检查是否为图像
        let image_type = tensor.is_image()?;
        let mut buf = match image_type {
            ColorType::L8 => ImageBufferEnum::Luma(tensor.to_image_buff_for_luma8()),
            ColorType::Rgb8 => ImageBufferEnum::Rgb(tensor.to_image_buff_for_rgb8()),
            _ => todo!(),
        };
        let (x, y) = center;
        let left = x.saturating_sub(width / 2);
        let top = y.saturating_sub(height / 2);
        let right = x + width / 2;
        let bottom = y + height / 2;

        for dy in top..=bottom {
            for dx in left..=right {
                let is_contour = dx < left + thickness
                    || dx > right - thickness
                    || dy < top + thickness
                    || dy > bottom - thickness;

                if thickness == 0 || is_contour {
                    match &mut buf {
                        ImageBufferEnum::Rgb(buf) => {
                            buf.put_pixel(dx as u32, dy as u32, Rgb(rgb_color));
                        }
                        ImageBufferEnum::Luma(buf) => {
                            buf.put_pixel(dx as u32, dy as u32, Luma([rgb_color[0]]));
                        }
                    }
                }
            }
        }

        // 将ImageBuffer转回Tensor
        let new_tensor = match buf {
            ImageBufferEnum::Rgb(buf) => buf.to_tensor()?,
            ImageBufferEnum::Luma(buf) => buf.to_tensor()?,
        };
        Ok(new_tensor)
    }
}
