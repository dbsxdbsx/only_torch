use image::{ColorType, ImageBuffer, Luma, Rgb};

use crate::tensor::Tensor;
use crate::utils::traits::image::TraitForImageBuffer;

use super::Vision;

enum ImageBufferEnum {
    Rgb(ImageBuffer<image::Rgb<u8>, Vec<u8>>),
    Luma(ImageBuffer<image::Luma<u8>, Vec<u8>>),
}

impl Vision {
    pub fn draw_circle(
        tensor: &Tensor,
        center: (usize, usize),
        radius: usize,
        rgb_color: [u8; 3],
        contour_thickness: usize,
    ) -> Result<Tensor, String> {
        // 检查是否为图像
        let image_type = tensor.is_image()?;
        let mut buf = match image_type {
            ColorType::L8 => ImageBufferEnum::Luma(tensor.to_image_buff_for_luma8()),
            ColorType::Rgb8 => ImageBufferEnum::Rgb(tensor.to_image_buff_for_rgb8()),
            _ => todo!(),
        };
        let (x, y) = center;
        let inner_radius = radius.saturating_sub(contour_thickness);

        for dy in 0..radius * 2 {
            for dx in 0..radius * 2 {
                let xt = x as i32 + dx as i32 - radius as i32;
                let yt = y as i32 + dy as i32 - radius as i32;
                let distance_squared = (xt - x as i32).pow(2) + (yt - y as i32).pow(2);

                if distance_squared <= radius.pow(2) as i32
                    && (contour_thickness == 0 || distance_squared > inner_radius.pow(2) as i32)
                {
                    match &mut buf {
                        ImageBufferEnum::Rgb(buf) => {
                            buf.put_pixel(xt as u32, yt as u32, Rgb(rgb_color))
                        }
                        ImageBufferEnum::Luma(buf) => {
                            buf.put_pixel(xt as u32, yt as u32, Luma([rgb_color[0]]))
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

// impl Vision {
//     pub fn draw_circle(
//         tensor: &Tensor,
//         center: (usize, usize),
//         radius: usize,
//         rgb_color: [u8; 3],
//         filled: bool,
//     ) -> Result<Tensor, String> {
//         // 检查是否为图像
//         let image_type = tensor.is_image()?;
//         let mut buf = match image_type {
//             ColorType::L8 => ImageBufferEnum::Luma(tensor.to_image_buff_for_luma8()),
//             ColorType::Rgb8 => ImageBufferEnum::Rgb(tensor.to_image_buff_for_rgb8()),
//             _ => todo!(),
//         };
//         let (x, y) = center;
//         // 在Tensor上绘制圆形
//         match filled {
//             // 填充绘制
//             true => {
//                 for dy in 0..radius * 2 {
//                     for dx in 0..radius * 2 {
//                         let xt = x as i32 + dx as i32 - radius as i32;
//                         let yt = y as i32 + dy as i32 - radius as i32;
//                         if (xt - x as i32).pow(2) + (yt - y as i32).pow(2) <= radius.pow(2) as i32 {
//                             match &mut buf {
//                                 ImageBufferEnum::Rgb(buf) => {
//                                     buf.put_pixel(xt as u32, yt as u32, Rgb(rgb_color))
//                                 }
//                                 ImageBufferEnum::Luma(buf) => {
//                                     buf.put_pixel(xt as u32, yt as u32, Luma([rgb_color[0]]))
//                                 }
//                             }
//                         }
//                     }
//                 }
//             }
//             false => {
//                 // 只画轮廓

//             }
//         }

//         // 将ImageBuffer转回Tensor
//         let new_tensor = match buf {
//             ImageBufferEnum::Rgb(buf) => buf.to_tensor()?,
//             ImageBufferEnum::Luma(buf) => buf.to_tensor()?,
//         };
//         Ok(new_tensor)
//     }
// }
