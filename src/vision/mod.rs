/*
 * @Author       : 老董
 * @Date         : 2023-08-30 19:16:48
 * @LastEditors  : 老董
 * @LastEditTime : 2023-09-04 09:01:04
 * @Description  : 本模块提供计算机视觉相关的功能。
 *                 在本模块中，不严谨地说：
 *                 1. 所谓的image/图像是指RGB(A)格式的图像；
 *                 2. “灰度”（图）等同于英文中luma、luminance、grey、gray的概念。
 */

use std::fs;

use crate::tensor::Tensor;
use crate::utils::traits::image::TraitForDynamicImage;
use image::{ColorType, DynamicImage, GenericImageView};

pub mod draw;
pub mod process;
pub mod shape;

#[cfg(test)]
mod tests;

pub struct Vision {
    // ...
}
impl Vision {
    // 将本地的图像加载到Tensor中
    pub fn load_image(path: &str) -> Result<Tensor, Box<dyn std::error::Error>> {
        let image = image::open(path)?;
        let (width, height) = image.dimensions();
        let channel_count = image.color().channel_count() as usize;
        let mut tensor_data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x, y);
                for c in 0..channel_count {
                    tensor_data.push(pixel[c] as f32);
                }
            }
        }

        let mut tensor = Tensor::new(
            &tensor_data,
            &[height as usize, width as usize, channel_count],
        );

        tensor.squeeze_mut();

        Ok(tensor)
    }

    /// 保存Tensor为图像到本地
    pub fn save_image(tensor: &Tensor, file_path: &str) -> Result<(), String> {
        let image_type = tensor.is_image()?;
        match image_type {
            ColorType::L8 => {
                let imgbuf = tensor.to_image_buff_for_luma8();
                if fs::metadata(file_path).is_ok() {
                    fs::remove_file(file_path).unwrap();
                }
                imgbuf.save(file_path).map_err(|e| e.to_string())?;
            }
            ColorType::Rgb8 => {
                let imgbuf = tensor.to_image_buff_for_rgb8();
                if fs::metadata(file_path).is_ok() {
                    fs::remove_file(file_path).unwrap();
                }
                imgbuf.save(file_path).map_err(|e| e.to_string())?;
            }
            _ => todo!(),
        }
        Ok(())
    }

    /// 确定是图像的情况下，返回该图像的灰度图， 否则返回错误信息
    /// * `tensor` - 输入张量
    ///
    /// 注：如果输入张量是单通道的图像张量，则直接返回该张量。
    /// 这里用英文`luma`指代“灰度”（图），也大致等价于`luminance`、`grey`、`gray`。
    pub fn to_luma(tensor: &Tensor) -> Result<Tensor, String> {
        let image = tensor.to_image()?;
        let luma = image.to_luma8();
        let image = DynamicImage::ImageLuma8(luma);
        image.to_tensor()
    }
}

// TODO:
//  show_image()

//     pub fn draw_circle(&self, image: &mut DynamicImage, x: u32, y: u32, radius: u32) {
//         // 在图像上绘制圆形
//     }

//     pub fn draw_rectangle(
//         &self,
//         image: &mut DynamicImage,
//         x: u32,
//         y: u32,
//         width: u32,
//         height: u32,
//     ) {
//         // 在图像上绘制矩形
//     }
