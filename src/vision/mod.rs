/*
 * @Author       : 老董
 * @Date         : 2023-08-30 19:16:48
 * @LastEditors  : 老董
 * @LastEditTime : 2023-09-01 14:02:01
 * @Description  : 本模块提供计算机视觉相关的功能。
 *                 在本模块中，不严谨地说：
 *                 1. 所谓的image/图像是指RGB(A)格式的图像；
 *                 2. “灰度”（图）等同于英文中luma、luminance、grey、gray的概念。
 */

use crate::tensor::Tensor;
use crate::utils::traits::dynamic_image::TraitForDynamicImage;
use image::{GenericImageView, GrayImage, RgbImage};

#[cfg(test)]
mod tests;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ImageType {
    SingleOrNoneChannel, // 单通道或者只有高（行）、宽（列）2个维度的图像张量
    RGB,                 // 3通道的图像张量
    RGBA,                // 4通道的图像张量
}

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
        let shape = tensor.shape();
        let height = shape[0];
        let width = shape[1];

        let view = tensor.view();
        // TODO：use Tensor .to_image()
        match image_type {
            ImageType::SingleOrNoneChannel => {
                let mut imgbuf: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
                    GrayImage::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        let pixel = view[[y, x]] as u8;
                        imgbuf.put_pixel(x as u32, y as u32, image::Luma([pixel]));
                    }
                }
                imgbuf.save(file_path).map_err(|e| e.to_string())?;
            }
            ImageType::RGB => {
                let mut imgbuf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                    RgbImage::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        let r = view[[y, x, 0]] as u8;
                        let g = view[[y, x, 1]] as u8;
                        let b = view[[y, x, 2]] as u8;
                        imgbuf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
                    }
                }
                imgbuf.save(file_path).map_err(|e| e.to_string())?;
            }
            ImageType::RGBA => todo!(),
        }

        Ok(())
    }

    /// 确定是图像的情况下，返回该图像的灰度图， 否则返回错误信息
    /// * `input_tensor` - 输入张量
    ///
    /// 注：如果输入张量是单通道的图像张量，则直接返回该张量。
    /// 这里用英文`luma`指代“灰度”（图），也大致等价于`luminance`、`grey`、`gray`。
    pub fn to_luma(input_tensor: &Tensor) -> Result<Tensor, String> {
        match input_tensor.is_image() {
            Ok(t) => match t {
                ImageType::SingleOrNoneChannel => Ok(input_tensor.clone()), // todo: 暂时把所有单通道图像张量都当作是灰度图
                ImageType::RGB => {
                    let height = input_tensor.shape()[0];
                    let width = input_tensor.shape()[1];
                    let input_view = input_tensor.view();
                    let mut luma_data = Vec::new();
                    // 多通道的图像转化为灰度图，需压缩到单通道
                    for y in 0..height {
                        for x in 0..width {
                            let r = input_view[[y, x, 0]];
                            let g = input_view[[y, x, 1]];
                            let b = input_view[[y, x, 2]];
                            let luma = 0.299 * r + 0.587 * g + 0.114 * b;
                            luma_data.push(luma.round());
                        }
                    }
                    Ok(Tensor::new(&luma_data, &[height, width]))
                }
                ImageType::RGBA => todo!(),
            },
            Err(e) => Err(e),
        }
    }

    /// 调整图像大小
    /// * `image` - 原始图像
    /// * `width` - 调整后的宽度
    /// * `height` - 调整后的高度
    /// * `crop` - `true`则执行基于中心的裁剪，`false`则执行基于中心的缩放
    ///
    /// 这里特意用`resize`而不是`reshape`，只为强调其只会改变尺寸，而不会改变张量本身的维度。
    pub fn resize_image(
        image: &Tensor,
        height: usize,
        width: usize,
        crop: bool,
    ) -> Result<Tensor, String> {
        let image = image.to_image()?;
        if crop {
            assert!(
                height <= image.height() as usize,
                "裁剪图像：新高度必须小于原始高度。"
            );
            assert!(
                width <= image.width() as usize,
                "裁剪图像：新宽度必须小于原始宽度。"
            );
            image
                .crop_imm(
                    (image.width() - width as u32) / 2,
                    (image.height() - height as u32) / 2,
                    width as u32,
                    height as u32,
                )
                .to_tensor()
        } else {
            image
                .resize_exact(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::Triangle,
                )
                .to_tensor()
        }
    }
}

//     pub fn blur_image(&self, image: &DynamicImage) -> DynamicImage {
//         // 模糊图像
//     }

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
