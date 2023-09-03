/*
 * @Author       : 老董
 * @Date         : 2023-08-30 19:16:48
 * @LastEditors  : 老董
 * @LastEditTime : 2023-09-03 13:40:30
 * @Description  : 本模块提供计算机视觉相关的功能。
 *                 在本模块中，不严谨地说：
 *                 1. 所谓的image/图像是指RGB(A)格式的图像；
 *                 2. “灰度”（图）等同于英文中luma、luminance、grey、gray的概念。
 */

use std::fs;

use crate::tensor::Tensor;
use crate::utils::traits::dynamic_image::TraitForDynamicImage;
use image::{ColorType, DynamicImage, GenericImageView};

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
        if fs::metadata(file_path).is_ok() {
            fs::remove_file(file_path).unwrap();
        }
        let image_type = tensor.is_image()?;
        match image_type {
            ColorType::L8 => {
                let imgbuf = tensor.to_image_buff_for_luma8();
                imgbuf.save(file_path).map_err(|e| e.to_string())?;
            }
            ColorType::Rgb8 => {
                let imgbuf = tensor.to_image_buff_for_rgb8();
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
        // way1: 转成DynamicImage后调用自身的luma算法，再转回Tensor
        let image = tensor.to_image()?;
        let luma = image.to_luma8();
        // convert to tensor
        let image = DynamicImage::ImageLuma8(luma);
        image.to_tensor()
        // way2: 直接在Tensor上运算
        // match input_tensor.is_image() {
        //     Ok(t) => match t {
        //         ColorType::SingleOrNoneChannel => Ok(input_tensor.clone()), // todo: 暂时把所有单通道图像张量都当作是灰度图
        //         ColorType::RGB => {
        //             let height = input_tensor.shape()[0];
        //             let width = input_tensor.shape()[1];
        //             let input_view = input_tensor.view();
        //             let mut luma_data = Vec::new();
        //             // 多通道的图像转化为灰度图，需压缩到单通道
        //             for y in 0..height {
        //                 for x in 0..width {
        //                     let r = input_view[[y, x, 0]];
        //                     let g = input_view[[y, x, 1]];
        //                     let b = input_view[[y, x, 2]];
        //                     let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        //                     luma_data.push(luma.round());
        //                 }
        //             }
        //             Ok(Tensor::new(&luma_data, &[height, width]))
        //         }
        //         _ => todo!(),
        //     },
        //     Err(e) => Err(e),
        // }
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

    pub fn median_blur(image: &Tensor, ksize: usize) -> Tensor {
        assert!(ksize >= 2);
        let mut blurred_tensor = image.clone();
        let mut blurred = blurred_tensor.view_mut();

        let (h, w, c) = image.get_image_shape().unwrap();
        let half_ksize = ksize / 2;
        let orig_view = image.view();

        for y in half_ksize..h - half_ksize {
            for x in half_ksize..w - half_ksize {
                if c == 0 {
                    let mut values = Vec::with_capacity(ksize * ksize);
                    for ky in y - half_ksize..y + half_ksize + 1 {
                        for kx in x - half_ksize..x + half_ksize + 1 {
                            values.push(orig_view[[ky, kx]]);
                        }
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = values[values.len() / 2];

                    blurred[[y, x]] = median;
                } else {
                    for z in 0..c {
                        let mut values = Vec::with_capacity(ksize * ksize);
                        for ky in y - half_ksize..y + half_ksize + 1 {
                            for kx in x - half_ksize..x + half_ksize + 1 {
                                values.push(orig_view[[ky, kx, z]]);
                            }
                        }
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = values[values.len() / 2];

                        blurred[[y, x, z]] = median;
                    }
                }
            }
        }

        blurred_tensor
    }
}
// TODO:
//  show_image()

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
