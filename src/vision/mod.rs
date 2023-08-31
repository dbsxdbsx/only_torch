use crate::tensor::Tensor;
use crate::utils::dynamic_image_trait::TensorTraitForDynamicImage;
use image::{GenericImageView, GrayImage, RgbImage};

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

    pub fn to_luma(input_tensor: &Tensor) -> Result<Tensor, String> {
        match input_tensor.is_image() {
            Ok(t) => match t {
                ImageType::SingleOrNoneChannel => Ok(input_tensor.clone()), // todo: 暂时把所有单通道图像张量都当作是灰度图
                ImageType::RGB => {
                    let height = input_tensor.shape()[0];
                    let width = input_tensor.shape()[1];
                    let input_view = input_tensor.view();
                    let mut luma_data = Vec::new();
                    // 多通道的图像转化为灰度图，需依据压缩到单通道
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

    // TODO：reshape
    /// 调整图像大小
    /// * `image` - 原始图像
    /// * `width` - 调整后的宽度
    /// * `height` - 调整后的高度
    /// * `preserve_aspect_ratio` - 是否保持原始宽高比
    pub fn resize_image(
        image: &Tensor,
        width: u32,
        height: u32,
        preserve_aspect_ratio: bool,
    ) -> Result<Tensor, String> {
        let image = image.to_image()?;
        if preserve_aspect_ratio {
            // TODO: 需要另外计算高宽？
            // 计算调整后的大小,保持原始宽高比
            let (width, height) =
                Self::get_resized_dims(image.width(), image.height(), width, height);
            image
                .resize_exact(width, height, image::imageops::FilterType::Triangle)
                .to_tensor()
        } else {
            // 不保持原始宽高比,强制调整为指定大小
            image
                .resize(width, height, image::imageops::FilterType::Triangle)
                .to_tensor()
        }
    }

    /// 计算保持原始宽高比的调整后大小
    fn get_resized_dims(
        original_width: u32,
        original_height: u32,
        width: u32,
        height: u32,
    ) -> (u32, u32) {
        let original_ratio = original_width as f32 / original_height as f32;

        if original_ratio > width as f32 / height as f32 {
            // 根据高度调整宽度
            let resized_width = (height as f32 * original_ratio) as u32;
            (resized_width, height)
        } else {
            // 根据宽度调整高度
            let resized_height = (width as f32 / original_ratio) as u32;
            (width, resized_height)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_image() {
        test_load_save_color_image();
        test_load_save_luma_image();
    }

    fn test_load_save_color_image() {
        // 1.测试载入、保存本地的png图片
        let loaded_image = Vision::load_image("./assets/lenna.png").unwrap();
        assert_eq!(loaded_image.shape(), &[512, 512, 3]);
        assert_eq!(loaded_image.is_image().unwrap(), ImageType::RGB);
        // (再次保存载入检查一致性)
        Vision::save_image(&loaded_image, "./assets/lenna_copy.png").unwrap();
        let new_load_image = Vision::load_image("./assets/lenna_copy.png").unwrap();
        assert_eq!(loaded_image, new_load_image);

        // 2.测试保存、载入为jpg图片
        Vision::save_image(&loaded_image, "./assets/lenna.jpg").unwrap();
        let loaded_image = Vision::load_image("./assets/lenna.jpg").unwrap();
        // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
        assert_eq!(loaded_image.shape(), &[512, 512, 3]);
        assert_eq!(loaded_image.is_image().unwrap(), ImageType::RGB);

        // TODO: 3.测试rgba
        // TODO: 4.测试lumaA
    }

    fn test_load_save_luma_image() {
        // 1.测试载入本地的png彩色图片，并转化为灰度图
        let image = Vision::load_image("./assets/lenna.png").unwrap();
        let luma_image = Vision::to_luma(&image).unwrap();
        assert_eq!(luma_image.shape(), &[512, 512]);
        assert_eq!(
            luma_image.is_image().unwrap(),
            ImageType::SingleOrNoneChannel
        );
        // (再次保存载入检查一致性)
        Vision::save_image(&luma_image, "./assets/lenna_luma.png").unwrap();
        let loaded_image = Vision::load_image("./assets/lenna_luma.png").unwrap();
        assert_eq!(luma_image, loaded_image);

        // 2.测试载入本地的jpg彩色图片，并转化为灰度图
        let image = Vision::load_image("./assets/lenna.jpg").unwrap();
        let luma_image = Vision::to_luma(&image).unwrap();
        assert_eq!(luma_image.shape(), &[512, 512]);
        assert_eq!(
            luma_image.is_image().unwrap(),
            ImageType::SingleOrNoneChannel
        );
        // (再次保存载入检查一致性)
        Vision::save_image(&luma_image, "./assets/lenna_luma.jpg").unwrap();
        let loaded_image = Vision::load_image("./assets/lenna_luma.jpg").unwrap();
        // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
        assert_eq!(loaded_image.shape(), &[512, 512]);
        assert_eq!(
            loaded_image.is_image().unwrap(),
            ImageType::SingleOrNoneChannel
        );

        // TODO: 3.测试rgba
        // TODO: 4.测试lumaA
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
