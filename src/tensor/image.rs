use super::Tensor;
use crate::utils::traits::float::FloatTrait;
use crate::vision::Vision;
use image::{ColorType, DynamicImage, GrayImage, ImageBuffer, RgbImage};

impl Tensor {
    pub fn is_image(&self) -> Result<ColorType, String> {
        let dims = self.dimension();
        if !(2..=3).contains(&dims) {
            return Err("图像张量应该仅有2或3个维度。".to_string());
        }
        let channels = if dims == 2 { 0 } else { self.shape()[2] };
        // TODO: 目前的判断仍不严谨
        let image_type = match channels {
            0 | 1 => ColorType::L8,
            3 => ColorType::Rgb8,
            4 => ColorType::Rgba8,
            _ => return Err("图像张量的通道数只可能是0、1、3或4。".to_string()),
        };

        let data = self.data.as_slice().unwrap();
        for pixel in data {
            // 检查每个像素值是否在[0,255]闭区间内
            if *pixel < 0.0 || *pixel > 255.0 {
                return Err(format!(
                    "检测到像素值{pixel}：图像张量的每个像素值必须在[0,255]之间。"
                ));
            }
            // 确保每个像素都是整数，即使它的类型是浮点数
            if !pixel.is_integer() {
                return Err(format!(
                    "检测到像素值{pixel}：图像张量的每个像素值必须没有小数。"
                ));
            }
        }

        Ok(image_type)
    }

    /// 将张量转换为Image库的`DynamicImage`格式
    pub fn to_image(&self) -> Result<DynamicImage, String> {
        let image_type = self.is_image()?;
        match image_type {
            ColorType::L8 => Ok(DynamicImage::ImageLuma8(self.to_image_buff_for_luma8())),
            ColorType::Rgb8 => Ok(DynamicImage::ImageRgb8(self.to_image_buff_for_rgb8())),
            _ => todo!(),
        }
    }

    pub fn to_image_buff_for_rgb8(&self) -> ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let shape = self.shape();
        let height = shape[0];
        let width = shape[1];

        let mut imgbuf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            RgbImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let r = self[[y, x, 0]] as u8;
                let g = self[[y, x, 1]] as u8;
                let b = self[[y, x, 2]] as u8;
                imgbuf.put_pixel(x as u32, y as u32, image::Rgb([r, g, b]));
            }
        }
        imgbuf
    }
    pub fn to_image_buff_for_luma8(&self) -> ImageBuffer<image::Luma<u8>, Vec<u8>> {
        let shape = self.shape();
        let height = shape[0];
        let width = shape[1];

        let mut imgbuf: image::ImageBuffer<image::Luma<u8>, Vec<u8>> =
            GrayImage::new(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let pixel = self[[y, x]] as u8;
                imgbuf.put_pixel(x as u32, y as u32, image::Luma([pixel]));
            }
        }
        imgbuf
    }

    /// 确定是图像的情况下，返回该图像的高度和宽度（不含通道数）
    pub fn get_image_size(&self) -> Result<(usize, usize), String> {
        self.get_image_shape().map(|(h, w, _)| (h, w))
    }

    /// 确定是图像的情况下，返回该图像的高度和宽度和通道数（没有则为0）
    pub fn get_image_shape(&self) -> Result<(usize, usize, usize), String> {
        let _ = self.is_image()?;
        let shape = self.shape();
        match self.dimension() {
            2 => Ok((shape[0], shape[1], 0)),
            3 => Ok((shape[0], shape[1], shape[2])),
            _ => Err("图像张量应该仅有2或3个维度。".to_string()),
        }
    }
}

// 这里是一些可以直接用于Tensor实例的Vision静态方法
impl Tensor {
    pub fn to_luma(&self) -> Result<Self, String> {
        Vision::to_luma(self)
    }

    pub fn to_luma_mut(&mut self) {
        *self = Vision::to_luma(self).unwrap();
    }
}
