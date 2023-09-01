use super::Tensor;
use crate::utils::traits::float::FloatTrait;
use crate::vision::ImageType;
use image::{DynamicImage, ImageBuffer, Pixel};

// TODO: unit test
impl Tensor {
    pub fn is_image(&self) -> Result<ImageType, String> {
        let dims = self.dims();
        if !(2..=3).contains(&dims) {
            return Err("图像张量应该仅有2或3个维度。".to_string());
        }
        let channels = if dims == 2 { 0 } else { self.shape()[2] };
        let image_type = match channels {
            0 | 1 => ImageType::SingleOrNoneChannel,
            3 => ImageType::RGB,
            4 => ImageType::RGBA,
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
        let img = self.to_image_buff()?;
        Ok(DynamicImage::ImageRgb8(img))
    }

    /// 将张量转换为Image库的`ImageBuffer`格式
    pub fn to_image_buff<P>(&self) -> Result<ImageBuffer<P, Vec<u8>>, String>
    where
        P: Pixel<Subpixel = u8> + 'static,
    {
        let image_type = self.is_image()?;
        let shape = self.shape();
        let height = shape[0];
        let width = shape[1];

        let view = self.view();
        match image_type {
            ImageType::SingleOrNoneChannel => {
                let mut imgbuf = ImageBuffer::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        let pixel = view[[y, x]] as u8;
                        let px = P::from_channels(pixel, pixel, pixel, 0); // 传入相同的值到RGB通道
                        imgbuf.put_pixel(x as u32, y as u32, px);
                    }
                }
                Ok(imgbuf)
            }
            ImageType::RGB => {
                let mut imgbuf = ImageBuffer::new(width as u32, height as u32);
                for y in 0..height {
                    for x in 0..width {
                        let r = view[[y, x, 0]] as u8;
                        let g = view[[y, x, 1]] as u8;
                        let b = view[[y, x, 2]] as u8;
                        imgbuf.put_pixel(x as u32, y as u32, P::from_channels(r, g, b, 0));
                    }
                }
                Ok(imgbuf)
            }
            ImageType::RGBA => todo!(),
        }
    }

    /// 确定是图像的情况下，返回该图像的高度和宽度
    pub fn get_image_size(&self) -> Result<(usize, usize), String> {
        let image_type = self.is_image()?;
        let shape = self.shape();
        let height = shape[0];
        let width = shape[1];
        match image_type {
            ImageType::SingleOrNoneChannel => Ok((height, width)),
            ImageType::RGB => Ok((height, width)),
            ImageType::RGBA => todo!(),
        }
    }
}
