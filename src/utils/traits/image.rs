use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView};

pub trait ForDynamicImage {
    fn get_channel_len(&self) -> usize;
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl ForDynamicImage for DynamicImage {
    fn get_channel_len(&self) -> usize {
        self.color().channel_count() as usize
    }
    fn to_tensor(&self) -> Result<Tensor, String> {
        let channels = self.get_channel_len();
        let width = self.dimensions().0 as usize;
        let height = self.dimensions().1 as usize;
        let mut tensor = Tensor::uninited(&[height, width, channels]);

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x as u32, y as u32);
                tensor[[y, x, 0]] = pixel[0] as f32;
                if channels >= 3 {
                    tensor[[y, x, 1]] = pixel[1] as f32;
                    tensor[[y, x, 2]] = pixel[2] as f32;
                }
                if channels >= 4 {
                    tensor[[y, x, 3]] = pixel[3] as f32;
                }
            }
        }
        tensor.squeeze_mut();
        Ok(tensor)
    }
}

use image::{ImageBuffer, Pixel};

pub trait ForImageBuffer {
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl<P: Pixel + 'static> ForImageBuffer for ImageBuffer<P, Vec<P::Subpixel>> {
    fn to_tensor(&self) -> Result<Tensor, String> {
        let width = self.width() as usize;
        let height = self.height() as usize;
        let channels = P::CHANNEL_COUNT as usize;

        let mut tensor_data = Vec::with_capacity(height * width * channels);
        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x as u32, y as u32);
                for c in 0..channels {
                    let value = pixel.channels()[c];
                    let sample = num_traits::NumCast::from(value).unwrap_or(0.0);
                    tensor_data.push(sample);
                }
            }
        }

        let mut tensor = Tensor::new(&tensor_data, &[height, width, channels]);
        tensor.squeeze_mut();
        Ok(tensor)
    }
}
