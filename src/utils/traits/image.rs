use crate::tensor::Tensor;
use image::{DynamicImage, GenericImageView};

pub trait TraitForDynamicImage {
    fn get_channel_len(&self) -> usize;
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl TraitForDynamicImage for DynamicImage {
    fn get_channel_len(&self) -> usize {
        self.color().channel_count() as usize
    }
    fn to_tensor(&self) -> Result<Tensor, String> {
        let channels = self.get_channel_len();
        let width = self.dimensions().0 as usize;
        let height = self.dimensions().1 as usize;
        let mut tensor = Tensor::new_empty(&[height, width, channels]);

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel(x as u32, y as u32);
                let mut view = tensor.view_mut();
                view[[y, x, 0]] = pixel[0] as f32;
                if channels >= 3 {
                    view[[y, x, 1]] = pixel[1] as f32;
                    view[[y, x, 2]] = pixel[2] as f32;
                }
                if channels >= 4 {
                    view[[y, x, 3]] = pixel[3] as f32;
                }
            }
        }
        tensor.squeeze_mut();
        Ok(tensor)
    }
}

use image::{ImageBuffer, Pixel};

pub trait TraitForImageBuffer {
    fn to_tensor(&self) -> Result<Tensor, String>;
}

impl<P: Pixel + 'static> TraitForImageBuffer for ImageBuffer<P, Vec<P::Subpixel>> {
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
