/*
 * @Author       : 老董
 * @Date         : 2023-09-05 10:00:57
 * @LastEditors  : 老董
 * @LastEditTime : 2023-09-08 08:20:53
 * @Description  : TODO：目前图像识别这块先放一放
 */
use super::{ImageBufferEnum, Vision};
use crate::tensor::Tensor;
use image::{ColorType, GrayImage, ImageBuffer, Luma};
use std::collections::HashMap;

impl Vision {
    pub fn detect_circles_by_hough(
        tensor: &Tensor,
        // dp: f32,
        min_dist: f32,
        param1: f32,
        param2: f32,
        min_radius: usize,
        max_radius: usize,
    ) -> Result<Vec<(usize, usize, usize)>, String> {
        // 检查是否为图像
        let image_type = tensor.is_image()?;
        let buf = match image_type {
            ColorType::L8 => ImageBufferEnum::Luma(tensor.to_image_buff_for_luma8()),
            ColorType::Rgb8 => ImageBufferEnum::Rgb(tensor.to_image_buff_for_rgb8()),
            _ => todo!(),
        };

        let gray_image = match buf {
            ImageBufferEnum::Luma(luma_image) => luma_image,
            ImageBufferEnum::Rgb(rgb_image) => {
                let (width, height) = rgb_image.dimensions();
                ImageBuffer::from_fn(width, height, |x, y| {
                    let pixel = rgb_image.get_pixel(x, y);
                    let luma = 0.114f32.mul_add(
                        f32::from(pixel[2]),
                        0.299f32.mul_add(f32::from(pixel[0]), 0.587 * f32::from(pixel[1])),
                    );
                    Luma([luma.round() as u8])
                })
            }
        };

        let circles = detect_circles_gray_image(
            &gray_image,
            // dp,
            min_dist,
            param1,
            param2,
            min_radius,
            max_radius,
        );
        Ok(circles)
    }
}

use num_integer::Roots;

fn detect_circles_gray_image(
    image: &GrayImage,
    // dp: f32,
    min_dist: f32,
    param1: f32,
    param2: f32,
    min_radius: usize,
    max_radius: usize,
) -> Vec<(usize, usize, usize)> {
    let width = image.width();
    let height = image.height();
    let mut accumulator: HashMap<(usize, usize, usize), usize> = HashMap::new();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let gradient_x = i32::from(image.get_pixel(x + 1, y).0[0])
                - i32::from(image.get_pixel(x - 1, y).0[0]);
            let gradient_y = i32::from(image.get_pixel(x, y + 1).0[0])
                - i32::from(image.get_pixel(x, y - 1).0[0]);
            let gradient_magnitude = (gradient_x.pow(2) + gradient_y.pow(2)).sqrt() as f32;

            if gradient_magnitude > param1 {
                let gradient_angle = (gradient_y as f32).atan2(gradient_x as f32);
                for radius in min_radius..=max_radius {
                    let a = (radius as f32)
                        .mul_add(gradient_angle.cos(), x as f32)
                        .round() as u32;
                    let b = (radius as f32)
                        .mul_add(gradient_angle.sin(), y as f32)
                        .round() as u32;

                    if a > 0 && a < width && b > 0 && b < height {
                        let counter = accumulator
                            .entry((a as usize, b as usize, radius))
                            .or_insert(0);
                        *counter += 1;
                    }
                }
            }
        }
    }

    let mut circles = Vec::new();
    for (&(a, b, radius), &votes) in &accumulator {
        if votes > param2 as usize {
            let mut is_valid = true;
            for &(x, y, _) in &circles {
                let distance =
                    ((x as i32 - a as i32).pow(2) + (y as i32 - b as i32).pow(2)).sqrt() as f32;
                if distance < min_dist {
                    is_valid = false;
                    break;
                }
            }
            if is_valid {
                circles.push((a, b, radius));
            }
        }
    }

    circles
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    // unit test for detecting circle
    fn test_detect_circles_by_hough() {
        let tensor = Vision::load_image("./assets/lenna.png").unwrap();
        let circles = Vision::detect_circles_by_hough(
            &tensor, // 1.0,
            20.0, 70.0, // 增加以过滤更多噪声
            20.0, // 增加以过滤更多噪声
            10,   // 调整以匹配圆的大小
            30,   // 调整以匹配圆的大小
        )
        .unwrap();
        println!("{:?}", circles);
    }
}
