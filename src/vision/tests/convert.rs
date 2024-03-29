use image::ColorType;

use crate::utils::traits::image::{ForDynamicImage, ForImageBuffer};
use crate::vision::Vision;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓DynamicImage<->Tensor↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_to_image_and_tensor_with_color_image() {
    // 1.用Image库的默认方式打开图片
    let img = image::open("./assets/lenna.png").unwrap();
    // 先转成Tensor验证下
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512, 3]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::Rgb8);
    // 再转成DynamicImage检查一致性
    let d_image = tensor.to_image().unwrap();
    assert_eq!(d_image.color(), image::ColorType::Rgb8);
    assert_eq!(d_image.get_channel_len(), 3);

    // 2.用Vision模块的方式打开图片
    let tensor = Vision::load_image("./assets/lenna.png").unwrap();
    // 先转成DynamicImage检查一致性
    let image = tensor.to_image().unwrap();
    assert_eq!(image.color(), image::ColorType::Rgb8);
    assert_eq!(d_image.get_channel_len(), 3);
    // 再转成Tensor验证下
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512, 3]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::Rgb8);
}

#[test]
fn test_to_image_and_tensor_with_luma_image() {
    // 1.用Image库的默认方式打开图片
    let img = image::open("./assets/lenna_luma.png").unwrap();
    // 先转成Tensor验证下
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::L8);
    // 再转成DynamicImage检查一致性
    let d_image = tensor.to_image().unwrap();
    assert_eq!(d_image.color(), image::ColorType::L8);
    assert_eq!(d_image.get_channel_len(), 1);

    // 2.用Vision模块的方式打开图片
    let tensor = Vision::load_image("./assets/lenna_luma.png").unwrap();
    // 先转成DynamicImage检查一致性
    let image = tensor.to_image().unwrap();
    assert_eq!(image.color(), image::ColorType::L8);
    assert_eq!(d_image.get_channel_len(), 1);
    // 再转成Tensor验证下
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::L8);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑DynamicImage<->Tensor↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ImageBuffer<->Tensor↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_to_tensor_with_color_image_buffer() {
    // 1.用Image库的默认方式打开图片
    let img1 = image::open("./assets/lenna.png").unwrap();
    let imgbuf1 = img1.to_rgb8();
    let t1 = imgbuf1.to_tensor().unwrap();
    // 2.用Vision模块的方式打开图片
    let t2 = Vision::load_image("./assets/lenna.png").unwrap();
    // 3.检查一致性
    assert_eq!(t1, t2);
    assert_eq!(t1.shape(), &[512, 512, 3]);
}
#[test]
fn test_to_tensor_with_luma_image_buffer() {
    // 1.用Image库的默认方式打开图片
    let img1 = image::open("./assets/lenna_luma.png").unwrap();
    let imgbuf1 = img1.to_luma8();
    let t1 = imgbuf1.to_tensor().unwrap();
    // 2.用Vision模块的方式打开图片
    let t2 = Vision::load_image("./assets/lenna_luma.png").unwrap();
    // 3.检查一致性
    assert_eq!(t1, t2);
    assert_eq!(t1.shape(), &[512, 512]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ImageBuffer<->Tensor↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
