use image::ColorType;

use crate::utils::traits::image::{ForDynamicImage, ForImageBuffer};
use crate::vision::io::load_image;

#[test]
fn test_to_image_and_tensor_with_color_image() {
    // 用 image crate 默认方式打开
    let img = image::open("./assets/lenna.png").unwrap();
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512, 3]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::Rgb8);
    let d_image = tensor.to_image().unwrap();
    assert_eq!(d_image.color(), image::ColorType::Rgb8);
    assert_eq!(d_image.get_channel_len(), 3);

    // 用 vision::io 方式打开应得到相同结果
    let tensor2 = load_image("./assets/lenna.png")
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_eq!(tensor, tensor2);
}

#[test]
fn test_to_image_and_tensor_with_luma_image() {
    let img = image::open("./assets/lenna_luma.png").unwrap();
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::L8);
    let d_image = tensor.to_image().unwrap();
    assert_eq!(d_image.color(), image::ColorType::L8);
    assert_eq!(d_image.get_channel_len(), 1);
}

#[test]
fn test_to_tensor_with_color_image_buffer() {
    let img = image::open("./assets/lenna.png").unwrap();
    let imgbuf = img.to_rgb8();
    let t1 = imgbuf.to_tensor().unwrap();
    let t2 = load_image("./assets/lenna.png")
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_eq!(t1, t2);
    assert_eq!(t1.shape(), &[512, 512, 3]);
}

#[test]
fn test_to_tensor_with_luma_image_buffer() {
    let img = image::open("./assets/lenna_luma.png").unwrap();
    let imgbuf = img.to_luma8();
    let t1 = imgbuf.to_tensor().unwrap();
    let t2 = load_image("./assets/lenna_luma.png")
        .unwrap()
        .to_tensor()
        .unwrap();
    assert_eq!(t1, t2);
    assert_eq!(t1.shape(), &[512, 512]);
}
