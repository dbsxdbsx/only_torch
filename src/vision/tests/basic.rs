use image::ColorType;

use crate::utils::traits::image::ForDynamicImage;
use crate::vision::color::to_luma;
use crate::vision::io::{load_image, save_image};

fn temp_image_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("only_torch_{}_{}", std::process::id(), name))
        .to_string_lossy()
        .into_owned()
}

#[test]
fn test_save_load_image() {
    test_load_save_color_image();
    test_load_save_luma_image();
}

fn test_load_save_color_image() {
    // 1.加载本地 PNG，验证基础元信息
    let img = load_image("./assets/lenna.png").unwrap();
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512, 3]);
    assert_eq!(tensor.is_image().unwrap(), ColorType::Rgb8);

    // 2.再次保存载入检查 PNG 一致性
    let png_path = temp_image_path("lenna_copy.png");
    save_image(&img, &png_path).unwrap();
    let loaded = load_image(&png_path).unwrap();
    let _ = std::fs::remove_file(&png_path);
    assert_eq!(loaded.to_rgb8(), img.to_rgb8());

    // 3.保存为 JPG（有损）后，只检查能成功重新载入与基本元信息
    let jpg_path = temp_image_path("lenna.jpg");
    save_image(&img, &jpg_path).unwrap();
    let loaded_jpg = load_image(&jpg_path).unwrap();
    let _ = std::fs::remove_file(&jpg_path);
    let jpg_tensor = loaded_jpg.to_tensor().unwrap();
    assert_eq!(jpg_tensor.shape(), &[512, 512, 3]);
    assert_eq!(jpg_tensor.is_image().unwrap(), ColorType::Rgb8);
}

fn test_load_save_luma_image() {
    // 1.PNG → Luma 转换 + roundtrip
    let img = load_image("./assets/lenna.png").unwrap();
    let luma = to_luma(&img);
    let luma_tensor = luma.to_tensor().unwrap();
    assert_eq!(luma_tensor.shape(), &[512, 512]);
    assert_eq!(luma_tensor.is_image().unwrap(), ColorType::L8);

    let copy_path = temp_image_path("lenna_luma.png");
    save_image(&luma, &copy_path).unwrap();
    let loaded = load_image(&copy_path).unwrap();
    let _ = std::fs::remove_file(&copy_path);
    assert_eq!(loaded.to_luma8(), luma.to_luma8());

    // 2.JPG → Luma 转换（有损，仅检查元信息）
    let jpg_img = load_image("./assets/lenna.jpg").unwrap();
    let jpg_luma = to_luma(&jpg_img);
    let jpg_luma_tensor = jpg_luma.to_tensor().unwrap();
    assert_eq!(jpg_luma_tensor.shape(), &[512, 512]);
    assert_eq!(jpg_luma_tensor.is_image().unwrap(), ColorType::L8);
}
