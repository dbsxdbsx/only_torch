use crate::vision::Vision;
use image::ColorType;

fn temp_image_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("only_torch_{}_{}", std::process::id(), name))
        .to_string_lossy()
        .into_owned()
}

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓保存、载入↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_save_load_image() {
    test_load_save_color_image();
    test_load_save_luma_image();
}
fn test_load_save_color_image() {
    // 1.测试载入、保存本地的png图片
    let loaded_image = Vision::load_image("./assets/lenna.png").unwrap();
    assert_eq!(loaded_image.shape(), &[512, 512, 3]);
    assert_eq!(loaded_image.is_image().unwrap(), ColorType::Rgb8);
    // (再次保存载入检查一致性)
    let png_path = temp_image_path("lenna_copy.png");
    Vision::save_image(&loaded_image, &png_path).unwrap();
    let new_load_image = Vision::load_image(&png_path).unwrap();
    let _ = std::fs::remove_file(&png_path);
    assert_eq!(loaded_image, new_load_image);

    // 2.测试保存、载入为jpg图片
    let jpg_path = temp_image_path("lenna.jpg");
    Vision::save_image(&loaded_image, &jpg_path).unwrap();
    let loaded_image = Vision::load_image(&jpg_path).unwrap();
    let _ = std::fs::remove_file(&jpg_path);
    // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
    assert_eq!(loaded_image.shape(), &[512, 512, 3]);
    assert_eq!(loaded_image.is_image().unwrap(), ColorType::Rgb8);

    // TODO: 3.测试rgba
    // TODO: 4.测试lumaA
}
fn test_load_save_luma_image() {
    // 1.测试载入本地的png彩色图片，并转化为灰度图
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    assert_eq!(luma_image.shape(), &[512, 512]);
    assert_eq!(luma_image.is_image().unwrap(), ColorType::L8);
    // (再次保存载入检查一致性；写入临时目录，避免并发测试覆盖共享 fixture)
    let copy_path = temp_image_path("lenna_luma.png");
    Vision::save_image(&luma_image, &copy_path).unwrap();
    let loaded_image = Vision::load_image(&copy_path).unwrap();
    let _ = std::fs::remove_file(&copy_path);
    assert_eq!(luma_image, loaded_image);

    // 2.测试载入本地的jpg彩色图片，并转化为灰度图
    let image = Vision::load_image("./assets/lenna.jpg").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    assert_eq!(luma_image.shape(), &[512, 512]);
    assert_eq!(luma_image.is_image().unwrap(), ColorType::L8);
    // (再次保存载入检查一致性)
    let jpg_path = temp_image_path("lenna_luma.jpg");
    Vision::save_image(&luma_image, &jpg_path).unwrap();
    let loaded_image = Vision::load_image(&jpg_path).unwrap();
    let _ = std::fs::remove_file(&jpg_path);
    // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
    assert_eq!(loaded_image.shape(), &[512, 512]);
    assert_eq!(loaded_image.is_image().unwrap(), ColorType::L8);

    // TODO: 3.测试rgba
    // TODO: 4.测试lumaA
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑保存、载入↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
