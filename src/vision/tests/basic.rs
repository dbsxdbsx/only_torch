use crate::vision::Vision;
use image::ColorType;

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
    Vision::save_image(&loaded_image, "./assets/lenna_copy.png").unwrap();
    let new_load_image = Vision::load_image("./assets/lenna_copy.png").unwrap();
    assert_eq!(loaded_image, new_load_image);

    // 2.测试保存、载入为jpg图片
    Vision::save_image(&loaded_image, "./assets/lenna.jpg").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna.jpg").unwrap();
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
    // (再次保存载入检查一致性)
    Vision::save_image(&luma_image, "./assets/lenna_luma.png").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna_luma.png").unwrap();
    assert_eq!(luma_image, loaded_image);

    // 2.测试载入本地的jpg彩色图片，并转化为灰度图
    let image = Vision::load_image("./assets/lenna.jpg").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    assert_eq!(luma_image.shape(), &[512, 512]);
    assert_eq!(luma_image.is_image().unwrap(), ColorType::L8);
    // (再次保存载入检查一致性)
    Vision::save_image(&luma_image, "./assets/lenna_luma.jpg").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna_luma.jpg").unwrap();
    // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
    assert_eq!(loaded_image.shape(), &[512, 512]);
    assert_eq!(loaded_image.is_image().unwrap(), ColorType::L8);

    // TODO: 3.测试rgba
    // TODO: 4.测试lumaA
}
//*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑保存、载入↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
