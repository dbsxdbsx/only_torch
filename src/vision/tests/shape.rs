use image::ColorType;

use crate::assert_panic;
use crate::vision::Vision;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓改变图像尺寸↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_resize_color_image() {
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    // 1.1测试不保留宽高比的图像尺寸调整（高宽均小于原始尺寸）
    let height = 200;
    let width = 256;
    let resized_image = Vision::resize_image(&image, height, width, true).unwrap();
    assert_eq!(resized_image.shape(), &[height, width, 3]);
    assert_eq!(resized_image.is_image().unwrap(), ColorType::Rgb8);
    Vision::save_image(&resized_image, "./assets/lenna_resized_crop.png").unwrap();
    // 1.2测试不保留宽高比的图像尺寸调整（高大于原始尺寸，而宽小于原始尺寸）
    let height = 600;
    let width = 256;
    assert_panic!(Vision::resize_image(&image, height, width, true));
    // 1.3测试不保留宽高比的图像尺寸调整（高小于原始尺寸，而宽大于原始尺寸）
    let height = 256;
    let width = 600;
    assert_panic!(Vision::resize_image(&image, height, width, true));

    // 2.1测试保留宽高比的图像尺寸调整：缩小
    let resized_image = Vision::resize_image(&image, height, width, false).unwrap();
    assert_eq!(resized_image.shape(), &[height, width, 3]);
    assert_eq!(resized_image.is_image().unwrap(), ColorType::Rgb8);
    Vision::save_image(&resized_image, "./assets/lenna_resized_shrink.png").unwrap();
    // 2.2测试保留宽高比的图像尺寸调整：扩大
    let (mut height, mut width) = image.get_image_size().unwrap();
    height *= 1.5 as usize;
    width *= 1.5 as usize;
    let resized_image = Vision::resize_image(&image, height, width, false).unwrap();
    assert_eq!(resized_image.shape(), &[height, width, 3]);
    assert_eq!(resized_image.is_image().unwrap(), ColorType::Rgb8);
    Vision::save_image(&resized_image, "./assets/lenna_resized_expand.png").unwrap();
}

#[test]
fn test_resize_luma_image() {
    let luma_image = Vision::load_image("./assets/lenna_luma.png").unwrap();
    // 1.1测试不保留宽高比的图像尺寸调整（高宽均小于原始尺寸）
    let height = 200;
    let width = 256;
    let resized_image = Vision::resize_image(&luma_image, height, width, true).unwrap();
    let _shape = resized_image.shape();
    assert_eq!(resized_image.is_image().unwrap(), ColorType::L8);
    Vision::save_image(&resized_image, "./assets/lenna_luma_resized_crop.png").unwrap();
    // 1.2测试不保留宽高比的图像尺寸调整（高大于原始尺寸，而宽小于原始尺寸）
    let height = 600;
    let width = 256;
    assert_panic!(Vision::resize_image(&luma_image, height, width, true));
    // 1.3测试不保留宽高比的图像尺寸调整（高小于原始尺寸，而宽大于原始尺寸）
    let height = 256;
    let width = 600;
    assert_panic!(Vision::resize_image(&luma_image, height, width, true));

    // 2.1测试保留宽高比的图像尺寸调整：缩小
    let resized_image = Vision::resize_image(&luma_image, height, width, false).unwrap();
    assert_eq!(resized_image.is_image().unwrap(), ColorType::L8);
    Vision::save_image(&resized_image, "./assets/lenna_luma_resized_shrink.png").unwrap();
    // 2.2测试保留宽高比的图像尺寸调整：扩大
    let (mut height, mut width) = luma_image.get_image_size().unwrap();
    height *= 1.5 as usize;
    width *= 1.5 as usize;
    let resized_image = Vision::resize_image(&luma_image, height, width, false).unwrap();
    assert_eq!(resized_image.shape(), &[height, width]);
    assert_eq!(resized_image.is_image().unwrap(), ColorType::L8);
    Vision::save_image(&resized_image, "./assets/lenna_resized_expand.png").unwrap();
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑改变图像尺寸↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
