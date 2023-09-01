use crate::assert_panic;
use crate::utils::traits::dynamic_image::TraitForDynamicImage;
use crate::vision::{ImageType, Vision};

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓保存、载入↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_save_load_image() {
    test_load_save_color_image();
    test_load_save_luma_image();
}
fn test_load_save_color_image() {
    // 1.测试载入、保存本地的png图片
    let loaded_image = Vision::load_image("./assets/lenna.png").unwrap();
    assert_eq!(loaded_image.shape(), &[512, 512, 3]);
    assert_eq!(loaded_image.is_image().unwrap(), ImageType::RGB);
    // (再次保存载入检查一致性)
    Vision::save_image(&loaded_image, "./assets/lenna_copy.png").unwrap();
    let new_load_image = Vision::load_image("./assets/lenna_copy.png").unwrap();
    assert_eq!(loaded_image, new_load_image);

    // 2.测试保存、载入为jpg图片
    Vision::save_image(&loaded_image, "./assets/lenna.jpg").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna.jpg").unwrap();
    // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
    assert_eq!(loaded_image.shape(), &[512, 512, 3]);
    assert_eq!(loaded_image.is_image().unwrap(), ImageType::RGB);

    // TODO: 3.测试rgba
    // TODO: 4.测试lumaA
}
fn test_load_save_luma_image() {
    // 1.测试载入本地的png彩色图片，并转化为灰度图
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    assert_eq!(luma_image.shape(), &[512, 512]);
    assert_eq!(
        luma_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );
    // (再次保存载入检查一致性)
    Vision::save_image(&luma_image, "./assets/lenna_luma.png").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna_luma.png").unwrap();
    assert_eq!(luma_image, loaded_image);

    // 2.测试载入本地的jpg彩色图片，并转化为灰度图
    let image = Vision::load_image("./assets/lenna.jpg").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    assert_eq!(luma_image.shape(), &[512, 512]);
    assert_eq!(
        luma_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );
    // (再次保存载入检查一致性)
    Vision::save_image(&luma_image, "./assets/lenna_luma.jpg").unwrap();
    let loaded_image = Vision::load_image("./assets/lenna_luma.jpg").unwrap();
    // (由于jpg是有损压损，故只检查形状，不检查数据一致性)
    assert_eq!(loaded_image.shape(), &[512, 512]);
    assert_eq!(
        loaded_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );

    // TODO: 3.测试rgba
    // TODO: 4.测试lumaA
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑保存、载入↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓改变图像尺寸↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
#[test]
fn test_resize_color_image() {
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    // 1.1测试不保留宽高比的图像尺寸调整（高宽均小于原始尺寸）
    let height = 200;
    let width = 256;
    let resized_image = Vision::resize_image(&image, height, width, true).unwrap();
    assert_eq!(resized_image.shape(), &[height, width, 3]);
    assert_eq!(resized_image.is_image().unwrap(), ImageType::RGB);
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
    assert_eq!(resized_image.is_image().unwrap(), ImageType::RGB);
    Vision::save_image(&resized_image, "./assets/lenna_resized_shrink.png").unwrap();
    // 2.2测试保留宽高比的图像尺寸调整：扩大
    let (mut height, mut width) = image.get_image_size().unwrap();
    height *= 1.5 as usize;
    width *= 1.5 as usize;
    let resized_image = Vision::resize_image(&image, height, width, false).unwrap();
    assert_eq!(resized_image.shape(), &[height, width, 3]);
    assert_eq!(resized_image.is_image().unwrap(), ImageType::RGB);
    Vision::save_image(&resized_image, "./assets/lenna_resized_expand.png").unwrap();
}

#[test]
fn test_resize_luma_image() {
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    let luma_image = Vision::to_luma(&image).unwrap();
    // 1.1测试不保留宽高比的图像尺寸调整（高宽均小于原始尺寸）
    let height = 200;
    let width = 256;
    let resized_image = Vision::resize_image(&luma_image, height, width, true).unwrap();
    let _shape = resized_image.shape();
    assert_eq!(
        resized_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );
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
    assert_eq!(
        resized_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );
    Vision::save_image(&resized_image, "./assets/lenna_luma_resized_shrink.png").unwrap();
    // 2.2测试保留宽高比的图像尺寸调整：扩大
    let (mut height, mut width) = image.get_image_size().unwrap();
    height *= 1.5 as usize;
    width *= 1.5 as usize;
    let resized_image = Vision::resize_image(&image, height, width, false).unwrap();
    assert_eq!(resized_image.shape(), &[height, width]);
    assert_eq!(
        resized_image.is_image().unwrap(),
        ImageType::SingleOrNoneChannel
    );
    Vision::save_image(&resized_image, "./assets/lenna_resized_expand.png").unwrap();
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑改变图像尺寸↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

#[test]
fn test_to_tensor() {
    // 测试彩色图像
    let img = image::open("./assets/lenna.png").unwrap();
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512, 3]);
    assert_eq!(tensor.dims(), 3);
    // 测试灰度图像
    // xxxx
    let img = image::open("./assets/lenna_luma.png").unwrap();
    let tensor = img.to_tensor().unwrap();
    assert_eq!(tensor.shape(), &[512, 512]);
    assert_eq!(tensor.dims(), 3);
}
