/*
 * detect 模块单元测试（图像检测功能）
 */

use crate::vision::Vision;

#[test]
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
