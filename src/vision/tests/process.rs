use crate::utils::tests::get_file_size_in_byte;
use crate::vision::Vision;

#[test]
fn test_median_blur() {
    // 测试彩色图像
    let image = Vision::load_image("./assets/lenna.png").unwrap();
    // println!("{:.2} MB", file_size as f64 / (1024.0 * 1024.0));
    let kernel_sizes = [3, 5, 7];
    let mut former_size = get_file_size_in_byte("./assets/lenna.png");
    for ksize in &kernel_sizes {
        let blurred_image = Vision::median_blur(&image, *ksize);
        let filename = format!("./assets/lenna_blurred_{}.png", ksize);
        Vision::save_image(&blurred_image, &filename).unwrap();
        let new_size = get_file_size_in_byte(filename);
        assert!(new_size <= former_size);
        former_size = new_size;
        // print in mb
        println!("{:.2} MB", new_size as f64 / (1024.0 * 1024.0));
    }
    // 测试灰度图像
    let image = Vision::load_image("./assets/lenna_luma.png").unwrap();
    let kernel_sizes = [3, 5, 7];
    let mut former_size = get_file_size_in_byte("./assets/lenna.png");
    for ksize in &kernel_sizes {
        let blurred_image = Vision::median_blur(&image, *ksize);
        let filename = format!("./assets/lenna_blurred_{}.png", ksize);
        Vision::save_image(&blurred_image, &filename).unwrap();
        let new_size = get_file_size_in_byte(filename);
        assert!(new_size <= former_size);
        former_size = new_size;
        // print in mb
        println!("{:.2} MB", new_size as f64 / (1024.0 * 1024.0));
    }
}
