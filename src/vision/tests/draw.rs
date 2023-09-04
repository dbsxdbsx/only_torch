use crate::vision::Vision;

#[test]
fn test_draw_circle() {
    // 1.测试彩色图片
    let tensor = Vision::load_image("./assets/lenna.png").unwrap();
    let (h, w) = tensor.get_image_size().unwrap();
    let x = w / 2;
    let y = h / 2;
    let radius = if x < y { x } else { y };
    let thickness = [0, 1, 3, radius / 3, radius / 2, radius, radius + 1];
    for t in thickness.iter() {
        let new_tensor = Vision::draw_circle(&tensor, (x, y), radius, [255, 100, 255], *t).unwrap();
        Vision::save_image(&new_tensor, &format!("./assets/lenna_circle_{}.png", t)).unwrap();
    }
    // 2.测试灰度图片
    let tensor = Vision::load_image("./assets/lenna_luma.png").unwrap();
    let (h, w) = tensor.get_image_size().unwrap();
    let x = w / 2;
    let y = h / 2;
    let radius = if x < y { x } else { y };
    let thickness = [0, 1, 3, radius / 3, radius / 2, radius, radius + 1];
    for t in thickness.iter() {
        let new_tensor = Vision::draw_circle(&tensor, (x, y), radius, [250, 0, 0], *t).unwrap();
        Vision::save_image(
            &new_tensor,
            &format!("./assets/lenna_luma_circle_{}.png", t),
        )
        .unwrap();
    }
}

// write test for draw_rectangle
#[test]
fn test_draw_rectangle() {
    // 1.测试彩色图片
    let tensor = Vision::load_image("./assets/lenna.png").unwrap();
    let (h, w) = tensor.get_image_size().unwrap();
    let x = w / 2;
    let y = h / 2;
    let width = w / 2;
    let height = h / 4;
    let thickness = [0, 1, 3, height / 3, height / 2, height, height + 1];
    for t in thickness.iter() {
        let new_tensor =
            Vision::draw_rectangle(&tensor, (x, y), height, width, [255, 100, 255], *t).unwrap();
        Vision::save_image(&new_tensor, &format!("./assets/lenna_rectangle_{}.png", t)).unwrap();
    }
    // 2.测试灰度图片
    let tensor = Vision::load_image("./assets/lenna_luma.png").unwrap();
    let (h, w) = tensor.get_image_size().unwrap();
    let x = w / 2;
    let y = h / 2;
    let width = w / 2;
    let height = h / 4;
    let thickness = [0, 1, 3, width / 3, width / 2, width, width + 1];
    for t in thickness.iter() {
        let new_tensor =
            Vision::draw_rectangle(&tensor, (x, y), height, width, [250, 0, 0], *t).unwrap();
        Vision::save_image(
            &new_tensor,
            &format!("./assets/lenna_luma_rectangle_{}.png", t),
        )
        .unwrap();
    }
}
