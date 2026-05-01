use crate::vision::detection::BBox;
use crate::vision::draw::{draw_bbox, draw_circle, draw_rectangle_xyxy};
use image::{DynamicImage, ImageBuffer, Rgb};

fn make_canvas(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(ImageBuffer::from_pixel(w, h, Rgb([0, 0, 0])))
}

#[test]
fn test_draw_bbox_outline_thin() {
    let mut canvas = make_canvas(20, 20);
    draw_bbox(
        &mut canvas,
        BBox::from_xyxy(5.0, 5.0, 14.0, 14.0),
        [255, 100, 50],
        1,
    );
    let buf = canvas.to_rgb8();
    assert_eq!(buf.get_pixel(5, 5).0, [255, 100, 50]);
    assert_eq!(buf.get_pixel(14, 14).0, [255, 100, 50]);
    // thickness=1 时框内应为空（黑）
    assert_eq!(buf.get_pixel(9, 9).0, [0, 0, 0]);
}

#[test]
fn test_draw_bbox_filled() {
    let mut canvas = make_canvas(10, 10);
    draw_bbox(
        &mut canvas,
        BBox::from_xyxy(2.0, 2.0, 7.0, 7.0),
        [10, 20, 30],
        0,
    );
    let buf = canvas.to_rgb8();
    assert_eq!(buf.get_pixel(4, 4).0, [10, 20, 30]);
    assert_eq!(buf.get_pixel(0, 0).0, [0, 0, 0]);
}

#[test]
fn test_draw_rectangle_xyxy_clips_outside() {
    let mut canvas = make_canvas(8, 8);
    // 越界部分应被裁掉，不能 panic
    draw_rectangle_xyxy(&mut canvas, -2, -2, 4, 4, [200, 200, 200], 0);
    let buf = canvas.to_rgb8();
    assert_eq!(buf.get_pixel(0, 0).0, [200, 200, 200]);
    assert_eq!(buf.get_pixel(4, 4).0, [200, 200, 200]);
    assert_eq!(buf.get_pixel(7, 7).0, [0, 0, 0]);
}

#[test]
fn test_draw_circle_filled_includes_center() {
    let mut canvas = make_canvas(20, 20);
    draw_circle(&mut canvas, (10, 10), 5, [255, 0, 0], 0);
    let buf = canvas.to_rgb8();
    assert_eq!(buf.get_pixel(10, 10).0, [255, 0, 0]);
    // 圆外仍黑
    assert_eq!(buf.get_pixel(0, 0).0, [0, 0, 0]);
}

#[test]
fn test_draw_circle_ring_excludes_center() {
    let mut canvas = make_canvas(20, 20);
    draw_circle(&mut canvas, (10, 10), 5, [0, 255, 0], 1);
    let buf = canvas.to_rgb8();
    // 圆心未涂色（thickness=1 只画环）
    assert_eq!(buf.get_pixel(10, 10).0, [0, 0, 0]);
    // 半径正好为 5 的位置应该被涂
    assert_eq!(buf.get_pixel(10, 5).0, [0, 255, 0]);
}

#[test]
fn test_draw_circle_clips_outside() {
    let mut canvas = make_canvas(10, 10);
    // 圆心在角落，部分越界
    draw_circle(&mut canvas, (0, 0), 4, [123, 123, 123], 0);
    let buf = canvas.to_rgb8();
    assert_eq!(buf.get_pixel(0, 0).0, [123, 123, 123]);
}
