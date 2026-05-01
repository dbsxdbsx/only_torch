use crate::vision::viz::{Palette, TinyFont, blend_alpha, pixel_block_scale};
use image::{ImageBuffer, Rgb};

fn black_canvas(w: u32, h: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    ImageBuffer::from_pixel(w, h, Rgb([0u8, 0, 0]))
}

#[test]
fn test_palette_default_categorical_size() {
    let p = Palette::default_categorical();
    assert_eq!(p.len(), 8);
    assert!(!p.is_empty());
}

#[test]
fn test_palette_color_wraps_on_overflow() {
    let p = Palette::new(vec![[10, 20, 30], [40, 50, 60]]);
    assert_eq!(p.color(0), [10, 20, 30]);
    assert_eq!(p.color(1), [40, 50, 60]);
    assert_eq!(p.color(2), [10, 20, 30]);
    assert_eq!(p.color(123), [40, 50, 60]);
}

#[test]
#[should_panic(expected = "Palette: 颜色数必须 >= 1")]
fn test_palette_empty_panics() {
    let _ = Palette::new(Vec::new());
}

#[test]
fn test_pixel_block_scale_fills_correct_area() {
    let mut canvas = black_canvas(10, 10);
    pixel_block_scale(&mut canvas, 1, 1, [255, 0, 0], 3);

    // (1, 1) 块覆盖 x=3..6, y=3..6
    for y in 0..10 {
        for x in 0..10 {
            let p = canvas.get_pixel(x, y).0;
            let inside = (3..6).contains(&x) && (3..6).contains(&y);
            if inside {
                assert_eq!(p, [255, 0, 0], "({x},{y}) 应该被填充");
            } else {
                assert_eq!(p, [0, 0, 0], "({x},{y}) 应保持原色");
            }
        }
    }
}

#[test]
fn test_pixel_block_scale_zero_is_noop() {
    let mut canvas = black_canvas(5, 5);
    pixel_block_scale(&mut canvas, 1, 1, [255, 0, 0], 0);
    for (_, _, pixel) in canvas.enumerate_pixels() {
        assert_eq!(pixel.0, [0, 0, 0]);
    }
}

#[test]
fn test_pixel_block_scale_clips_outside_canvas() {
    let mut canvas = black_canvas(4, 4);
    // block (1, 1) 在 scale=3 下覆盖 x=3..6, y=3..6，超出 4x4 canvas 边界
    pixel_block_scale(&mut canvas, 1, 1, [200, 100, 50], 3);
    // (3, 3) 是 canvas 内唯一被填充的像素，越界部分静默裁剪不应 panic
    assert_eq!(canvas.get_pixel(3, 3).0, [200, 100, 50]);
}

#[test]
fn test_blend_alpha_endpoints_and_midpoint() {
    let base = [100, 100, 100];
    let overlay = [200, 50, 0];

    assert_eq!(blend_alpha(base, overlay, 0.0), base);
    assert_eq!(blend_alpha(base, overlay, 1.0), overlay);

    let mid = blend_alpha(base, overlay, 0.5);
    // (100 + 200) / 2 = 150, (100 + 50) / 2 = 75, (100 + 0) / 2 = 50
    assert_eq!(mid, [150, 75, 50]);
}

#[test]
fn test_blend_alpha_clamps_out_of_range() {
    let base = [10, 10, 10];
    let overlay = [200, 200, 200];

    assert_eq!(blend_alpha(base, overlay, -0.5), base);
    assert_eq!(blend_alpha(base, overlay, 2.0), overlay);
}

#[test]
fn test_tiny_font_text_width() {
    assert_eq!(TinyFont::text_width(""), 0);
    assert_eq!(TinyFont::text_width("A"), 3);
    assert_eq!(TinyFont::text_width("AB"), 7);
    assert_eq!(TinyFont::text_width("IoU 50%"), 7 * 4 - 1);
}

#[test]
fn test_tiny_font_draw_writes_pixels() {
    let mut canvas = black_canvas(20, 10);
    TinyFont::draw(&mut canvas, 1, 1, "I", [255, 255, 255]);
    // 'I' 5x3 pattern: ["111","010","010","010","111"]
    // 第 0 行 (y=1) 应该是三列全亮
    assert_eq!(canvas.get_pixel(1, 1).0, [255, 255, 255]);
    assert_eq!(canvas.get_pixel(2, 1).0, [255, 255, 255]);
    assert_eq!(canvas.get_pixel(3, 1).0, [255, 255, 255]);
    // 第 1 行 (y=2) 中间列亮
    assert_eq!(canvas.get_pixel(1, 2).0, [0, 0, 0]);
    assert_eq!(canvas.get_pixel(2, 2).0, [255, 255, 255]);
    assert_eq!(canvas.get_pixel(3, 2).0, [0, 0, 0]);
}

#[test]
fn test_tiny_font_draw_with_box_fills_background() {
    let mut canvas = black_canvas(40, 20);
    TinyFont::draw_with_box(&mut canvas, 5, 5, "OK", [255, 255, 255], [50, 50, 50]);
    // 底框尺寸：text_width("OK") + 4 = 7 + 4 = 11，高度 = 5 + 4 = 9
    // 角落像素应被填上 bg_color
    assert_eq!(canvas.get_pixel(5, 5).0, [50, 50, 50]);
    assert_eq!(canvas.get_pixel(15, 5).0, [50, 50, 50]);
    assert_eq!(canvas.get_pixel(5, 13).0, [50, 50, 50]);
}

#[test]
fn test_tiny_font_lowercase_falls_back_to_uppercase() {
    let mut a = black_canvas(10, 10);
    let mut b = black_canvas(10, 10);
    TinyFont::draw(&mut a, 0, 0, "A", [255, 255, 255]);
    TinyFont::draw(&mut b, 0, 0, "a", [255, 255, 255]);
    // 小写 'a' 应渲染成跟 'A' 一样
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        assert_eq!(pa, pb);
    }
}

#[test]
fn test_tiny_font_unknown_character_renders_blank() {
    let mut canvas = black_canvas(10, 10);
    TinyFont::draw(&mut canvas, 0, 0, "@", [255, 255, 255]);
    // '@' 不在字典中，整个 canvas 应保持黑色
    for (_, _, pixel) in canvas.enumerate_pixels() {
        assert_eq!(pixel.0, [0, 0, 0]);
    }
}
