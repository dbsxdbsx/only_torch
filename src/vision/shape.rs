use super::Vision;
use crate::tensor::Tensor;
use crate::utils::traits::image::ForDynamicImage;

impl Vision {
    /// 调整图像大小
    /// * `image` - 原始图像
    /// * `width` - 调整后的宽度
    /// * `height` - 调整后的高度
    /// * `crop` - `true`则执行基于中心的裁剪，`false`则执行基于中心的缩放
    ///
    /// 这里特意用`resize`而不是`reshape`，只为强调其只会改变尺寸，而不会改变张量本身的维度。
    pub fn resize_image(
        image: &Tensor,
        height: usize,
        width: usize,
        crop: bool,
    ) -> Result<Tensor, String> {
        let image = image.to_image()?;
        if crop {
            assert!(
                height <= image.height() as usize,
                "裁剪图像：新高度必须小于原始高度。"
            );
            assert!(
                width <= image.width() as usize,
                "裁剪图像：新宽度必须小于原始宽度。"
            );
            image
                .crop_imm(
                    (image.width() - width as u32) / 2,
                    (image.height() - height as u32) / 2,
                    width as u32,
                    height as u32,
                )
                .to_tensor()
        } else {
            image
                .resize_exact(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::Triangle,
                )
                .to_tensor()
        }
    }
}
