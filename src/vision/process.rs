use super::Vision;
use crate::tensor::Tensor;

impl Vision {
    pub fn median_blur(image: &Tensor, ksize: usize) -> Tensor {
        assert!(ksize >= 2);
        let mut blurred_tensor = image.clone();
        let mut blurred = blurred_tensor.view_mut();

        let (h, w, c) = image.get_image_shape().unwrap();
        let half_ksize = ksize / 2;
        let orig_view = image.view();

        for y in half_ksize..h - half_ksize {
            for x in half_ksize..w - half_ksize {
                if c == 0 {
                    let mut values = Vec::with_capacity(ksize * ksize);
                    for ky in y - half_ksize..y + half_ksize + 1 {
                        for kx in x - half_ksize..x + half_ksize + 1 {
                            values.push(orig_view[[ky, kx]]);
                        }
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = values[values.len() / 2];

                    blurred[[y, x]] = median;
                } else {
                    for z in 0..c {
                        let mut values = Vec::with_capacity(ksize * ksize);
                        for ky in y - half_ksize..y + half_ksize + 1 {
                            for kx in x - half_ksize..x + half_ksize + 1 {
                                values.push(orig_view[[ky, kx, z]]);
                            }
                        }
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = values[values.len() / 2];

                        blurred[[y, x, z]] = median;
                    }
                }
            }
        }

        blurred_tensor
    }
}
