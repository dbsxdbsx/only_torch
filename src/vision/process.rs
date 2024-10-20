use super::Vision;
use crate::tensor::Tensor;

impl Vision {
    pub fn median_blur(image: &Tensor, ksize: usize) -> Tensor {
        assert!(ksize >= 2);
        let mut blurred = image.clone();

        let (h, w, c) = image.get_image_shape().unwrap();
        let half_ksize = ksize / 2;

        for y in half_ksize..h - half_ksize {
            for x in half_ksize..w - half_ksize {
                if c == 0 {
                    let mut values = Vec::with_capacity(ksize * ksize);
                    for ky in (y - half_ksize)..=(y + half_ksize) {
                        for kx in (x - half_ksize)..=(x + half_ksize) {
                            values.push(image[[ky, kx]]);
                        }
                    }
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = values[values.len() / 2];

                    blurred[[y, x]] = median;
                } else {
                    for z in 0..c {
                        let mut values = Vec::with_capacity(ksize * ksize);
                        for ky in (y - half_ksize)..=(y + half_ksize) {
                            for kx in (x - half_ksize)..=(x + half_ksize) {
                                values.push(image[[ky, kx, z]]);
                            }
                        }
                        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let median = values[values.len() / 2];

                        blurred[[y, x, z]] = median;
                    }
                }
            }
        }

        blurred
    }
}
