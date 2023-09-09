use crate::tensor::Tensor;
use std::fs::File;

#[test]
fn test_save_load_tensor() {
    let orig_tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let mut file = File::create("test_save_load_tensor.bin").unwrap();
    orig_tensor.save(&mut file);

    let mut file = File::open("test_save_load_tensor.bin").unwrap();
    let loaded_tensor = Tensor::load(&mut file);
    assert_eq!(loaded_tensor, orig_tensor);
}
