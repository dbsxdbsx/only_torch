use crate::tensor::Tensor;
use std::fs::File;

#[test]
fn test_save_load_disk() {
    let orig_tensor = Tensor::new(&[1., 2., 3., 4.], &[2, 2]);
    let mut file = File::create("test_save_load_disk.bin").unwrap();
    orig_tensor.save_to_disk(&mut file);

    let mut file = File::open("test_save_load_disk.bin").unwrap();
    let loaded_tensor = Tensor::load_from_disk(&mut file);
    assert_eq!(loaded_tensor, orig_tensor);
}
