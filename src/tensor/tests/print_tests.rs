use crate::tensor::Tensor;

#[test]
fn test_print() {
    use std::fmt::Write;

    // 测试向量
    // let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    // let mut buffer = String::new();
    // write!(&mut buffer, "{}", tensor).unwrap();
    // assert_eq!(buffer, "[  1.0000,\n   2.0000,\n   3.0000]\n形状: [3]\n");

    // 测试矩阵
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mut buffer = String::new();
    write!(&mut buffer, "{}", tensor).unwrap();
    assert_eq!(
        buffer,
        "[[  1.0000,   2.0000,   3.0000], \n [  4.0000,   5.0000,   6.0000]]\n形状: [2, 3]\n"
    );
}

// #[test]
// fn test_print() {
//     let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
//     println!("{}", tensor);
//     let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
//     println!("{}", tensor);

//     // let tensor = Tensor::new_eye(1);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_eye(2);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_eye(3);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_eye(7);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(2, 2, 0.0, 1.0);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(7, 4, 0.0, 1.0);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(4, 7, 0.0, 1.0);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(1, 7, 0.0, 1.0);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(7, 1, 0.0, 1.0);
//     // println!("{}", tensor);
//     // let tensor = Tensor::new_random(6, 6, 0.0, 1.0);
//     // println!("{}", tensor);
// }
