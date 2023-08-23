use crate::tensor::Tensor;

#[test]
fn test_index() {
    let data = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = &[2, 3];
    let tensor = Tensor::new(data, shape);
    tensor.print();

    println!("{:?}", tensor[&[0]]); // 输出 [1.0, 2.0, 3.0]
    println!("{:?}", tensor[&[0, 1]]); // 输出 [2.0]
    println!("{:?}", tensor[&[1, 2]]); // 输出 [6.0]

    // let  xxxx = t[&[1,1]];

    // println!("xxxx = {:?}", xxxx);
    // assert_eq!(t[0], Tensor::new(&vec![1.0, 2.0], &[1, 2]));
    // assert_eq!(t[(1, 1)], &4.0);
    // assert!(std::panic::catch_unwind(|| t[(1, 1, 1)]).is_err());
}
