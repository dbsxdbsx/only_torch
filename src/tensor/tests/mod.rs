mod add_tests;
mod div_tests;
mod eq_tests;
mod index_tests;
mod mul_tests;
mod new_tests;
mod others_tests;
mod shape_tests;
mod sub_tests;

mod print_tests;

#[derive(Debug)]
struct TensorCheck {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub expected: Vec<Vec<f32>>, // 外层Vec的每个元素代表一个期望值，内层Vec（期望值）整体代表张量的数据
}
