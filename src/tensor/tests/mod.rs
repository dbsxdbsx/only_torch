mod add;
mod div;
mod eq;
mod index;
mod mul;
mod new;
mod others;
mod shape;
mod sub;

mod print;

mod save_load;

#[derive(Debug)]
struct TensorCheck {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub expected: Vec<Vec<f32>>, // 外层Vec的每个元素代表一个期望值，内层Vec（期望值）整体代表张量的数据
}
