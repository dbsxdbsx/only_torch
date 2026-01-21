/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-21
 * @Description  : Tensor类的测试模块
 */

mod add;
mod add_assign;
mod argmax;
mod div;
mod div_assign;
mod eq;
mod index;
mod mat_mul;
mod mul;
mod mul_assign;
mod new;
mod others;
mod property;
mod sub;
mod sub_assign;

mod filter;
mod print;

mod save_load;

mod slice;

#[derive(Debug)]
struct TensorCheck {
    pub input_shape: Vec<usize>,
    pub input_data: Vec<f32>,
    pub expected_output: Vec<Vec<f32>>, // 外层Vec的每个元素代表一个期望值，内层Vec（期望值）整体代表张量的数据
}
