/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @LastEditors  : 老董
 * @LastEditTime : 2024-01-11 21:14:09
 * @Description  : Tensor类的测试模块
 */

mod add;
mod add_assign;
mod div;
mod div_assign;
mod eq;
mod index;
mod mat_mul;
mod mul;
mod mul_assign;
mod new;
mod others;
mod shape;
mod sub;
mod sub_assign;

mod print;

mod save_load;

#[derive(Debug)]
struct TensorCheck {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub expected: Vec<Vec<f32>>, // 外层Vec的每个元素代表一个期望值，内层Vec（期望值）整体代表张量的数据
}
