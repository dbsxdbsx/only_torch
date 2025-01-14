/*
 * @Author       : 老董
 * @Date         : 2024-10-24 09:18:44
 * @Description  : 自适应线性神经元（Adaptive Linear Neuron，ADALINE）网络测试，参考自：https://github.com/zc911/MatrixSlow/blob/master/example/ch02/adaline.py
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-14 16:32:26
 */
use only_torch::nn::{Graph, GraphError};
use only_torch::tensor::Tensor;
use only_torch::tensor_where;

#[test]
fn test_adaline() -> Result<(), GraphError> {
    let male_heights = Tensor::normal(171.0, 6.0, &[500]);
    let female_heights = Tensor::normal(158.0, 5.0, &[500]);

    let male_weights = Tensor::normal(70.0, 10.0, &[500]);
    let female_weights = Tensor::normal(57.0, 8.0, &[500]);

    let male_bfrs = Tensor::normal(16.0, 2.0, &[500]);
    let female_bfrs = Tensor::normal(22.0, 2.0, &[500]);

    let male_labels = Tensor::new(&[1.0; 500], &[500]);
    let female_labels = Tensor::new(&[-1.0; 500], &[500]);

    let mut train_set = Tensor::stack(
        &[
            &Tensor::stack(&[&male_heights, &female_heights], false),
            &Tensor::stack(&[&male_weights, &female_weights], false),
            &Tensor::stack(&[&male_bfrs, &female_bfrs], false),
            &Tensor::stack(&[&male_labels, &female_labels], false),
        ],
        true,
    );
    train_set.permute_mut(&[1, 0]);
    train_set.shuffle_mut(Some(0)); // 随机打乱样本顺序
    println!("{:?}", train_set.shape());

    // 创建计算图
    let mut graph = Graph::new();

    // 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
    let x = graph.new_input_node(&[3, 1], Some("x"))?;
    // 类别标签，1男，-1女
    let label = graph.new_input_node(&[1, 1], Some("label"))?;
    // 权重向量，是一个1x3矩阵，需要初始化，参与训练
    let w = graph.new_parameter_node(&[1, 3], Some("w"))?;
    // 阈值，是一个1x1矩阵，需要初始化，参与训练
    let b = graph.new_parameter_node(&[1, 1], Some("b"))?;

    // ADALINE的预测输出
    let wx = graph.new_mat_mul_node(w, x, None)?;
    let output = graph.new_add_node(&[wx, b], None)?;
    let predict = graph.new_step_node(output, None)?;

    // 损失函数
    let loss_input = graph.new_mat_mul_node(label, output, Some("loss_input"))?;
    let loss = graph.new_perception_loss_node(loss_input, Some("loss"))?;

    // 学习率
    let learning_rate = 0.0001;

    // 训练执行50个epoch
    for epoch in 0..50 {
        // 遍历训练集中的样本
        for i in 0..train_set.shape()[0] {
            // 取第i个样本的前4列（除最后一列的所有列），构造3x1矩阵对象
            let features = train_set.slice(&[&i, &(0..3)]).transpose();
            // 取第i个样本的最后一列，是该样本的性别标签（1男，-1女），构造1x1矩阵对象
            let l = train_set.slice(&[&i, &3]);

            // 将特征赋给x节点，将标签赋给label节点
            graph.set_node_value(x, Some(&features))?;
            graph.set_node_value(label, Some(&l))?;

            // 在loss节点上执行前向传播，计算损失值
            graph.forward_node(loss)?;

            // 在w和b节点上执行反向传播，计算损失值对它们的雅可比矩阵
            graph.backward_node(w, loss)?;
            graph.backward_node(b, loss)?;

            // 更新参数
            let w_value = graph.get_node_value(w)?.unwrap();
            let w_jacobi = graph.get_node_jacobi(w)?.unwrap();
            graph.set_node_value(
                w,
                Some(&(w_value - learning_rate * w_jacobi.transpose().reshape(w_value.shape()))),
            )?;

            let b_value = graph.get_node_value(b)?.unwrap();
            let b_jacobi = graph.get_node_jacobi(b)?.unwrap();
            graph.set_node_value(
                b,
                Some(&(b_value - learning_rate * b_jacobi.transpose().reshape(b_value.shape()))),
            )?;

            // 清除所有节点的雅可比矩阵
            graph.clear_jacobi()?;
        }

        // 每个epoch结束后评价模型的正确率
        let mut pred_vec = Vec::with_capacity(train_set.shape()[0]);

        // 遍历训练集，计算当前模型对每个样本的预测值
        for i in 0..train_set.shape()[0] {
            let features = train_set.slice(&[&i, &(0..3)]).transpose();
            graph.set_node_value(x, Some(&features))?;

            // 在模型的predict节点上执行前向传播
            graph.forward_node(predict)?;
            let v = graph.get_node_value(predict)?.unwrap().get(&[0, 0]);
            pred_vec.push(v.get_data_number().unwrap()); // 模型的预测结果：1男，0女
        }
        let pred = Tensor::new(&pred_vec, &[pred_vec.len(), 1]) * 2.0 - 1.0; // 将1/0结果转化成1/-1结果，好与训练标签的约定一致

        // 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
        let train_set_labels = train_set.slice(&[&(..), &3]);
        let filtered_sum = tensor_where!(train_set_labels == pred, 1.0, 0.0).sum();
        let train_set_len = train_set.shape()[0] as f32;
        let accuracy = filtered_sum / train_set_len;

        // 打印当前epoch数和模型在训练集上的正确率
        println!(
            "训练回合: {}, 正确率: {:.1}%",
            epoch + 1,
            accuracy.get_data_number().unwrap() * 100.0
        );
    }
    Ok(())
}

