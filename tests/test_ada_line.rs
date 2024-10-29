/*
 * @Author       : 老董
 * @Date         : 2024-10-24 09:18:44
 * @Description  : ⾃适应线性神经元（Adaptive Linear Neuron，ADALINE）网络测试，参考自：https://github.com/zc911/MatrixSlow/blob/master/example/ch02/adaline.py
 * @LastEditors  : 老董
 * @LastEditTime : 2024-10-29 12:22:29
 */
use only_torch::nn::nodes::{Add, MatMul, PerceptionLoss, Step, TraitForNode, Variable};
use only_torch::tensor::Tensor;
use only_torch::tensor_where;

#[test]
fn test_ada_line() {
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

    // 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
    let mut x = Variable::new(&[3, 1], false, false, None);
    // 类别标签，1男，-1女
    let mut label = Variable::new(&[1, 1], false, false, None);
    // 权重向量，是一个1x3矩阵，需要初始化，参与训练
    let mut w = Variable::new(&[1, 3], true, true, None);
    // 阈值，是一个1x1矩阵，需要初始化，参与训练
    let mut b = Variable::new(&[1, 1], true, true, None);

    // ADALINE的预测输出
    let output = Add::new(
        &[
            MatMul::new(&[w.as_node_enum(), x.as_node_enum()], None).as_node_enum(),
            b.as_node_enum(),
        ],
        None,
    );
    let mut predict = Step::new(&[output.as_node_enum()], None);

    // 损失函数
    let mut loss = PerceptionLoss::new(
        &[MatMul::new(&[label.as_node_enum(), output.as_node_enum()], None).as_node_enum()],
        None,
    );

    // 学习率
    let learning_rate = 0.0001;

    // 训练执行50个epoch
    for epoch in 0..50 {
        //     // 遍历训练集中的样本
        for i in 0..train_set.shape()[0] {
            // 取第i个样本的前4列（除最后一列的所有列），构造3x1矩阵对象
            let features = train_set.slice(&[&i, &(0..3)]).transpose();
            // 取第i个样本的最后一列，是该样本的性别标签（1男，-1女），构造1x1矩阵对象
            let l = train_set.slice(&[&i, &3]);

            // 将特征赋给x节点，将标签赋给label节点
            x.set_value(&features);
            label.set_value(&l);

            // 在loss节点上执行前向传播，计算损失值
            loss.forward();

            // 在w和b节点上执行反向传播，计算损失值对它们的雅可比矩阵
            w.backward(&loss.as_node_enum());
            b.backward(&loss.as_node_enum());

            // 更新参数:
            // 用损失值对w和b的雅可比矩阵（梯度的转置）更新参数值。我们想优化的节点
            // 都应该是标量节点（才有所谓降低其值一说），它对变量节点的雅可比矩阵的
            // 形状都是1 x n。这个雅可比的转置是结果节点对变量节点的梯度。将梯度再
            // reshape成变量矩阵的形状，对应位置上就是结果节点对变量元素的偏导数。
            // 将改变形状后的梯度乘上学习率，从当前变量值中减去，再赋值给变量节点，
            // 完成梯度下降更新。
            w.set_value(
                &(w.value() - learning_rate * w.jacobi().transpose().reshape(w.value().shape())),
            );
            b.set_value(
                &(b.value() - learning_rate * b.jacobi().transpose().reshape(b.value().shape())),
            );

            // 清除所有节点的雅可比矩阵
            // TODO: default_graph.clear_jacobi();
        }

        // 每个epoch结束后评价模型的正确率
        let mut pred_vec = Vec::with_capacity(train_set.shape()[0]);

        // 遍历训练集，计算当前模型对每个样本的预测值
        for i in 0..train_set.shape()[0] {
            let features = train_set.slice(&[&i, &(0..3)]).transpose();
            x.set_value(&features);

            // 在模型的predict节点上执行前向传播
            predict.forward();
            let v = predict.value().get(&[0, 0]);
            pred_vec.push(v.get_data_number().unwrap()); // 模型的预测结果：1男，0女
        }
        let pred = Tensor::new(&pred_vec, &[pred_vec.len()]) * 2.0 - 1.0; // 将1/0结果转化成1/-1结果，好与训练标签的约定一致

        // 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
        let train_set_labels = train_set.slice(&[&(..), &3]);
        let filtered_sum = tensor_where!(train_set_labels == pred, 1.0, 0.0).sum();
        let train_set_len = train_set.shape()[0] as f32;
        let accuracy = filtered_sum / train_set_len;

        // 打印当前epoch数和模型在训练集上的正确率
        println!("epoch: {}, accuracy: {:.3}", epoch + 1, accuracy);
    }
}
