use only_torch::nn::nodes::{Add, MatMul, TraitForNode, Variable};
use only_torch::tensor::Tensor;

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
    let x = Variable::new(&[3, 1], false, false, None);
    // 类别标签，1男，-1女
    let label = Variable::new(&[1, 1], false, false, None);
    // 权重向量，是一个1x3矩阵，需要初始化，参与训练
    let w = Variable::new(&[1, 3], true, true, None);
    // 阈值，是一个1x1矩阵，需要初始化，参与训练
    let b = Variable::new(&[1, 1], true, true, None);

    // ADALINE的预测输出
    let output = Add::new(
        &[
            MatMul::new(&[w.as_node_enum(), x.as_node_enum()], None).as_node_enum(),
            b.as_node_enum(),
        ],
        None,
    );
    let predict = Step::new(&[output.as_node_enum()], None);

    // 损失函数
    let loss = PerceptionLoss::new(
        &[MatMul::new(&[label.as_node_enum(), output.as_node_enum()], None).as_node_enum()],
        None,
    );

    // 学习率
    let learning_rate = 0.0001;

    // 训练执行50个epoch
    for epoch in 0..50 {
        // 遍历训练集中的样本
        for i in 0..train_set.shape()[0] {
            // 取第i个样本的特征和标签
            let features = train_set.slice(&[i, 0..3]).reshape(&[3, 1]);
            let l = train_set.slice(&[i, 3]).reshape(&[1, 1]);

            // 将特征赋给x节点，将标签赋给label节点
            x.set_value(features);
            label.set_value(l);

            // 在loss节点上执行前向传播，计算损失值
            loss.forward();

            // 在w和b节点上执行反向传播，计算损失值对它们的雅可比矩阵
            w.backward(&loss);
            b.backward(&loss);

            // // 更新参数
            // // w.set_value(w.value - learning_rate * w.jacobi.T.reshape(w.shape()))
            // // b.set_value(b.value - learning_rate * b.jacobi.T.reshape(b.shape()))
            // w.update(learning_rate);
            // b.update(learning_rate);

            // // 清除所有节点的雅可比矩阵
            // TODO: default_graph.clear_jacobi();
        }

        // // 评价模型的正确率
        // let mut correct_count = 0;
        // for i in 0..train_set.shape()[0] {
        //     let features = train_set.slice(&[i, 0..3]).reshape(&[3, 1]);
        //     x.set_value(features);

        //     predict.forward();
        //     let pred = if predict.value().get(&[0, 0]) > &0.0 { 1.0 } else { -1.0 };
        //     let true_label = train_set.get(&[i, 3]);

        //     if (pred - true_label).abs() < 1e-5 {
        //         correct_count += 1;
        //     }
        // }

        // let accuracy = correct_count as f32 / train_set.shape()[0] as f32;
        // println!("epoch: {}, accuracy: {:.3}", epoch + 1, accuracy);
    }
}
