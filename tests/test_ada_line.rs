use only_torch::nn::nodes::{Add, Variable};
use only_torch::tensor::Tensor;

#[test]
fn test_ada_line() {
    let male_heights = Tensor::normal(171.0, 6.0, &[500]);
    let female_heights = Tensor::normal(158.0, 5.0, &[500]);

    let male_weights = Tensor::normal(70.0, 10.0, &[500]);
    let female_weights = Tensor::normal(57.0, 8.0, &[500]);

    let male_bfrs = Tensor::normal(16.0, 2.0, &[500]);
    let female_bfrs = Tensor::normal(22.0, 2.0, &[500]);

    let male_labels = Tensor::new(&vec![1.0; 500], &[500]);
    let female_labels = Tensor::new(&vec![-1.0; 500], &[500]);

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
    // let output = Add::new(&vec![MatMul::new(&vec![w, x]), b], None);
    // predict = ms.ops.Step(output)
    // 损失函数
    // loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(label, output))
    // 学习率
    let _learning_rate = 0.0001;

    // 训练执行50个epoch
    for _i in 0..50 {
        // 遍历训练集中的样本
        for j in 0..train_set.shape()[0] {
            // 获取当前样本的特征和标签
            let sample = train_set.get(&[j]); // TODO: use view?
            println!("{:?}", sample.shape());
            // let features = sample.get(&[0, 0..3]); //TODO:
            // let label = sample.get(&[0, 3]);

            // // 计算当前样本的预测值
            // let mut pred = features.mat_mul(&Tensor::new(&[0.0, 0.0, 0.0], &[3, 1]));
            // pred += Tensor::new(1.0, &[]);
            // // 计算当前样本的误差
            // let mut error = label - pred;
            // // 计算当前样本的梯度
            // let mut grad = features.transpose().mat_mul(&error);
            // // 更新模型参数
        }
    }
}
