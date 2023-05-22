## #描述
用纯rust（不想用c++）打造的仿造pytorch的个人玩具型AI框架（尚不成熟,请勿使用）。不打算支持gpu(因后期可能要支持安卓等平台，不想受制于某种非cpu设备)，但可能会加入NEAT等网络进化的算法。

### 名字由来
一部分原因是受到pytorch的影响，希望能写个和pytorch一样甚至更易用的AI框架；另一部分是希望本框架只触及（touch）一些关键的东西：

- only torch rust --- 只用rust（不用c++是因为其在复杂逻辑项目中容易写出内存不安全代码）；也不用第三方lib（所以排除[tch-rs](https://github.com/LaurentMazare/tch-rs)），这样对跨平台支持会比较友好。
- only torch cpu --- 不用gpu，因要照顾多平台，也不想被某个GPU厂商制约，且基于NEAT进化的网络结构也不太好被GPU优化；如此也省得考虑数据从cpu内存迁移到其他设备内存的开销问题了。
- only torch node --- 没有全连接、卷积、resnet这类先入为主的算子概念，具体模型结构均基于NEAT进化。
- only torch tensor --- 所有的数据类型都是内置类型tensor（实现可能会参考[peroxide](https://crates.io/crates/peroxide)），不需要第三方处理库，如[numpy](https://github.com/PyO3/rust-numpy)，[array](https://doc.rust-lang.org/std/primitive.array.html)或[openBLAS](https://github.com/xianyi/OpenBLAS/wiki/User-Manual)（[关于blas的一些说明](https://blog.csdn.net/u013677156/article/details/77865405)）。
- only torch f32 --- 网络的参数（包括模型的输入、输出）不需要除了f32外的数据结构。

## 使用示例
（无）

## 文档
目前无人性化的文档。可直接看rust自动生成的[Api Doc](https://docs.rs/only_torch)即可。

## TODO
（目前需要先解决有没有的问题；而不是好不好）
- [] 常用激活函数，tanh，Softplus，[sech](https://discuss.pytorch.org/t/implementing-sech/66862)
- [] 基于本框架解决XOR监督学习问题
- [] 基于本框架解决Mnist（数字识别）的监督学习问题
- [] 基于本框架解决CartPole（需要openAI Gym）的深度强化学习问题
- [] 尝试实现下[CFC](https://github.com/raminmh/CfC)
- [] [保存的json网络结构设计方案](https://www.perplexity.ai/search/516c7ae4-e5ec-47d2-a67a-22cd1d9285d2?s=c)

## 参考资料

### IT原理
- [陈天奇的机器学习编译课](https://www.bilibili.com/video/BV15v4y1g7EU/?is_story_h5=false&p=1&share_from=ugc&share_medium=android&share_plat=android&share_session_id=5a312434-ccf7-4cb9-862a-17a601cc4d35&share_source=COPY&share_tag=s_i&timestamp=1661386914&unique_k=zCWMKGC&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [基于梯度的机器学习IT原理](https://zhuanlan.zhihu.com/p/518198564)

### 开源示例
- [radiate--衍生NEAT的纯rust库](https://github.com/pkalivas/radiate)
- [纯rust的NEAT+GRU](https://github.com/sakex/neat-gru-rust)
- [rusty_sr-纯rust的基于dl的图像超清](https://github.com/millardjn/rusty_sr)
- [ndarray_glm(可参考下`array!`，分布，以及原生的BLAS)](https://docs.rs/ndarray-glm/latest/ndarray_glm/)

- [PyToy--基于MatrixSlow的python机器学习框架](https://github.com/ysj1173886760/PyToy)

- [n-nalgebra](https://github.com/dimforge/nalgebra)
- [rust-CNN](https://github.com/goldstraw/RustCNN)
- [rust-dfdx---支持cuda的rust深度学习库](https://docs.rs/dfdx/latest/dfdx/)
- [neuronika--纯rust深度学习库（更新停滞了）](https://github.com/neuronika/neuronika)

- [深度学习框架InsNet简介](https://zhuanlan.zhihu.com/p/378684569)
- [C++机器学习库MLPACK](https://www.mlpack.org/)
- [经典机器学习算法rust库](https://github.com/rust-ml/linfa)
- [peroxide--纯rust的线代及周边库](https://crates.io/crates/peroxide)

- [C++实现的NEAT+LSTM/GRU/CNN](https://github.com/travisdesell/exact)
- [pytorch+NEAT](https://github.com/ddehueck/pytorch-neat)
- [avalog--基于avatar的rust逻辑推理库](https://crates.io/crates/avalog)

### NEAT、神经架构进化
- [用梯度指导神经架构进化：Splitting Steepest Descent](https://www.cs.utexas.edu/~qlearning/project.html?p=splitting)
- [autoML介绍](https://www.zhihu.com/question/554255720/answer/2750670583)

### 逻辑/推理
- [scryer-prolog--rust逻辑推理库](https://github.com/mthom/scryer-prolog)
- [那迷人的被遗忘的语言：Prolog](https://zhuanlan.zhihu.com/p/41908829)
- [结合prolog和RL](https://arxiv.org/abs/2004.06997)
- [prolog与4证人难题](https://prolog.longluntan.com/t9-topic)
- [logic+mL提问](https://ai.stackexchange.com/questions/16224/has-machine-learning-been-combined-with-logical-reasoning-for-example-prolog)- [prolog解决数度问题](https://prolog.longluntan.com/t107-topic)
- [贝叶斯与逻辑推理](https://stats.stackexchange.com/questions/243746/what-is-probabilistic-inference)
- [神经网络的可解释性](https://zhuanlan.zhihu.com/p/341153242)

### CPU加速
- [SLIDE](https://arxiv.org/abs/2103.10891)
- [rust+AVX](https://medium.com/@Razican/learning-simd-with-rust-by-finding-planets-b85ccfb724c3)
- [矩阵加速-GEMM](https://www.jianshu.com/p/6d3f013d8aba)

### 其他
- [动手学深度学习-李沐著](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html#subsec-lin-algebra-norms)
- [openMMLab-Yolo](https://github.com/open-mmlab/mmyolo)
- [GRU解释](https://www.pluralsight.com/guides/lstm-versus-gru-units-in-rnn)
- [基于人类语音指挥的AI](https://arxiv.org/abs/1703.09831)
- [webGPT会上网的gpt](https://arxiv.org/abs/2112.09332)
- [chatGpt相关论文](https://arxiv.org/abs/2203.02155)
- [awesome rust](https://github.com/rust-unofficial/awesome-rust#genetic-algorithms)
- [去雾算法](https://blog.csdn.net/IT_job/article/details/78864236)

## 遵循协议
本项目遵循MIT协议（简言之：不约束，不负责）。
