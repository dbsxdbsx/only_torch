## 这是啥？

一个用纯Rust（不想用C++）打造的仿Pytorch的玩具型AI框架（目前尚不成熟，请勿使用）。该项目不打算支持GPU--因后期可能要支持安卓等平台，不想受制于某（几）种非CPU设备。但可能会加入NEAT等网络进化的算法。

### 名字由来

一部分原因是受到pytorch的影响，希望能写个和pytorch一样甚至更易用的AI框架；另一部分是希望本框架只触及（touch）一些关键的东西：

- only torch Rust --- 只用Rust（不用C++是因为其在复杂逻辑项目中容易写出内存不安全代码，也不打算支持Python接口）；也不用第三方lib（所以排除[tch-rs](https://github.com/LaurentMazare/tch-rs)），这样对跨平台支持会比较友好。
- only torch CPU --- 不用GPU，因要照顾多平台，也不想被某个GPU厂商制约，且基于NEAT进化的网络结构也不太好被GPU优化；如此也省得考虑数据从CPU内存迁移到其他设备内存的开销问题了。
- only torch node --- 没有全连接、卷积、resnet这类先入为主的算子概念，具体模型结构均基于NEAT进化。
- only torch tensor --- 所有的数据类型都是内置类型tensor（实现可能会参考[peroxide](https://crates.io/crates/peroxide)），不需要第三方处理库，如[numpy](https://github.com/PyO3/Rust-numpy)，[array](https://doc.Rust-lang.org/std/primitive.array.html)或[openBLAS](https://github.com/xianyi/OpenBLAS/wiki/User-Manual)（[关于blas的一些说明](https://blog.csdn.net/u013677156/article/details/77865405)）。
- only torch f32 --- 网络的参数（包括模型的输入、输出）不需要除了f32外的数据结构。

## 文档

目前无人性化的文档。可直接看Rust自动生成的[Api Doc](https://docs.rs/only_torch)即可。
### 使用示例

（无）

## TODO

**目前需要先解决有没有的问题，而不是好不好**
- [] 常用激活函数，tanh，Softplus，[sech](https://discuss.pytorch.org/t/implementing-sech/66862)
- [] 基于本框架解决XOR监督学习问题
- [] 基于本框架解决Mnist（数字识别）的监督学习问题
- [] 基于本框架解决CartPole（需要openAI Gym）的深度强化学习问题
- [] 尝试实现下[CFC](https://github.com/raminmh/CfC)
- [] [保存的json网络结构设计方案](https://www.perplexity.ai/search/516c7ae4-e5ec-47d2-a67a-22cd1d9285d2?s=c)

## 参考资料

### 训练用数据集

- [Mnist](http://yann.lecun.com/exdb/mnist/)
- [FashionMnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)
- [ChineseMnist](https://www.kaggle.com/datasets/gpreda/chinese-mnist)
- [训练用的各种数据集（包括强化学习）](https://huggingface.co/FUXI)

### 数学/IT原理

- [矩阵和向量的各种乘法](https://www.jianshu.com/p/9165e3264ced)
- [神经网络与记忆](https://www.bilibili.com/video/BV1fV4y1i7hZ/?spm_id_from=333.1007.0.0&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [陈天奇的机器学习编译课](https://www.bilibili.com/video/BV15v4y1g7EU/?is_story_h5=false&p=1&share_from=ugc&share_medium=android&share_plat=android&share_session_id=5a312434-ccf7-4cb9-862a-17a601cc4d35&share_source=COPY&share_tag=s_i&timestamp=1661386914&unique_k=zCWMKGC&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [基于梯度的机器学习IT原理](https://zhuanlan.zhihu.com/p/518198564)

### 开源示例

- [radiate--衍生NEAT的纯Rust库](https://github.com/pkalivas/radiate)
- [纯Rust的NEAT+GRU](https://github.com/sakex/neat-gru-Rust)
- [Rusty_sr-纯Rust的基于dl的图像超清](https://github.com/millardjn/Rusty_sr)
- [ndarray_glm(可参考下`array!`，分布，以及原生的BLAS)](https://docs.rs/ndarray-glm/latest/ndarray_glm/)

- [PyToy--基于MatrixSlow的Python机器学习框架](https://github.com/ysj1173886760/PyToy)

- [python：遗传算法（GE）玩FlappyBird](https://github.com/ShuhuaGao/gpFlappyBird)

（下面4个可以一起看）
- [手工微分：Rust-CNN](https://github.com/goldstraw/RustCNN)
- [Rust-dfdx---支持cuda的Rust深度学习库](https://docs.rs/dfdx/latest/dfdx/)
- [neuronika--纯Rust深度学习库（更新停滞了）](https://github.com/neuronika/neuronika)
- [用纯numpy写各类机器学习算法](https://github.com/ddbourgin/numpy-ml)

- []

- [深度学习框架InsNet简介](https://zhuanlan.zhihu.com/p/378684569)
- [C++机器学习库MLPACK](https://www.mlpack.org/)
- [经典机器学习算法Rust库](https://github.com/Rust-ml/linfa)
- [peroxide--纯Rust的线代及周边库](https://crates.io/crates/peroxide)

- [C++实现的NEAT+LSTM/GRU/CNN](https://github.com/travisdesell/exact)
- [pytorch+NEAT](https://github.com/ddehueck/pytorch-neat)
- [avalog--基于avatar的Rust逻辑推理库](https://crates.io/crates/avalog)

### NEAT、神经架构进化

- [用梯度指导神经架构进化：Splitting Steepest Descent](https://www.cs.utexas.edu/~qlearning/project.html?p=splitting)
- [Deep Mad，将卷积网络设计为一个数学建模问题](https://www.bilibili.com/video/BV1HP411R74T/?spm_id_from=333.999.0.0&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [autoML介绍](https://www.zhihu.com/question/554255720/answer/2750670583)

### 符号派：逻辑/因果推断

- [scryer-prolog--Rust逻辑推理库](https://github.com/mthom/scryer-prolog)
- [那迷人的被遗忘的语言：Prolog](https://zhuanlan.zhihu.com/p/41908829)
- [结合prolog和RL](https://arxiv.org/abs/2004.06997)
- [prolog与4证人难题](https://prolog.longluntan.com/t9-topic)
- [logic+mL提问](https://ai.stackexchange.com/questions/16224/has-machine-learning-been-combined-with-logical-reasoning-for-example-prolog)- [prolog解决数度问题](https://prolog.longluntan.com/t107-topic)
- [贝叶斯与逻辑推理](https://stats.stackexchange.com/questions/243746/what-is-probabilistic-inference)
- [用一阶逻辑辅佐人工神经网络](https://www.cs.cmu.edu/~hovy/papers/16ACL-NNs-and-logic.pdf)
- [二阶逻辑杂谈](https://blog.csdn.net/VucNdnrzk8iwX/article/details/128928166)
- [关于二阶逻辑的概念问题](https://www.zhihu.com/question/321025032/answer/702580771?utm_id=0)
- 书：《The Book of Why》
- 书：《Causality:Models,Reasoning,and Inference》
- [信息不完备下基于贝叶斯推断的可靠度优化方法](https://www.docin.com/p-2308549828.html)
- [贝叶斯网络中的因果推断](https://www.docin.com/p-1073204271.html?docfrom=rrela)

### 神经网络的可解释性

- [可解释性核心——神经网络的知识表达瓶颈](https://zhuanlan.zhihu.com/p/422420088/)
- [神经网络可解释性：论统一14种输入重要性归因算法](https://zhuanlan.zhihu.com/p/610774894/)
- [神经网络的可解释性](https://zhuanlan.zhihu.com/p/341153242)
- [可解释的哈萨尼网络](https://zhuanlan.zhihu.com/p/643213054)

### 超参数优化

- [mle-hyperopt](https://github.com/mle-infrastructure/mle-hyperopt)

### CPU加速

- [SLIDE](https://arxiv.org/abs/2103.10891)
- [Rust+AVX](https://medium.com/@Razican/learning-simd-with-Rust-by-finding-planets-b85ccfb724c3)
- [矩阵加速-GEMM](https://www.jianshu.com/p/6d3f013d8aba)

### AI实例项目
#### 强化学习

[Redeeming Intrinsic Rewards via Constrained Optimization](https://williamd4112.github.io/pubs/neurips22_eipo.pdf)
[Learning Reward Machines for Partially Observable Reinforcement Learning](https://arxiv.org/abs/2112.09477)

#### 其他
- [动手学深度学习-李沐著](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html#subsec-lin-algebra-norms)
- [openMMLab-Yolo](https://github.com/open-mmlab/mmyolo)
- [GRU解释](https://www.pluralsight.com/guides/lstm-versus-gru-units-in-rnn)
- [基于人类语音指挥的AI](https://arxiv.org/abs/1703.09831)
- [webGPT会上网的gpt](https://arxiv.org/abs/2112.09332)
- [chatGpt相关论文](https://arxiv.org/abs/2203.02155)
- [LeCun的自监督世界模型](https://zhuanlan.zhihu.com/p/636997984)
- [awesome Rust](https://github.com/Rust-unofficial/awesome-Rust#genetic-algorithms)
- [去雾算法](https://blog.csdn.net/IT_job/article/details/78864236)
- [rust人工智能相关的项目](https://github.com/rust-unofficial/awesome-rust#artificial-intelligence)

## 遵循协议

本项目遵循MIT协议（简言之：不约束，不负责）。
