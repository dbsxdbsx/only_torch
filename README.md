## 这是啥？

一个用纯Rust（不想用C++）打造的仿Pytorch的玩具型AI框架（目前尚不成熟，请勿使用）。该项目不打算支持GPU--因后期可能要支持安卓等平台，不想受制于某（几）种非CPU设备。但可能会加入NEAT等网络进化的算法。

### 名字由来

一部分原因是受到pytorch的影响，希望能写个和pytorch一样甚至更易用的AI框架；另一部分是希望本框架只触及（touch）一些关键的东西：

- only torch Rust --- 只用Rust（不用C++是因为其在复杂逻辑项目中容易写出内存不安全代码，也不打算支持Python接口）；也不用第三方lib（所以排除[tch-rs](https://github.com/LaurentMazare/tch-rs)），这样对跨平台支持会比较友好。
- only torch CPU --- 不用GPU，因要照顾多平台也不想被某个GPU厂商制约，且基于NEAT进化的网络结构也不太好被GPU优化（也省得考虑数据从CPU的堆栈迁移到其他设备内存的开销问题了）。
- only torch node --- 没有全连接、卷积、resnet这类先入为主的算子概念，具体模型结构均基于NEAT进化。
- only torch tensor --- 所有的数据类型都是内置类型tensor（实现可能会参考[peroxide](https://crates.io/crates/peroxide)），不需要第三方处理库，如[numpy](https://github.com/PyO3/Rust-numpy)，[array](https://doc.Rust-lang.org/std/primitive.array.html)或[openBLAS](https://github.com/xianyi/OpenBLAS/wiki/User-Manual)（[关于blas的一些说明](https://blog.csdn.net/u013677156/article/details/77865405)）。
- only torch f32 --- 网络的参数（包括模型的输入、输出）不需要除了f32外的数据类型。

## 文档

目前无人性化的文档。可直接看Rust自动生成的[Api Doc](https://docs.rs/only_torch)即可。

### 使用示例

（无）

## TODO
- 各种assign类的op（如：add_assign）是否需要重载而不是复用基本算子？
- `test_dot_sum_operator_for_inconsistent_shape_1`好像内容不是测试的dot_sum, 而是乘法？
- ada_line还是有问题
- graph反向传播中有些节点没有值需要过滤怎么添加（如多个output的网络结构）？
- 尝试添加add节点的测试，然后再统一优化Variable节点和Add节点的布局？
- 除了Variable节点，其他节点须遵循`tests\calc_jacobi_by_pytorch`的测试
- Graph测试中该包含各种pub method的正确及错误测试，如何set_node_trainable，is_node_trainable...
- 对比Node_variable和Graph的测试，看看如何优化精简Graph的测试
- Graph测试中最好添加某个节点后，测试该节点还有其父节点的parents/children属性（又比如：同2个节点用于不同图的add节点，测试其parents/children属性是否正确）(Variable 节点无父节点)、“节点var1在图default_graph中重复”
- add a `graph` for unit test to test the 多层的jacobi计算，就像ada_line那样?

- 在python中仿造ada_Line构造一个复合多节点，然后基于此在rust中测试这种复合节点，已验证在复合多层节点中的反向传播正确性
- jacobi到底该测试对parent还是children？

- how to expose only `in crate::nn` to the nn::Graph`?
- should completely hide the NodeHandle?
- Graph/NodeHandle rearrange blocks due to visibility and funciontality

- NodeHandle重命名为Node? 各种`parent/children/node_id`重命名为`parents/children/id`?
- should directly use `parents` but not `parents_ids`?

- unit test for Graph, and parent/children
- unit test for each current module methods
- check other unused methods
- draw_graph(graphvi画图)
- save/load网络模型（已有test_save_load_tensor）
- 后期当NEAT，可以给已存在节点添加父子节点后，需要把现有节点检测再完善下；
- 当后期（NEAT阶段）需要在一个已经forwarded的图中添加节点（如将已经被使用过的var1、var2结合一个新的未使用的var3构建一个add节点），可能需要添加一个`reset_forward_cnt`方法来保证图forward的一致性。
- NEAT之后，针对图backward的`loss1.backward(retain_graph=True)`和`detach()`机制的实现（可在GAN和强化学习算法实例中针对性实现测试），可能须和`forward_cnt`机制结合, 还要考虑一次forward后多次backward()后的结果。
- Tensor 真的需要uninit吗？
- 各种命名规范“2维”，“二维”，“二阶”，“2阶”，“一个”，“两个”，“三个”，“需要”，“需”，“须要”，“须”，“值/value”,"变量/variable","node/handle"，“注/注意：”
-
- 根据matrixSlow+我笔记重写全部实现！保证可以后期以NEAT进化,能ok拓展至linear等常用层，还有detach，，容易添加edge(如已存在的add节点的父节点)，。
- 等ada_line例子跑通后：`Variable`节点做常见的运算重载（如此便不需要用那些丑陋的节点算子了）
- 图错误“InvalidOperation” vs “ComputationError”
- `parent.borrow_mut()`或`.children_mut()`改变后如何保证其matrix形状是合法的该节点运算后matrix?
- `fn as_node_enum(&self) -> NodeEnum {
        NodeEnum::Step(self.clone())
    }`会否影响计算图graph？
- Tensorlei的index将`[[`优化成`[`?
- Tensor类的`slice(&[0..m, j..j+1])`是否需要？
- `children_mut`是否可合并至`children()`? and `value_mut`是否可合并至`value`?
- `fn as_node_enum(&self) -> NodeEnum` trait method 是否多余，对于具体实现的节点，可否隐式转换或直接各节点返回NodeEnum？(只要不要影响后期各种算子的重载)？
- use approx::assert_abs_diff_eq; need or not?
- 使用f16代替f32？

**目前需要先解决有没有的问题，而不是好不好**
- [] 实现类似tch-rs中`tch::no_grad(|| {});`的无梯度功能；
- [] 常用激活函数，tanh，Softplus，[sech](https://discuss.pytorch.org/t/implementing-sech/66862)
- [] 基于本框架解决XOR监督学习问题
- [] 基于本框架解决Mnist（数字识别）的监督学习问题
- [] 基于本框架解决CartPole（需要openAI Gym或相关crate支持）的深度强化学习问题
- [] 尝试实现下[CFC](https://github.com/raminmh/CfC)

## 参考资料

### 训练用数据集（包括强化学习gym）

- [Mnist](http://yann.lecun.com/exdb/mnist/)
- [FashionMnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)
- [ChineseMnist](https://www.kaggle.com/datasets/gpreda/chinese-mnist)
- [训练用的各种数据集（包括强化学习）](https://huggingface.co/FUXI)
- [bevy_rl](https://crates.io/crates/bevy_rl)
- [pure_rust_gym](https://github.com/MathisWellmann/gym-rs/tree/master)
- [老式游戏rom](https://www.myabandonware.com/)

### 数学/IT原理
- [早期pytorch关于Tensor、Variable等的探讨](https://pytorch.org/blog/pytorch-0_4_0-migration-guide/#merging-tensor-and-variable-and-classes)
- [矩阵和向量的各种乘法](https://www.jianshu.com/p/9165e3264ced)
- [神经网络与记忆](https://www.bilibili.com/video/BV1fV4y1i7hZ/?spm_id_from=333.1007.0.0&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [陈天奇的机器学习编译课](https://www.bilibili.com/video/BV15v4y1g7EU/?is_story_h5=false&p=1&share_from=ugc&share_medium=android&share_plat=android&share_session_id=5a312434-ccf7-4cb9-862a-17a601cc4d35&share_source=COPY&share_tag=s_i&timestamp=1661386914&unique_k=zCWMKGC&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [基于梯度的机器学习IT原理](https://zhuanlan.zhihu.com/p/518198564)

### 开源示例
- [KAN 2.0](https://blog.csdn.net/qq_44681809/article/details/141355718)
- [radiate--衍生NEAT的纯Rust库](https://github.com/pkalivas/radiate)
- [neat-rs](https://github.com/dbsxdbsx/neat-rs)
- [纯Rust的NEAT+GRU](https://github.com/sakex/neat-gru-Rust)
- [Rusty_sr-纯Rust的基于dl的图像超清](https://github.com/millardjn/Rusty_sr)
- [ndarray_glm(可参考下`array!`，分布，以及原生的BLAS)](https://docs.rs/ndarray-glm/latest/ndarray_glm/)

- [PyToy--基于MatrixSlow的Python机器学习框架](https://github.com/ysj1173886760/PyToy)
- [MatrixSlow--纯python写的神经网络库](https://github.com/zc911/MatrixSlow)
- [python：遗传算法（GE）玩FlappyBird](https://github.com/ShuhuaGao/gpFlappyBird)

- [python包：遗传规划gplearn](https://gplearn.readthedocs.io/en/stable/examples.html)
- [python包：遗传规划deap](https://deap.readthedocs.io/en/master/examples/gp_symbreg.html)
- [python包：特征自动提取](https://github.com/IIIS-Li-Group/OpenFE)
- [NTK网络](https://zhuanlan.zhihu.com/p/682231092)

（较为成熟的3方库）
- [Burn—纯rust深度学习库](https://github.com/Tracel-AI/burn)
- [Candle:纯rust较成熟的机器学习库](https://github.com/huggingface/candle)
- [用纯numpy写各类机器学习算法](https://github.com/ddbourgin/numpy-ml)
（自动微分参考）
- [手工微分：Rust-CNN](https://github.com/goldstraw/RustCNN)
- [neuronika--纯Rust深度学习库（更新停滞了，参考下自动微分部分）](https://github.com/neuronika/neuronika)
- [基于TinyGrad的python深度学习库的RL示例](https://github.com/DHDev0/TinyRL/tree/main)
- [重点：Rust- ---支持cuda的Rust深度学习库(参考下自动微分部分)](https://docs.rs/dfdx/latest/dfdx/)
- [重点：基于ndarray的反向autoDiff库](https://github.com/raskr/rust-autograd)
- [前向autoDiff(貌似不成熟)](https://github.com/elrnv/autodiff)
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
- [动态蛇形卷积DSCNet](https://www.bilibili.com/video/BV1J84y1d7yG/?spm_id_from=333.1007.0.0&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [autoML介绍](https://www.zhihu.com/question/554255720/answer/2750670583)

### 符号派：逻辑/因果推断

- [scryer-prolog--Rust逻辑推理库](https://github.com/mthom/scryer-prolog)
- [vampire:自动证明器](https://github.com/vprover/vampire?tab=readme-ov-file)
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
- [知乎：因果推断杂谈](https://www.zhihu.com/question/266812683/answer/895210894)
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

### 强化学习

- [Sac用以复合Action](https://arxiv.org/pdf/1912.11077v1.pdf)
- [EfficientZero](https://arxiv.org/abs/2111.00210)
- [[EfficientZero Remastered](https://www.gigglebit.net/blog/efficientzero)]
- [SpeedyZero](https://openreview.net/forum?id=Mg5CLXZgvLJ)
- [LightZero系列](https://github.com/opendilab/LightZero?tab=readme-ov-file)
- [随机MuZero代码](https://github.com/DHDev0/Stochastic-muzero)
- [Redeeming Intrinsic Rewards via Constrained Optimization](https://williamd4112.github.io/pubs/neurips22_eipo.pdf)
- [Learning Reward Machines for Partially Observable Reinforcement Learning](https://arxiv.org/abs/2112.09477)
- [combo代码](https://github.com/Shylock-H/COMBO_Offline_RL)
- [2023最新model-based offline算法：MOREC](https://arxiv.org/abs/2310.05422)
- [众多model-base/free的offline算法](https://github.com/yihaosun1124/OfflineRL-Kit)
- [model-free offline算法：MCQ解析](https://zhuanlan.zhihu.com/p/588444380)
- [RL论文列表（curiosity、offline、uncertainty，safe）](https://github.com/yingchengyang/Reinforcement-Learning-Papers)
- [代替Gym的综合库](https://gymnasium.farama.org/)

### rust+大语言模型（LLM）

- [BionicGpt](https://github.com/bionic-gpt/bionic-gpt)
- [适用对话的Rust终端UI？](https://dustinblackman.com/posts/oatmeal/)
- [chatGpt相关论文](https://arxiv.org/abs/2203.02155)

### （自动、交互式）定理证明

- [关于lean的一篇文章](https://zhuanlan.zhihu.com/p/183902909#%E6%A6%82%E8%A7%88)
- [Lean+LLM](https://github.com/lean-dojo/LeanDojo)
- [陶哲轩使用Lean4](https://mp.weixin.qq.com/s/TYB6LgbhjvHYvkbWrEoDOg)

### 博弈论（game）

- [Sprague-Grundy介绍1](https://zhuanlan.zhihu.com/p/157731188)
- [Sprague-Grundy介绍2](https://zhuanlan.zhihu.com/p/20611132)
- [Sprague-Grundy介绍3](https://zhuanlan.zhihu.com/p/357893255)

### 其他

- [动手学深度学习-李沐著](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html#subsec-lin-algebra-norms)
- [openMMLab-Yolo](https://github.com/open-mmlab/mmyolo)
- [GRU解释](https://www.pluralsight.com/guides/lstm-versus-gru-units-in-rnn)
- [基于人类语音指挥的AI](https://arxiv.org/abs/1703.09831)
- [webGPT会上网的gpt](https://arxiv.org/abs/2112.09332)
- [LeCun的自监督世界模型](https://zhuanlan.zhihu.com/p/636997984)
- [awesome Rust](https://github.com/Rust-unofficial/awesome-Rust#genetic-algorithms)
- [去雾算法](https://blog.csdn.net/IT_job/article/details/78864236)
- [rust人工智能相关的项目](https://github.com/rust-unofficial/awesome-rust#artificial-intelligence)

## 遵循协议

本项目遵循MIT协议（简言之：不约束，不负责）。
