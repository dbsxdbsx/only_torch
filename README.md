## 描述
用纯rust（不想用c++）打造的仿造pytorch的个人玩具型AI框架（尚不成熟,请勿使用）。不打算支持gpu(因后期可能要支持安卓等平台，不想受制于某种非cpu设备)，但可能会加入NEAT等网络进化的算法。
### 名字由来
一部分原因是受到pytorch的影响，希望能写个和pytorch一样甚至更易用的AI框架；另一部分是希望本框架只触及（touch）一些关键的东西：
- only torch rust --- 只用rust（不用c++是因为其在复杂逻辑项目中容易写出内存不安全代码）；也不用第三方lib（所以排除[tch-rs](https://github.com/LaurentMazare/tch-rs)），这样对跨平台支持会比较友好。
- only torch cpu --- 不用gpu，因要照顾多平台，也不想被某个GPU厂商制约，且基于NEAT进化的网络结构也不太好被GPU优化；如此也省得考虑数据从cpu内存迁移到其他设备内存的开销问题了。
- only torch node --- 没有全连接、卷积、resnet这类先入为主的算子概念，具体模型结构均基于NEAT进化。
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

## 参考资料
### IT原理
- [陈天奇的机器学习编译课](https://www.bilibili.com/video/BV15v4y1g7EU/?is_story_h5=false&p=1&share_from=ugc&share_medium=android&share_plat=android&share_session_id=5a312434-ccf7-4cb9-862a-17a601cc4d35&share_source=COPY&share_tag=s_i&timestamp=1661386914&unique_k=zCWMKGC&vd_source=3facc3cb195be0a27a0ea9a4eb3bb6fe)
- [基于梯度的机器学习IT原理](https://zhuanlan.zhihu.com/p/518198564)
### 开源示例
- [PyToy--基于MatrixSlow的建议python机器学习框架](https://github.com/ysj1173886760/PyToy)
- [深度学习框架InsNet简介](https://zhuanlan.zhihu.com/p/378684569)
- [neuronika--纯rust深度学习库（更新停滞了）](https://github.com/neuronika/neuronika)
- [经典机器学习算法rust库](https://github.com/rust-ml/linfa)
- [peroxide--纯rust的线代及周边库](https://crates.io/crates/peroxide)
- [radiate--衍生NEAT的纯rust库](https://github.com/pkalivas/radiate)
- [纯rust的NEAT+GRU](https://github.com/sakex/neat-gru-rust)
- [C++实现的NEAT+LSTM/GRU/CNN](https://github.com/travisdesell/exact)
- [pytorch+NEAT](https://github.com/ddehueck/pytorch-neat)
- [avalog--基于avatar的rust逻辑推理库](https://crates.io/crates/avalog)
- [awesome rust](https://github.com/rust-unofficial/awesome-rust#genetic-algorithms)
### 逻辑推理
- [scryer-prolog--rust逻辑推理库](https://github.com/mthom/scryer-prolog)
- [那迷人的被遗忘的语言：Prolog](https://zhuanlan.zhihu.com/p/41908829)
- [结合prolog和RL](https://arxiv.org/abs/2004.06997)
- [prolog与4证人难题](https://prolog.longluntan.com/t9-topic)
- [logic+mL提问](https://ai.stackexchange.com/questions/16224/has-machine-learning-been-combined-with-logical-reasoning-for-example-prolog)- [prolog解决数度问题](https://prolog.longluntan.com/t107-topic)
### 其他
- [动手学深度学习-李沐著](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html#subsec-lin-algebra-norms)
- [基于人类语音指挥的AI](https://arxiv.org/abs/1703.09831)
## 遵循协议
本项目遵循MIT协议（简言之：不约束，不负责）。