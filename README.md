## 描述
用纯rust（不想用c++）打造的仿造pytorch的个人玩具型AI框架（尚不成熟）。不打算支持gpu(因后期可能要支持安卓等平台，不想受制于某种非cpu设备)，但可能会加入NEAT等网络进化的算法。
### 名字由来
- only torch cpu --- 不用gpu，因要照顾多平台，也不想被某个GPU厂商制约（基于NEAT进化的网络结构也不太好被GPU优化）
- only torch node --- 没有全连接、卷积这类先入为主的网络结构，具体结构基于NEAT进化
- only torch f32 --- 网络的参数不需要除了f32外的数据结构

## 使用示例
（无）

## 文档
目前无人性化的文档。可直接看rust自动生成的[Api Doc](https://docs.rs/only_torch)即可。

## TODO
- [ ] 基于本框架解决XOR监督学习问题
- [ ] 基于本框架解决Mnist（数字识别）的监督学习问题
- [ ] 基于本框架解决CartPole（需要openAI Gym）的深度强化学习问题


## 参考资料
### 原理
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
- [awesome rust](https://github.com/rust-unofficial/awesome-rust#genetic-algorithms)

### 其他
- [动手学深度学习-李沐著](https://zh-v2.d2l.ai/chapter_preliminaries/linear-algebra.html#subsec-lin-algebra-norms)

## 遵循协议
本项目遵循MIT协议（简言之：不约束，不负责）。