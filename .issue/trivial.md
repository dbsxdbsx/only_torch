for ```impl Module for ToySegmentationNet {
    fn parameters(&self) -> Vec<Var> {
        [
            self.conv1.parameters(),
            self.conv2.parameters(),
            self.head.parameters(),
        ]
        .concat()
    }
}``` 这种手写的模型定义，是否有办法通过其他一些优雅的方法把它处理掉？

不管是通过宏，或者说不用宏，有没有什么更加优雅的方式？