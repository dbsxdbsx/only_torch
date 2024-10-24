import torch

def step_gradient(x):
    # 定义阶跃函数的导数（在x=0处的导数定义为0）
    return torch.zeros_like(x)

def step_node(x):
    return (x >= 0).float()

# 使用自定义求导
class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return step_node(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0  # 阶跃函数的导数为0

# 创建测试用例
x = torch.tensor([[0.5, -1.0],
                  [0.0, 2.0]], requires_grad=True)

# 使用优化后的函数
step = StepFunction.apply
y = step(x)

# 计算梯度
y.sum().backward()

print("x的维度:", x.shape)
print("y的值:\n", y)
print("y的维度:", y.shape)
print("x的梯度:\n", x.grad)