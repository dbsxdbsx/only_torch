"""Generate PyTorch LR scheduler reference values for unit test comparison."""
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

model = torch.nn.Linear(10, 1)

# === CosineAnnealingLR ===
opt = optim.SGD(model.parameters(), lr=0.1)
s = sched.CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
print("CosineAnnealingLR (T_max=10, eta_min=0, lr=0.1):")
for epoch in range(10):
    s.step()
    lr = opt.param_groups[0]["lr"]
    print(f"  epoch {epoch+1}: lr={lr:.6f}")

# === CosineAnnealingLR with eta_min ===
opt = optim.SGD(model.parameters(), lr=0.1)
s = sched.CosineAnnealingLR(opt, T_max=10, eta_min=0.01)
print("\nCosineAnnealingLR (T_max=10, eta_min=0.01, lr=0.1):")
for epoch in range(10):
    s.step()
    lr = opt.param_groups[0]["lr"]
    print(f"  epoch {epoch+1}: lr={lr:.6f}")

# === StepLR ===
opt = optim.SGD(model.parameters(), lr=0.1)
s = sched.StepLR(opt, step_size=3, gamma=0.5)
print("\nStepLR (step_size=3, gamma=0.5, lr=0.1):")
for epoch in range(10):
    s.step()
    lr = opt.param_groups[0]["lr"]
    print(f"  epoch {epoch+1}: lr={lr:.6f}")

# === LambdaLR ===
opt = optim.SGD(model.parameters(), lr=0.1)
s = sched.LambdaLR(opt, lr_lambda=lambda epoch: 0.95 ** epoch)
print("\nLambdaLR (lambda: 0.95^epoch, lr=0.1):")
for epoch in range(10):
    s.step()
    lr = opt.param_groups[0]["lr"]
    print(f"  epoch {epoch+1}: lr={lr:.6f}")
