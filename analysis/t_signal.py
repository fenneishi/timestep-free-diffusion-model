import matplotlib.pyplot as plt
import torch
import numpy as np


# Define the function
@torch.no_grad()
def t_signal(x: torch.Tensor) -> torch.Tensor:
    x = x / 999
    # print(x)
    assert (0 <= x ).all()
    assert (x <= 1).all()
    x = x * 2 - 1
    # print(x)
    assert (-1 <= x).all()
    assert (x <= 1).all()
    mu = 0.0  # 均值
    sigma = 0.1  # 标准差
    normal_dist = torch.distributions.Normal(mu, sigma)
    # x作为自变量，y作为因变量,函数形式是高斯分布
    # y = torch.exp(-x ** 2 / 2) / (2 * np.pi) ** 0.5
    y = normal_dist.log_prob(x).exp()
    y = torch.clamp(y, 0, 1)

    # x = torch.clamp(700 - x, 0, 700)
    # x = x / 700.
    # x = x ** 0.25
    # x = torch.clamp(x, 0, 1)

    # x = torch.clamp(((700 - torch.clamp(x, 0, 700)) / 700) ** 0.25, 0, 1)
    return y


# Generate x values
# x_values = torch.linspace(0, 999, 1000)
x_values = torch.arange(0, 1000)
print(x_values)
# Calculate y values using the custom function
y_values = t_signal(x_values)

area = torch.trapz(y_values, x_values) / 1000
# area_discrete = y_values.sum()

# Convert torch tensors to numpy arrays for plotting
x_values_np = x_values.numpy()
y_values_np = y_values.numpy()

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(x_values_np, y_values_np, label='t_signal', color='blue')
plt.title('t = t_signal * t + (1 - t_signal) * torch.randn_like(t)')
plt.xlabel('t')
plt.ylabel('t_signal')
plt.fill_between(x_values_np, y_values_np, alpha=0.3)
plt.text(300, 0.45,
         f"{area * 100:.0f}%",
         horizontalalignment='center',
         verticalalignment='center', fontsize=25,
         color='blue',
         fontname='Comic Sans MS',
         bbox=dict(facecolor='white', alpha=0))
plt.legend()
# plt.grid(True)
plt.savefig('t_signal.png')
plt.close()
