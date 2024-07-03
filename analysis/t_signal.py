import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model import t_signal_fn, T_Signal_Type


# Define the function
# @torch.no_grad()
# def t_signal(input: torch.Tensor, fn: str = 'left') -> torch.Tensor:
#     def zero(x):
#         return torch.zeros_like(x)
#
#     def middle(x):
#         x = x / 999
#         # print(x)
#         assert (0 <= x).all()
#         assert (x <= 1).all()
#         x = x * 2 - 1
#         # print(x)
#         assert (-1 <= x).all()
#         assert (x <= 1).all()
#         mu = 0.0  # 均值
#         sigma = 0.1  # 标准差
#         normal_dist = torch.distributions.Normal(mu, sigma)
#         # x作为自变量，y作为因变量,函数形式是高斯分布
#         # y = torch.exp(-x ** 2 / 2) / (2 * np.pi) ** 0.5
#         y = normal_dist.log_prob(x).exp()
#         y = torch.clamp(y, 0, 1)
#         return y
#
#     def left(x):
#         assert (0 <= x).all()
#         assert (x <= 999).all()
#         x = torch.clamp(x, 0, 699) / 699.
#         y = (1 - x) ** 0.25
#         assert (0 <= y).all()
#         assert (y <= 1).all()
#         # x = torch.clamp(((700 - torch.clamp(x, 0, 700)) / 700) ** 0.25, 0, 1)
#         return y
#
#     def right(x):
#         assert (0 <= x).all()
#         assert (x <= 999).all()
#         x = (torch.clamp(x, 300, 999) - 300) / 699.
#         y = x ** 0.25
#         assert (0 <= y).all()
#         assert (y <= 1).all()
#         return y
#
#     FN = {
#         'left': left,
#         'middle': middle,
#         'right': right,
#         'zero': zero,
#     }
#
#     return FN[fn](input)


def draw(_t_signal_type: T_Signal_Type):
    # Generate x values
    x_values = torch.arange(0, 1000)
    # Calculate y values using the custom function
    y_values = t_signal_fn(x_values, _t_signal_type)

    area = torch.trapz(y_values, x_values).item() / 1000
    # area_discrete = y_values.sum()

    # Convert torch tensors to numpy arrays for plotting
    x_values_np = x_values.numpy()
    y_values_np = y_values.numpy()

    # Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_np, y_values_np, label='t_signal', color='blue')
    # plt.title('t = t_signal * t + (1 - t_signal) * torch.randn_like(t)')
    plt.title(f'{_t_signal_type.value}')
    plt.xlabel('t')
    plt.ylabel('t_signal')
    plt.fill_between(x_values_np, y_values_np, alpha=0.3)
    plt.text(
        {
            T_Signal_Type.zero: 500,
            T_Signal_Type.left: 300,
            T_Signal_Type.middle: 500,
            T_Signal_Type.right: 700
        }[_t_signal_type],
        {
            T_Signal_Type.zero: 0.02,
            T_Signal_Type.left: 0.35,
            T_Signal_Type.middle: 0.45,
            T_Signal_Type.right: 0.35
        }[_t_signal_type],
        f"{area * 100:.0f}%",
        horizontalalignment='center',
        verticalalignment='center', fontsize=25,
        color='blue',
        fontname='Comic Sans MS',
        bbox=dict(facecolor='white', alpha=0))
    plt.legend()
    # plt.grid(True)
    plt.savefig(f't_signal_{_t_signal_type.value}.png')
    plt.close()


if __name__ == "__main__":
    for t_type in tqdm(T_Signal_Type):
        draw(t_type)
