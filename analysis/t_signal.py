import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from model import t_signal_fn, T_Signal_Type


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
