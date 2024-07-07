import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from schedule import ScheduleDDPM as Schedule
from utils import noise_like

schedule_fn = Schedule.schedule_fn
T = Schedule.T
batch_size = 100
device = 'cpu'
schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)

degree = torch.linspace(np.pi / 6, np.pi / 3, batch_size)

x_0_unscale = torch.cat(
    [torch.cos(degree).unsqueeze(-1), torch.sin(degree).unsqueeze(-1)],
    -1
).unsqueeze(-1).unsqueeze(-1).to(device)


def transform_and_draw(scale):
    global x_0_unscale
    x_0 = x_0_unscale * scale
    plt.figure(figsize=(6, 6))
    plt.title(f"Distribution Transition at Scale {scale}")
    # plt.xlabel("x")
    # plt.ylabel("y")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])

    rainbow_colors = [
        (1.0, 0.2, 0.2),  # Red
        (1.0, 0.6, 0.2),  # Orange
        (1.0 * 2 / 3, 1.0 * 2 / 3, 0.2 * 2 / 3),  # Yellow
        (0.2 * 2 / 3, 1.0 * 2 / 3, 0.2 * 2 / 3),  # Green
        (0.2, 0.2, 1.0),  # Blue
        (0.5, 0.2, 0.7),  # Indigo
        (0.7, 0.2, 1.0)  # Violet
    ]
    ts = [
        1,
        int(T // 5),  # 200
        int(T // 2),  # 500
        T - 300,  # 700
        T - int(T // 5),  # 800
        T - int(T // 10),  # 900
        T  # 1000
    ]
    for t, color in zip(ts, rainbow_colors):
        t = torch.full((batch_size,), fill_value=int(t), device=device)
        X_t = schedule.q_sample(x_0=x_0, t=t, noise=noise_like(x_0))
        x = []
        y = []
        for i, img in tqdm(enumerate(X_t), desc='plotting point...'):
            img = img.squeeze(-1).squeeze(-1).cpu().numpy()
            # if i == 0:
            #     for _ in range(1000):
            #         x.append(img[0])
            #         y.append(img[1])
            x.append(img[0])
            y.append(img[1])
        plt.scatter(x, y, color=color, s=1, alpha=0.2, label=f"t={t[0]}")
        plt.scatter([x[0]], [y[0]], color='black', s=2, alpha=1)
        plt.scatter([x[-1]], [y[-1]], color='red', s=2, alpha=1)
    # plt.legend()
    # Modify legend properties and place it outside the plot area
    legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), markerscale=10, scatterpoints=1)
    for handle, text, color in zip(legend.legend_handles, legend.get_texts(), rainbow_colors):
        handle.set_alpha(1)
        handle._sizes = [30]
        text.set_color(color)
    plt.savefig(f'distribution_trans_scale{scale}.png', bbox_inches='tight')
    plt.close()


for scale in tqdm(
        [2, 5, 7, 10, 30, 50, 70, 100],
        desc='draw distribution transition...',
        colour='green'
):
    transform_and_draw(scale)
