import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from schedule import ScheduleDDPM as Schedule
from utils import noise_like

channels = 1
image_size = 28
schedule_fn = Schedule.schedule_fn
T = Schedule.T
batch_size = 1
device = 'cpu'
schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)

data = datasets.FashionMNIST(
    root="../.data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)


def img_wave(images: list[torch.Tensor], ts: list[int], _: int):
    images = [img[0, 0, int(image_size // 2), :].cpu().numpy() for img in images]
    _means = [img.mean() for img in images]
    _maxs = [np.max(img) for img in images]
    _mins = [np.min(img) for img in images]

    plt.figure(figsize=(10, 5))
    offset = 0
    spacing = int(len(images[0]) // 2)
    colors = ['r', 'g', 'b']
    for i, (t, img) in enumerate(zip(ts, images)):
        x = np.arange(len(img)) + offset
        plt.plot(x, img, label=f"t={t}")
        plt.axhline(y=_means[i], color=colors[i], linestyle='--', label=f"mean at t={t}={_means[i]:.2f}")
        plt.axhline(y=_maxs[i], color=colors[i], linestyle='solid', label=f"max at t={t}={_maxs[i]:.2f}")
        plt.axhline(y=_mins[i], color=colors[i], linestyle='dashdot', label=f"min at t={t}={_mins[i]:.2f}")
        offset += len(img) + spacing

    plt.title('Vector Image Waveform')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'vector_img_waveform{_}.png')
    # plt.show()


for x_0, _ in dataloader:
    imgs = []
    ts = [T, int(T // 2), 1]
    for t in ts:
        t = torch.full((batch_size,), fill_value=int(t), device=device)
        X_t = schedule.q_sample(x_0=x_0, t=t, noise=noise_like(x_0))
        imgs.append(X_t)
    img_wave(imgs, ts, _)
    if _ > 100:
        break
#
# for t in [1, int(T // 10), T]:
#     t = torch.full((batch_size,), fill_value=int(t), device=device)
#     X_t = schedule.q_sample(x_0=x_0, t=t, noise=noise_like(x_0))
#     imgs.append(X_t)
