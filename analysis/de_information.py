import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.lines as mlines

from schedule import ScheduleDDPM as Schedule
from utils import noise_like
from torchvision.utils import save_image

channels = 3
image_size = 32
schedule_fn = Schedule.schedule_fn
T = Schedule.T
batch_size = 1
device = 'cpu'
schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)

data = datasets.CIFAR10(
    root="../.data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda t: (t * 2) - 1)
    ])
)
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=True)


def schedule_show(images: list[torch.Tensor], ts: list[int], _: int):
    # images = [img[0, 0, int(image_size // 2), :].cpu().numpy() for img in images]
    b, c, h, w = images[0].shape
    images = [F.interpolate(img, size=(6, 6), mode='bilinear', align_corners=False)
              for img in images]
    # images = [img  for img in images]
    images = [img.permute(0, 2, 3, 1) for img in images]
    images = [img.flatten().cpu().numpy() for img in images]
    _means = [img.mean() for img in images]
    _stds = [img.std() for img in images]
    _maxs = [mean + std / 2. for mean, std in zip(_means, _stds)]
    _mins = [mean - std / 2. for mean, std in zip(_means, _stds)]

    plt.figure(figsize=(20, 10))
    offset = 0
    spacing = len(images[0])  # int(len(images[0]) // 2)
    colors = ['r', 'g', 'b']
    plt.ylim(-1, 1)
    # fig_range = [(0, 1 / 6), (2 / 6, 3 / 6), (4 / 6, 5 / 6)]
    sub_fig_range = (len(images[0]) + spacing) * len(images)
    fig_range = [
        (
            (len(img) + spacing) * i / sub_fig_range,
            (len(img) + spacing) * (i + 1) / sub_fig_range
        ) for i, img
        in enumerate(images)
    ]
    fig_range = [
        (f[0] + (f[1] - f[0]) / 4.0, f[1] - (f[1] - f[0]) / 4.0)
        for f in fig_range
    ]
    # plt.xlim(left=0, right=sub_fig_range)
    for i, (t, img) in enumerate(zip(ts, images)):
        x = offset + np.arange(len(img))
        # plt.plot(x, img, label=f"t={t}", color=colors[i], alpha=0.1)
        plt.axhline(y=_means[i],
                    xmin=fig_range[i][0],
                    xmax=fig_range[i][1],
                    color=colors[i], linestyle='--',
                    label=f"mean at t={t}={_means[i]:.2f}")
        plt.axhline(y=_maxs[i],
                    xmin=fig_range[i][0],
                    xmax=fig_range[i][1],
                    linestyle='solid',
                    color=colors[i],
                    label=f"max at t={t}={_maxs[i]:.2f}")
        plt.axhline(y=_mins[i],
                    xmin=fig_range[i][0],
                    xmax=fig_range[i][1],
                    color=colors[i],
                    linestyle='solid',
                    label=f"min at t={t}={_mins[i]:.2f}")
        offset += len(img) + spacing

    plt.axhline(y=0, color='black', linewidth=0.5)  # Adding the x-axis at y=0
    plt.gca().set_xticks([])  # Removing x-axis labels
    plt.title('Vector Image Waveform')
    # plt.xlabel('Index')
    plt.ylabel('Value')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # Create custom legends
    color_handles = [mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10, label=f't={t}') for
                     color, t in zip(colors, ts)]
    linestyle_handles = [
        mlines.Line2D([], [], color='black', linestyle='--', label='mean'),
        mlines.Line2D([], [], color='black', linestyle='-', label='std')
    ]

    # First legend: for colors
    legend1 = plt.legend(handles=color_handles, bbox_to_anchor=(1.01, 1), loc='upper left', title="Time Steps")
    plt.gca().add_artist(legend1)

    # Second legend: for line styles
    plt.legend(handles=linestyle_handles, bbox_to_anchor=(1.01, 0.85), loc='upper left', title="Statistics")
    plt.grid(False)
    plt.savefig(f'schedule_{_}.png', bbox_inches='tight')
    plt.close()
    # plt.show()


def denoise_show(images: list[torch.Tensor], ts: list[int], _: int):
    # images = [img[0, 0, int(image_size // 2), :].cpu().numpy() for img in images]
    b, c, h, w = images[0].shape
    images = [F.interpolate(img, size=(6, 6), mode='bilinear', align_corners=False)
              for img in images]
    # images = [img  for img in images]
    images = [img.permute(0, 2, 3, 1) for img in images]
    images = [img.flatten().cpu().numpy() for img in images]
    _means = [img.mean() for img in images]
    _stds = [img.std() for img in images]
    images = [(img - mean) / std for img, mean, std in zip(images, _means, _stds)]
    # _maxs = [mean + std / 2. for mean, std in zip(_means, _stds)]
    # _mins = [mean - std / 2. for mean, std in zip(_means, _stds)]

    plt.figure(figsize=(20, 10))
    offset = 0
    spacing = len(images[0])  # int(len(images[0]) // 2)
    colors = ['r', 'g', 'b']
    plt.ylim(-3, 3)
    # fig_range = [(0, 1 / 6), (2 / 6, 3 / 6), (4 / 6, 5 / 6)]
    sub_fig_range = (len(images[0]) + spacing) * len(images)
    fig_range = [
        (
            (len(img) + spacing) * i / sub_fig_range,
            (len(img) + spacing) * (i + 1) / sub_fig_range
        ) for i, img
        in enumerate(images)
    ]
    fig_range = [
        (f[0] + (f[1] - f[0]) / 4.0, f[1] - (f[1] - f[0]) / 4.0)
        for f in fig_range
    ]
    # plt.xlim(left=0, right=sub_fig_range)
    for i, (t, img) in enumerate(zip(ts, images)):
        x = offset + np.arange(len(img))
        plt.plot(x, img, label=f"t={t}", color=colors[i], alpha=0.1)
        # plt.axhline(y=_means[i],
        #             xmin=fig_range[i][0],
        #             xmax=fig_range[i][1],
        #             color=colors[i], linestyle='--',
        #             label=f"mean at t={t}={_means[i]:.2f}")
        # plt.axhline(y=_maxs[i],
        #             xmin=fig_range[i][0],
        #             xmax=fig_range[i][1],
        #             linestyle='solid',
        #             color=colors[i],
        #             label=f"max at t={t}={_maxs[i]:.2f}")
        # plt.axhline(y=_mins[i],
        #             xmin=fig_range[i][0],
        #             xmax=fig_range[i][1],
        #             color=colors[i],
        #             linestyle='solid',
        #             label=f"min at t={t}={_mins[i]:.2f}")
        offset += len(img) + spacing

    plt.axhline(y=0, color='black', linewidth=0.5)  # Adding the x-axis at y=0
    plt.gca().set_xticks([])  # Removing x-axis labels
    plt.title('Vector Image Waveform')
    # plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.savefig(f'denoise_{_}.png', bbox_inches='tight')
    plt.close()
    # plt.show()


def denoise_schedule_show(images: list[torch.Tensor], ts: list[int], _: int):
    # images = [img[0, 0, int(image_size // 2), :].cpu().numpy() for img in images]
    b, c, h, w = images[0].shape
    images = [F.interpolate(img, size=(6, 6), mode='bilinear', align_corners=False)
              for img in images]
    # images = [img  for img in images]
    images = [img.permute(0, 2, 3, 1) for img in images]
    images = [img.flatten().cpu().numpy() for img in images]
    _means = [img.mean() for img in images]
    _stds = [img.std() for img in images]
    _maxs = [mean + std / 2. for mean, std in zip(_means, _stds)]
    _mins = [mean - std / 2. for mean, std in zip(_means, _stds)]

    plt.figure(figsize=(20, 10))
    offset = 0
    spacing = len(images[0])  # int(len(images[0]) // 2)
    colors = ['r', 'g', 'b']
    linestyles = ['--', 'solid']
    plt.ylim(-1, 1)
    # fig_range = [(0, 1 / 6), (2 / 6, 3 / 6), (4 / 6, 5 / 6)]
    sub_fig_range = (len(images[0]) + spacing) * len(images)
    fig_range = [
        (
            (len(img) + spacing) * i / sub_fig_range,
            (len(img) + spacing) * (i + 1) / sub_fig_range
        ) for i, img
        in enumerate(images)
    ]
    fig_range = [
        (f[0] + (f[1] - f[0]) / 4.0, f[1] - (f[1] - f[0]) / 4.0)
        for f in fig_range
    ]
    # plt.xlim(left=0, right=sub_fig_range)
    for i, (t, img) in enumerate(zip(ts, images)):
        x = offset + np.arange(len(img))
        plt.plot(x, img, label=f"t={t}", color=colors[i], alpha=0.1)
        plt.axhline(
            y=_means[i],
            xmin=fig_range[i][0],
            xmax=fig_range[i][1],
            color=colors[i],
            linestyle=linestyles[0],
            label=f"mean"
        )
        plt.axhline(
            y=_maxs[i],
            xmin=fig_range[i][0],
            xmax=fig_range[i][1],
            linestyle=linestyles[1],
            color=colors[i],
            label=f"std"
        )
        plt.axhline(
            y=_mins[i],
            xmin=fig_range[i][0],
            xmax=fig_range[i][1],
            color=colors[i],
            linestyle=linestyles[1],
            label=f"std"
        )
        plt.fill_between(x, _mins[i], _maxs[i], color=colors[i], alpha=0.05, label='Standard Deviation')
        offset += len(img) + spacing

    plt.axhline(y=0, color='black', linewidth=0.5)  # Adding the x-axis at y=0
    plt.gca().set_xticks([])  # Removing x-axis labels
    plt.title('Vector Image Waveform')
    # plt.xlabel('Index')
    plt.ylabel('Value')

    # Create custom legends
    color_handles = [mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10, label=f't={t}') for
                     color, t in zip(colors, ts)]
    linestyle_handles = [
        mlines.Line2D([], [], color='black', linestyle='--', label='mean'),
        mlines.Line2D([], [], color='black', linestyle='-', label='std')
    ]

    # First legend: for colors
    legend1 = plt.legend(handles=color_handles, bbox_to_anchor=(1.01, 1), loc='upper left', title="Time Steps")
    plt.gca().add_artist(legend1)

    # Second legend: for line styles
    plt.legend(handles=linestyle_handles, bbox_to_anchor=(1.01, 0.85), loc='upper left', title="Statistics")

    plt.grid(False)
    plt.savefig(f'denoise_schedule_show_{_}.png', bbox_inches='tight')
    plt.close()
    # plt.show()


for i, (x_0, _) in tqdm(enumerate(dataloader)):
    imgs = []
    ts = [T, int(T // 5), 1]

    for t in ts:
        t = torch.full((batch_size,), fill_value=int(t), device=device)
        X_t = schedule.q_sample(x_0=x_0, t=t, noise=noise_like(x_0))
        imgs.append(X_t)
    _ = _.item()
    denoise_schedule_show(imgs, ts, i)
    schedule_show(imgs, ts, i)
    denoise_show(imgs, ts, i)
    save_image(x_0, f'X_{i}.png')
    img = PIL.Image.Image.open(f'X_{i}.png')

    if i > 10:
        break
