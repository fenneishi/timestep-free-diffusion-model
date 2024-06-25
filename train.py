import logging
import warnings
import os
from pathlib import Path
import shutil
import datetime

import torch
from einops import rearrange
from torch.optim import Adam
from torchvision.utils import save_image

from tqdm import tqdm
import wandb

from model import Unet, save_model_name, pretrain_model_name, how_to_t, HowTo_t
from schedule import ScheduleDDPM as Schedule
from loss import loss_f
# from dataset_CIFAR10 import build_data, image_size, channels
from dataset_FashionMNIST import build_data, image_size, channels
from utils import num_to_groups, noise_like

results_folder = Path("./results").absolute()
if results_folder.exists() and results_folder.is_dir():
    shutil.rmtree(results_folder)
if not results_folder.exists():
    os.makedirs(results_folder)

assert torch.cuda.is_available()
device = "cuda"
epochs = 6 * 3
T = Schedule.T
batch_size = 128  # 64
learning_rate = 1e-3
schedule_fn = Schedule.schedule_fn
save_and_sample_every = 1000 // 1

wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="timestep-free-diffusion-model",
    entity="fenneishi",
    name=save_model_name(f'withpretrain_time{datetime.datetime.now().strftime("%H%M")}'),
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "T": T,
        "schedule_fn": schedule_fn,
        "save_and_sample_every": save_and_sample_every,
        "model": "Unet",
        "image_size": image_size,
        "channels": channels,
        "how_to_t": how_to_t,
        "pretrain_model_name": pretrain_model_name,
        "save_model_name": save_model_name(),
    },
)

print(
    f'######################################\n'
    f'epochs: {epochs}\n'
    f'T: {T}\n'
    f'batch_size: {batch_size}\n'
    f'learning_rate: {learning_rate} \n'
    f'pretrain_model_name: {pretrain_model_name}\n'
    f'save_model_name: {save_model_name()}\n'
    f'how_to_t: {how_to_t}\n'
    f'channels: {channels}\n'
    f'image_size: {image_size}\n'
    f'schedule_fn: {schedule_fn}\n'
    f'\n######################################'
)

schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    out_dim=channels + 1 if how_to_t == HowTo_t.predict_t else None
).to(device)
# load the model from pretrained
if pretrain_model_name is not None:
    model.load_state_dict(torch.load(pretrain_model_name))

optimizer = Adam(model.parameters(), lr=learning_rate)
dataloader = build_data(batch_size=batch_size, train=True)


def sample(b: torch.Tensor) -> torch.Tensor:
    def call_model(*args, **kwargs):
        predicted: torch.Tensor = model(*args, **kwargs)
        if how_to_t == HowTo_t.predict_t:
            predicted_noise, predicted_t = predicted[:, :-1, :, :], predicted[:, -1:, :, :]
            predicted_t_show = ((predicted_t.mean().item() + 1) / 2) * 1000
            predicted_t_show = (1000 - predicted_t_show) / 1000
            # to percentage
            predicted_t_show = int(predicted_t_show * 100)
            print(f"predicted_t: {predicted_t_show}%")
            predicted = predicted_noise
        return predicted

    res = schedule.p_sample_loop(call_model, shape=(b, channels, image_size, image_size))
    # batch_diffusion_imgs = [[bc[i] for bc in res] for i in range(64)]
    # batch_diffusion_imgs = [(torch.stack(i) + 1) * 0.5 for i in batch_diffusion_imgs]
    # for i, img in tqdm(enumerate(batch_diffusion_imgs)):
    #     save_image(img, os.path.join(results_folder, f'diffusion_{i}.png'), nrow=50)

    return res[-1]


step = 0


def evaluate(force=False):
    if (step != 0 and step % save_and_sample_every == 0) or force:
        print(f"evaluating at step {step}")
        images = [sample(b) for b in num_to_groups(64, batch_size)]
        images = (torch.cat(images) + 1) * 0.5
        save_image(
            images,
            os.path.join(
                results_folder,
                f"sample_{step // save_and_sample_every if not force else 'final'}.png"
            )
        )


# evaluate(force=True)


def save_model():
    if save_model_name is not None:
        model_name = save_model_name(step)
        torch.save(model.state_dict(), model_name)
        print("Epoch", epoch, f"completed,saving model to {model_name}")


loss_log = None
for epoch in range(epochs):
    for x_0, _ in dataloader:
        optimizer.zero_grad()

        # x_0
        x_0 = x_0.to(device)
        b, c, h, w = x_0.shape
        if not (c == channels and h == w == image_size):
            raise ValueError(f"image size is not {channels}x{image_size}x{image_size}, but {c}x{h}x{w}")
        if not (b == batch_size):
            raise ValueError(f"batch size is not {batch_size}, but {b}")

        # t
        t = schedule.uniform_t_sample(b, device=device)

        # X_t
        noise = noise_like(x_0)
        X_t = schedule.q_sample(x_0=x_0, t=t, noise=noise)

        # predict noise
        predicted: torch.Tensor = model(X_t, t)
        if how_to_t == HowTo_t.predict_t:
            predicted_noise, predicted_t = predicted[:, :-1, :, :], predicted[:, -1:, :, :]
        else:
            predicted_noise, predicted_t = predicted, None

        # loss
        loss = loss_f(
            noise=noise, predicted_noise=predicted_noise,
            step=step,
            t=t, predicted_t=predicted_t,
            loss_type="huber"
        )
        # if predicted_t is not None:
        # noise_loss, loss = loss
        # wandb.log(data={"loss": noise_loss.item()}, step=step)
        # else:
        #     wandb.log(data={"loss": loss.item()}, step=step)

        # loss_log = loss.item() if loss_log is None else 0.99 * loss_log + 0.01 * loss.item()
        # if step % 10 == 0:
        #     print(f"smooth loss at step {step}:{loss_log:.5f}[{loss.item():.5f}]")
        loss.backward()
        optimizer.step()
        evaluate()
        step += 1
    save_model()

save_model()
evaluate(force=True)
wandb.finish()
