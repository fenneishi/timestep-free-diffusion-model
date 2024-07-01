import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings
from typing import Callable

warnings.filterwarnings("ignore", category=UserWarning, module='torch_fidelity.datasets')

import torch
import torch_fidelity
from einops import repeat
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm
import wandb

# from dataset_CIFAR10 import test_data, training_data, channels, image_size
from dataset_FashionMNIST import test_data, channels, image_size

from schedule import ScheduleDDPM as Schedule
from utils import num_to_groups, clamp

evaluate_folder = Path("./evaluate2").absolute()
fake_folder = Path("./evaluate2/fake").absolute()
real_folder = Path("./evaluate2/real").absolute()
FakeImgsCount = 10000

vmeory = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))


@torch.no_grad()
def build_fake_data(model: torch.nn.Module | Callable):
    assert torch.cuda.is_available()

    # config
    schedule_fn = Schedule.schedule_fn
    T = Schedule.T

    # clear fake folder
    global fake_folder
    if fake_folder.exists() and fake_folder.is_dir():
        shutil.rmtree(fake_folder)
        # subprocess.call(['rm', '-rf', fake_folder])
    os.makedirs(fake_folder, exist_ok=False)

    # schedule
    schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)

    # batch sizes
    batch_sizes = num_to_groups(FakeImgsCount, 1024 * 4)

    # generate images and save them
    def gen_imgs(batch_size: int) -> list[torch.Tensor]:
        res = schedule.p_sample_loop(
            model=model,
            shape=(batch_size, channels, image_size, image_size)
        )
        # batch_diffusion_imgs = [[bc[i] for bc in res] for i in range(64)]
        # batch_diffusion_imgs = [(torch.stack(i) + 1) * 0.5 for i in batch_diffusion_imgs]
        # for i, img in tqdm(enumerate(batch_diffusion_imgs)):
        #     save_image(img, os.path.join(results_folder, f'diffusion_{i}.png'), nrow=50)
        imgs = res[-1]
        imgs = (imgs + 1) / 2
        if channels == 1:
            imgs = repeat(imgs, 'b 1 h w -> b 3 h w')
        return [img for img in imgs]

    def save_img(name_img_tuple: tuple[str | int, torch.Tensor]) -> None:
        name, img = name_img_tuple
        save_image(img, os.path.join(fake_folder, f'{name}.png'))

    fake_imgs = []
    for b in tqdm(batch_sizes, desc='Generating images'):
        fake_imgs += gen_imgs(b)

    # with ThreadPoolExecutor() as executor:
    #     save_tasks = [(f'gen_{i:07d}', img) for i, img in enumerate(fake_imgs)]
    #     executor.map(save_img, save_tasks)
    save_tasks = [(f'gen_{i:07d}', img) for i, img in enumerate(fake_imgs)]
    for name, img in tqdm(save_tasks, desc='Saving images'):
        save_image(img, os.path.join(fake_folder, f'{name}.png'))

    print(f"fake images saved to {fake_folder}")

    return fake_imgs


class EvalDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.imgs = test_data

    @torch.no_grad()
    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = (((img + 1) / 2) * 255).to(torch.uint8).clamp_(0, 255)
        if channels == 1:
            img = repeat(img, '1 h w  -> 3 h w')
        return img

    def __len__(self):
        return len(self.imgs)


@torch.no_grad()
def build_real_data():
    global real_folder
    real_imgs = []

    if real_folder.exists() and real_folder.is_dir():
        return
    real_folder.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(test_data)), desc='Loading real images'):
        img, label = test_data[i]
        img = (img + 1.) / 2.
        real_imgs.append(img)

    def save_img(index_img_tuple):
        index, im = index_img_tuple
        save_image(im, os.path.join(real_folder, f'{index}.png'))

    with ThreadPoolExecutor() as executor:
        executor.map(save_img, tqdm(enumerate(real_imgs), desc='Saving real images'))


def evaluate(model: torch.nn.Module | Callable, step: int = 0):
    print(f"evaluating at step {step}")

    fake_imgs = build_fake_data(model)

    # visualize fake images
    visualize = os.path.join(evaluate_folder, f"sample_{step}.png")
    save_image(torch.stack(fake_imgs[:64]), visualize)
    if wandb.run is not None:
        wandb.log({"fake_imgs": [wandb.Image(visualize)]}, step=step)
    del fake_imgs

    build_real_data()

    metrics_dict = torch_fidelity.calculate_metrics(
        # fake data
        input1=str(fake_folder),
        # real data
        input2=str(real_folder),
        # options
        verbose=True,
        cuda=True,
        batch_size=clamp(64 * int(vmeory // 8), 64, 256),
        # metrics
        isc=True,
        fid=True,
        prc=True,
        kid=False,
    )
    print(metrics_dict)
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)
    # {'inception_score_mean': 4.148801176711407, 'inception_score_std': 0.041404909234431596,
    #  'frechet_inception_distance': 34.29237695877276, 'precision': 0.4097000062465668, 'recall': 0.7635999917984009,
    #  'f_score': 0.5332769486592858}


if __name__ == "__main__":
    from model import Unet, how_to_t, HowTo_t

    pretrain_model_name = 'FashionMNIST_predict_t_0701_1612_step14040.pth'
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        out_dim=channels + 1 if how_to_t == HowTo_t.predict_t else None
    )
    model.load_state_dict(torch.load(pretrain_model_name))
    model = model.cuda()  # .to("cuda")
    print(f"model loaded from {pretrain_model_name}")


    @torch.no_grad()
    def call_model(*args, **kwargs):
        predicted: torch.Tensor = model(*args, **kwargs)
        if how_to_t == HowTo_t.predict_t:
            predicted, predicted_t = predicted[:, :-1, :, :], predicted[:, -1:, :, :]
            # print(f"predicted_t: {1- (predicted_t.mean().item() + 1) / 2}")
        return predicted


    print(f"model loaded from {pretrain_model_name}")
    model.eval()
    evaluate(call_model)

# T=1000,input with t,learning rate decay 1e-3 1e-4 1e-5 1e-6,more train
# {
#     'inception_score_mean': 4.250638325513722,
#     'inception_score_std': 0.11893020987160077,
#     'frechet_inception_distance': 8.519418311322795,
#     'precision': 0.5543000102043152,
#     'recall': 0.786899983882904,
#     'f_score': 0.6504304667745237
# }
