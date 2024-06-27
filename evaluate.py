import os
import shutil
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

from model import Unet, pretrain_model_name, how_to_t, HowTo_t
from schedule import ScheduleDDPM as Schedule
from utils import num_to_groups, clamp

evaluate_folder = Path("./evaluate").absolute()
fake_folder = Path("./evaluate/fake").absolute()
real_folder = Path("./evaluate/real").absolute()
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
    for b in tqdm(batch_sizes, desc='Generating and saving generated images'):
        fake_imgs += gen_imgs(b)

    with ThreadPoolExecutor() as executor:
        save_tasks = [(f'gen_{i:07d}', img) for i, img in enumerate(fake_imgs)]
        executor.map(save_img, save_tasks)

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
    wandb.log(metrics_dict, step=step)
    # {'inception_score_mean': 4.148801176711407, 'inception_score_std': 0.041404909234431596,
    #  'frechet_inception_distance': 34.29237695877276, 'precision': 0.4097000062465668, 'recall': 0.7635999917984009,
    #  'f_score': 0.5332769486592858}
    print(metrics_dict)


if __name__ == "__main__":
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,),
        out_dim=channels + 1 if how_to_t == HowTo_t.predict_t else None
    ).to("cuda")
    model.load_state_dict(torch.load(pretrain_model_name))
    print(f"model loaded from {pretrain_model_name}")


    @torch.no_grad()
    def call_model(*args, **kwargs):
        predicted: torch.Tensor = model(*args, **kwargs)
        if how_to_t == HowTo_t.predict_t:
            predicted, predicted_t = predicted[:, :-1, :, :], predicted[:, -1:, :, :]
            # print(f"predicted_t: {1- (predicted_t.mean().item() + 1) / 2}")
        return predicted


    print(f"model loaded from {pretrain_model_name}")
    evaluate(call_model)

"""
T=300,input with t
{
    'inception_score_mean': 4.012789100944213, 
    'inception_score_std': 0.0832336997197499, 
    'frechet_inception_distance': 30.942926437637368, 
    'precision': 0.4952999949455261, 
    'recall': 0.6886000037193298, 
    'f_score': 0.576169573015133
}
T=1000,input with t
{
    'inception_score_mean': 4.2865708800405775, 
    'inception_score_std': 0.08070050760798674, 
    'frechet_inception_distance': 28.816080882236122, 
    'precision': 0.35830000042915344,
    'recall': 0.7106999754905701, 
    'f_score': 0.47641498084073497
}
T=1000,input{   
    'inception_score_mean': 3.9877652116655478, 
    'inception_score_std': 0.08303174412055173, 
    'frechet_inception_distance': 20.092534537450263, 
    'precision': 0.49300000071525574, 'recall': 0.7110999822616577, 
    'f_score': 0.582297644248596
} without t

T=1000,input with t,dataset_CIFAR10
{
    'inception_score_mean': 4.489414021731528, 
    'inception_score_std': 0.10001167169569403, 
    'frechet_inception_distance': 81.06440838228718, 
    'precision': 0.040800001472234726, 
    'recall': 0.7907000184059143, 
    'f_score': 0.07759605807293884
}
T=1000,input with t,dataset_CIFAR10,
{   
    'inception_score_mean': 5.91294293357287, 
    'inception_score_std': 0.0918314265547397, 
    'frechet_inception_distance': 47.74804861197089, 
    'precision': 0.17589999735355377, 
    'recall': 0.8133999705314636, 
    'f_score': 0.2892490797705039
}
T=1000,input with t,learning rate 1e-3 1e-4 1e-5 1e-6,more train
{
    'inception_score_mean': 4.250638325513722, 
    'inception_score_std': 0.11893020987160077, 
    'frechet_inception_distance': 8.519418311322795, 
    'precision': 0.5543000102043152, 
    'recall': 0.786899983882904, 
    'f_score': 0.6504304667745237
}
T=1000,input without(70% **0.25) t,learning rate 1e-3,
            t_signal = torch.clamp(((700 - torch.clamp(t_signal,0,700)) / 700) ** 0.25, 0, 1).detach()
            t_noise = torch.randn_like(t).detach()  # remove this line to use the time embeddings
            t = t_signal*t + (1 - t_signal)*t_noise
{
    'inception_score_mean': 4.171455520414709, 
    'inception_score_std': 0.04308712302956141, 
    'frechet_inception_distance': 24.81214031140746, 
    'precision': 0.40689998865127563, 
    'recall': 0.7175999879837036,
    'f_score': 0.519326692812386
}
"""
