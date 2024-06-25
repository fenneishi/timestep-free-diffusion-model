import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch_fidelity.datasets')

import torch
import torch_fidelity
from einops import repeat
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

# from dataset_CIFAR10 import test_data, training_data, channels, image_size
from dataset_FashionMNIST import test_data, training_data, channels, image_size

from model import Unet, pretrain_model_name, how_to_t, HowTo_t
from schedule import ScheduleDDPM as Schedule
from utils import num_to_groups, clamp

root_dir = os.path.dirname(os.path.abspath(__file__))
fake_folder = Path("./evaluate/fake").absolute()
real_folder = Path("./evaluate/real").absolute()

vmeory = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))


def build_fake_data(model_):
    assert torch.cuda.is_available()
    device = "cuda"
    schedule_fn = Schedule.schedule_fn
    T = Schedule.T
    global fake_folder

    fake_imgs = []
    if fake_folder.exists() and fake_folder.is_dir():
        shutil.rmtree(fake_folder)
    if not fake_folder.exists():
        os.makedirs(fake_folder)
    # model && schedule
    # load the model from pretrained
    if isinstance(model_, str):
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,),
            out_dim=channels + 1 if how_to_t == HowTo_t.predict_t else None
        ).to(device)
        model.load_state_dict(torch.load(model_))
        print(f"model loaded from {model_}")
    else:
        model = model_

    schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)

    # generate images and save them

    for b in tqdm(
            num_to_groups(10000, 1024 * 4),
            desc='Generating fake images'
    ):
        imgs = schedule.p_sample_loop(
            model=model,
            shape=(b, channels, image_size, image_size)
        )[-1]
        imgs = (imgs + 1) / 2
        if channels == 1:
            imgs = repeat(imgs, 'b 1 h w -> b 3 h w')
        imgs = [img for img in imgs]
        fake_imgs += imgs

    def save_img(index_img_tuple):
        index, img = index_img_tuple
        # save_image(img, str(evaluate_folder / f'gen_{index}.png'))
        save_image(img, os.path.join(fake_folder, f'{index}.png'))

    with ThreadPoolExecutor() as executor:
        executor.map(
            save_img,
            tqdm(
                enumerate(fake_imgs),
                desc='Saving fake images'
            )
        )


class EvalDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.imgs = test_data

    def __getitem__(self, index):
        img, label = self.imgs[index]
        img = (((img + 1) / 2) * 255).to(torch.uint8).clamp_(0, 255)
        if channels == 1:
            img = repeat(img, '1 h w  -> 3 h w')
        return img

    def __len__(self):
        return len(self.imgs)


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


def evaluate(step, model_: None | str = None):
    build_fake_data(model_)
    build_real_data()
    # build_train_data()
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


if __name__ == "__main__":
    evaluate(0, pretrain_model_name)

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
