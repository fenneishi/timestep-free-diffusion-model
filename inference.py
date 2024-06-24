from pathlib import Path
import torch
from model import Unet
from schedule import ScheduleDDPM as Schedule
from torchvision.utils import save_image

image_size = 28
channels = 1
assert torch.cuda.is_available()
device = "cuda"
schedule_fn = Schedule.linear_beta_schedule
T = 300
results_folder = Path("./results")

schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
).to(device)

model.load_state_dict(torch.load("model.pth"))
imgs = schedule.p_sample_loop(model, shape=(64, channels, image_size, image_size))[-1]
# imgs = torch.cat(imgs)
imgs = (imgs + 1) / 2
save_image(imgs, str(results_folder / f'infer.png'))
