import torch
import wandb
import warnings
from pathlib import Path
from model import save_model_name, pretrain_model_name, how_to_t, t_signal_type
from dataset_FashionMNIST import build_data, image_size, channels
# from schedule import ScheduleDDPM as Schedule
from schedule import ScheduleDDIM as Schedule

assert torch.cuda.is_available()
device = "cuda"
epochs = 22  # every 1 epoch has 468 steps when batch_size=128 in FashionMNIST
T = 4000
batch_size = 128
learning_rate = 1e-3
schedule_fn = Schedule.linear_beta_schedule
save_and_evaluate_every = 10000 // 1
start_save_and_evaluate = 10000

wandb.login()
run = wandb.init(
    project="timestep-free-diffusion-model",
    entity="fenneishi",
    name=save_model_name(f'scratch')[0:-4],
    # mode="disabled",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "T": T,
        'schedule': Schedule.__name__,
        "schedule_fn": schedule_fn.__name__,
        "save_and_sample_every": save_and_evaluate_every,
        "image_size": image_size,
        "channels": channels,
        "how_to_t": str(how_to_t.value),
        "pretrain_model_name": pretrain_model_name,
        "save_model_name": save_model_name(),
        "t_signal_type": t_signal_type.value,
        'start_save_and_evaluate': start_save_and_evaluate
    },
)

print(
    f'######################################\n'
    f'run_id: {run.id}\n'
    f'run_name: {run.name}\n'
    f'run_config: \n' +
    '\n'.join([f' * {key}: {value}' for key, value in run.config.items()]) + '\n'
                                                                             f'######################################'
)

schedule = Schedule(schedule_fn=schedule_fn, T=T)

warnings.filterwarnings("ignore", category=UserWarning, module='torch_fidelity.datasets')

evaluate_folder = f'./evaluate_{how_to_t.value}_{t_signal_type.value}'
evaluate_folder = Path(evaluate_folder).absolute()
fake_folder = evaluate_folder / 'fake'
real_folder = evaluate_folder / 'real'
FakeImgsCount = 4000

vmeory = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
