import json

import torch
import wandb
import warnings
from pathlib import Path
from model import save_model_name, pretrain_model_name, how_to_t, t_signal_type
from dataset_FashionMNIST import image_size, channels, test_data

# Argument Parser
import argparse

parser = argparse.ArgumentParser(description="Process some parameters.")
parser.add_argument('--device', type=str, required=False, help="The parameter to process", default="cuda")
parser.add_argument('--pretrain_model_name', type=str, required=False, help="The parameter to process",default=None)
args = parser.parse_args()

# Schedule Config
# from schedule import ScheduleDDPM as Schedule
from schedule import ScheduleDDIM as Schedule

T = 4000
schedule_fn = Schedule.linear_beta_schedule
sample_T = 100
sample_T_method = "linear"
eta = 0.0
schedule = Schedule(schedule_fn=schedule_fn, T=T)
time_steps_prev, time_steps = schedule.build_sub_steps(steps=sample_T, method=sample_T_method)

# Training Config
assert torch.cuda.is_available()
device = torch.device(args.device)
epochs = 450  #22 # every 1 epoch has 468 steps when batch_size=128 in FashionMNIST
batch_size = 128
learning_rate = 1e-3
save_and_evaluate_every = 10000 // 1
start_save_and_evaluate = 10000

pretrain_model_name = args.pretrain_model_name if args.pretrain_model_name is not None else pretrain_model_name

# Evaluate Config
evaluate_folder = Path(f'./evaluate/evaluate_{how_to_t.value}_{t_signal_type.value}').absolute()
fake_folder = evaluate_folder / 'fake'
real_folder = evaluate_folder / 'real'
FakeImgsCount = 10000

vmeory = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))

wandb.login()
run = wandb.init(
    project="timestep-free-diffusion-model",
    entity="fenneishi",
    name=save_model_name(f'scratch')[0:-4],
    # mode="disabled",
    config={
        "schedule": {
            'schedule': Schedule.__name__,
            "train": {
                "T": T,

                "schedule_fn": schedule_fn.__name__,
            },
            'sample': {
                'sample_T': sample_T,
                'sample_T_method': sample_T_method,
                'eta': eta,
                'time_steps': time_steps,
                'time_steps_prev': time_steps_prev,
            },
        },
        "train": {
            't': {
                'how_to_t': how_to_t.value,
                't_signal_type': t_signal_type.value,
            },
            "model": {
                "image_size": image_size,
                "channels": channels,
                "pretrain_model_name": pretrain_model_name,
                "save_model_name": save_model_name(),
            },
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_and_sample_every": save_and_evaluate_every,
            'start_save_and_evaluate': start_save_and_evaluate
        },
        'evaluate': {
            'FakeImgsCount': FakeImgsCount,
        }
    },
)

print(
    f'######################################\n'
    f'run_id: {run.id}\n'
    f'run_name: {run.name}\n'
    f'run_config: \n {json.dumps(run.config.as_dict(), indent=4, ensure_ascii=False)}'
    f'\n######################################'
)
