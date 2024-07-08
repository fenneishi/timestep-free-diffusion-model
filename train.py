import torch
from torch.optim import Adam

from tqdm import tqdm
import wandb

from model import Unet, save_model_name, pretrain_model_name, how_to_t, HowTo_t, t_signal_type
from dataset_FashionMNIST import build_data, image_size, channels
from schedule import ScheduleDDPM as Schedule
from loss import loss_f
from evaluate import evaluate
from utils import noise_like

assert torch.cuda.is_available()
device = "cuda"
epochs = 22  # every 1 epoch has 468 steps when batch_size=128 in FashionMNIST
T = Schedule.T
batch_size = 128
learning_rate = 1e-3
schedule_fn = Schedule.schedule_fn
save_and_evaluate_every = 1000 // 1
start_save_and_evaluate = 4000

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
    f'run_config: {run.config}\n'
    f'\n######################################'
)

schedule = Schedule(schedule_fn=schedule_fn, ddpm_T=T)
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
    out_dim=channels + 1 if how_to_t == HowTo_t.predict_t else None
).to(device)

if pretrain_model_name is not None:
    model.load_state_dict(torch.load(pretrain_model_name))

optimizer = Adam(model.parameters(), lr=learning_rate)
dataloader = build_data(batch_size=batch_size, train=True)


def call_model(*args, **kwargs):
    predicted: torch.Tensor = model(*args, **kwargs)
    if how_to_t == HowTo_t.predict_t:
        predicted_noise, predicted_t = predicted[:, :-1, :, :], predicted[:, -1:, :, :]
        predicted = predicted_noise
        # print(f"predicted_t: {1- (predicted_t.mean().item() + 1) / 2}")
    return predicted


step = 0


def evaluate_model():
    model.eval()
    evaluate(call_model, step)
    model.train()


def save_model():
    print(f"Saving model at step {step}")
    model_name = save_model_name(step)
    torch.save(model.state_dict(), model_name)
    print(f"Model saved at {model_name}")


for epoch in range(epochs):
    print(f"epoch {epoch},step {step}")
    for x_0, _ in tqdm(dataloader, desc=f"epoch {epoch}"):
        optimizer.zero_grad()
        # class labels
        # _ = (_ + 1).to(device)

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

        # predict
        predicted: torch.Tensor = model(X_t, t)
        # predicted: torch.Tensor = model(X_t, _)
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

        # optimize
        loss.backward()
        optimizer.step()

        # evaluate and save model
        if step % save_and_evaluate_every == 0 and step >= start_save_and_evaluate and step > 0:
            evaluate_model()
            save_model()

        step += 1

evaluate_model()
save_model()
wandb.finish()
