
from torch.optim import Adam
from dataset_FashionMNIST import build_data
from loss import loss_f
from model import Unet, HowTo_t
from tqdm import tqdm
from evaluate import evaluate
from config import *

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
    _predicted: torch.Tensor = model(*args, **kwargs)
    if how_to_t == HowTo_t.predict_t:
        _predicted_noise, _predicted_t = _predicted[:, :-1, :, :], _predicted[:, -1:, :, :]
        _predicted = _predicted_noise
        # print(f"predicted_t: {1- (predicted_t.mean().item() + 1) / 2}")
    return _predicted


step = 0


def evaluate_model():
    global step
    model.eval()
    evaluate(call_model, step)
    model.train()


def save_model():
    global step
    print(f"Saving model at step {step}")
    model_name = save_model_name(step)
    torch.save(model.state_dict(), model_name)
    print(f"Model saved at {model_name}")


for epoch in tqdm(range(epochs), desc="epochs", colour='green'):
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
        noise = torch.randn_like(x_0)
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
        if (
                (step % save_and_evaluate_every == 0 and step >= start_save_and_evaluate and step > 0)
                or
                step == start_save_and_evaluate
        ):
            evaluate_model()
            save_model()

        step += 1

evaluate_model()
save_model()
wandb.finish()
