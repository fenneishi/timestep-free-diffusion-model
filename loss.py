import torch.nn.functional as F
from einops import rearrange
import wandb


def loss_f(
        noise, predicted_noise,
        step,
        t=None, predicted_t=None,
        loss_type="huber"):
    loss_fn = {
        'l1': F.l1_loss,
        'l2': F.mse_loss,
        'huber': F.smooth_l1_loss
    }[loss_type]

    loss = loss_fn(predicted_noise,noise)
    if step % 10 == 0 and wandb.run is not None:
        wandb.log(data={"loss": loss.item()}, step=step)
    if t is not None and predicted_t is not None:
        t = rearrange(t, 'b ... -> b 1 1 1 ...')
        t = (t / 1000.0) * 2 - 1.0
        # t = t *2 or t *0.1
        predicted_t = predicted_t.mean(dim=[2, 3], keepdim=True)
        t_loss = loss_fn(predicted_t,t)
        if step % 10 == 0 and wandb.run is not None:
            wandb.log(data={"t_loss": t_loss.item()}, step=step)
        loss = 0.9 * loss + 0.1 * t_loss

    return loss
