import torch.nn.functional as F
from einops import rearrange


def loss_f(
        noise, predicted_noise,
        t=None, predicted_t=None,
        loss_type="huber"):
    loss_fn = {
        'l1': F.l1_loss,
        'l2': F.mse_loss,
        'huber': F.smooth_l1_loss
    }[loss_type]

    loss = loss_fn(noise, predicted_noise)
    if t is not None and predicted_t is not None:
        t = rearrange(t, 'b ... -> b 1 1 1 ...')
        t = (t / 1000.0) * 2 - 1.0
        # t = t *2 or t *0.1
        loss = 0.9 * loss + 0.1 * loss_fn(t, predicted_t)

    return loss
