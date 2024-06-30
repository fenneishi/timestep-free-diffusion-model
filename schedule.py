import functools
from typing import Callable

import numpy as np
from tqdm.auto import tqdm

import torch

from utils import noise_, noise_like


class ScheduleBase:

    @staticmethod
    def linear_beta_schedule(train_steps: int) -> torch.Tensor:
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, train_steps)

    @staticmethod
    def cosine_beta_schedule(train_steps: int, s=0.008) -> torch.Tensor:
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = train_steps + 1
        x = torch.linspace(0, train_steps, steps)
        alphas_cumprod = torch.cos(
            ((x / train_steps) + s) / (1 + s) * torch.pi * 0.5
        ) ** 2  # 0.99 ~ 0
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # 1 , 0.99, 0.98, 0.97, ...,0.02, 0.01, 0
        # 0.99, 0.98, 0.97, ..., 0.02,0.01, 0
        # 1.00, 0.99, 0.98, 0.97,..., 0.02, 0.01,
        # 0.99, 0.9898,0.9897, ..., 0.5, 0.00
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @staticmethod
    def quadratic_beta_schedule(train_steps: int) -> torch.Tensor:
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, train_steps) ** 2

    @staticmethod
    def sigmoid_beta_schedule(train_steps: int) -> torch.Tensor:
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, train_steps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class ScheduleDDPM(ScheduleBase):
    schedule_fn = ScheduleBase.linear_beta_schedule
    T = 1000

    @torch.no_grad()
    @torch.inference_mode()
    def __init__(
            self,
            schedule_fn: callable = ScheduleBase.linear_beta_schedule,
            ddpm_T: int = 1000,
    ):
        self.schedule_fn: callable = schedule_fn
        self.ddpm_T: int = ddpm_T

        # betas
        self.betas: torch.Tensor = self.schedule_fn(train_steps=self.ddpm_T)
        # torch.tensor([0] + self.schedule_fn(train_steps=self.ddpm_steps).tolist()))

        # alphas
        self.alphas: torch.Tensor = 1. - self.betas
        # self.alphas_sqrt: torch.Tensor = self.alphas ** 0.5
        self.alphas_sqrt_recip: torch.Tensor = (1.0 / self.alphas) ** 0.5

        # cumulative product of alphas
        self.alphasCumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.alphasCumprod_prev: torch.Tensor = torch.tensor([1] + self.alphasCumprod.tolist()[:-1])
        self.alphasCumprod_sqrt: torch.Tensor = self.alphasCumprod ** 0.5
        # self.alphasCumprod_oneMinus: torch.Tensor = 1. - self.alphasCumprod
        self.alphasCumprod_oneMinus_sqrt: torch.Tensor = (1. - self.alphasCumprod) ** 0.5
        self.std_decay = (1. - self.alphasCumprod_prev) / (1. - self.alphasCumprod)

        # posterior std dev
        eta = 1
        self.posteriorStdDev = eta * (self.std_decay * self.betas) ** 0.5

    @torch.no_grad()
    def uniform_t_sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(1, self.ddpm_T, (batch_size,), device=device)

    @torch.no_grad()
    def normal_t_sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        samples = torch.randn((batch_size,), device=device)
        # samples = (samples - samples.min()) / (samples.max() - samples.min())
        # samples = samples * 1000
        samples = samples * 150 + 500
        samples = torch.clamp(samples, min=0, max=999) + 1
        samples = samples.to(torch.int64)
        return samples

    @classmethod
    @torch.no_grad()
    @torch.inference_mode()
    def extract(cls, a, t, x_shape):
        batch_size = t.shape[0]
        index = t - 1
        # index = torch.clip(index, 0, a.shape[0] - 1)
        out = a.gather(-1, index.cpu())
        return out.reshape(
            batch_size,
            *((1,) * (len(x_shape) - 1))
        ).to(t.device)  # [B] -> [B, 1, 1, 1]

    @torch.no_grad()
    def q_sample(
            self,
            x_0: torch.Tensor,  # [B, C, H, W]
            t: torch.Tensor,  # [B]
            noise: torch.Tensor | None = None  # [B, C, H, W]
    ) -> torch.Tensor:
        """
        forward diffusion (using the nice property)
        """
        extract = functools.partial(self.extract, t=t, x_shape=x_0.shape)

        # extract the parameters at time t
        alphasCumprod_sqrt__t = extract(self.alphasCumprod_sqrt)
        alphasCumprod_oneMinus_sqrt__t = extract(self.alphasCumprod_oneMinus_sqrt)

        # get the distribution of x_t given x_0
        mean = alphasCumprod_sqrt__t * x_0
        std = alphasCumprod_oneMinus_sqrt__t

        # sample from the x_t distribution
        noise = noise_like(x_0) if noise is None else noise
        x_t = mean + std * noise  # parameterization sampling method

        return x_t

    @torch.no_grad()
    @torch.inference_mode()
    def p_sample(
            self,
            model: torch.nn.Module,
            x_t: torch.Tensor,  # [B, C, H, W]
            t: torch.Tensor,  # [B]
            noise: torch.Tensor | None = None  # [B, C, H, W]
    ) -> torch.Tensor:
        extract = functools.partial(self.extract, t=t, x_shape=x_t.shape)

        # 1.extract the parameters at time t
        beta__t = extract(self.betas)
        alphas_sqrt_recip__t = extract(self.alphas_sqrt_recip)
        alphasCumprod_oneMinus_sqrt__t = extract(self.alphasCumprod_oneMinus_sqrt)
        posteriorStdDev__t = extract(self.posteriorStdDev)

        # alphas_sqrt__t = extract(self.alphas_sqrt)
        # alphasCumprod_prev__t = extract(self.alphasCumprod_prev)
        # alphasCumprod_oneMinus__t = extract(self.alphasCumprod_oneMinus)
        # alphasCumprod_sqrt__t = extract(self.alphasCumprod_sqrt)
        # std_decay__t = extract(self.std_decay)
        # def get_mean(error_t):
        #     x_0 = alphasCumprod_sqrt__t(x_t - alphasCumprod_oneMinus_sqrt__t * error_t)
        #     mean = (
        #             alphas_sqrt__t * std_decay__t * x_t
        #             +
        #             alphasCumprod_oneMinus__t * (alphasCumprod_prev__t ** 0.5) * beta__t * x_0
        #     )
        #     return mean

        # 2.get the distribution of x_{t-1} given x_t
        mean = alphas_sqrt_recip__t * (
                x_t - (beta__t / alphasCumprod_oneMinus_sqrt__t) * model(x_t, t)
        )
        std = posteriorStdDev__t

        # 3.sample from the x_{t-1} distribution
        # noise = noise_like(x_t) if noise is None else noise
        noise = torch.randn_like(x_t)
        x_t_prev = mean + std * noise  # parameterization sampling method
        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(
            self,
            model: torch.nn.Module | Callable,
            shape=(16, 3, 256, 256),
            device: torch.device | str = "cuda"
    ) -> list[torch.Tensor]:
        batch_size = shape[0]
        steps = (torch.arange(0, self.ddpm_T) + 1).flip(0).repeat(batch_size, 1).t().to(device)
        # x = noise_(shape, device=device)
        x = torch.randn(shape, device=device)
        res = []
        for t in tqdm(steps, 'sampling loop time step'):
            x = self.p_sample(model, x, t)
            res.append(x)

        epio = 1e-3
        while True:
            x,t = self.p_sample(model, x)
            if t<epio:
                break


        return res


class ScheduleDDIM(ScheduleDDPM):
    def __init__(
            self,
            schedule_fn=ScheduleBase.linear_beta_schedule,
            ddpm_steps: int = 300,
            ddim_steps: int = 30,
            ddim_discretize: str = "uniform",
            ddim_eta: float = 0.
    ):
        raise NotImplementedError("DDIM is not implemented yet")
        super().__init__(schedule_fn, ddpm_steps)

        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta

        if ddim_discretize == 'uniform':
            self.ddim_steps_list = torch.arange(0, self.ddpm_T, self.ddpm_T // self.ddim_steps) + 1
        elif ddim_discretize == 'quad':
            self.ddim_steps_list = (np.linspace(0, np.sqrt(self.ddpm_T * .8), self.ddim_steps) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(f"Discretion method {ddim_discretize} not implemented")

        # betas
        self.ddim_betas: torch.Tensor = self.betas[
            self.ddim_steps_list
        ].clone()

        # alphas
        self.ddim_alphas: torch.Tensor = self.alphas[
            self.ddim_steps_list
        ].clone()
        self.ddim_alphas_recip_sqrt: torch.Tensor = self.alphas_sqrt_recip[
            self.ddim_steps_list
        ].clone()

        # cumulative product of alphas
        self.ddim_alphasCumprod: torch.Tensor = self.alphasCumprod[
            self.ddim_steps_list
        ].clone()
        self.ddim_alphasCumprod_prev: torch.Tensor = self.alphasCumprod[
            [0] + self.ddim_steps_list[:-1].tolist()
            ].clone()
        self.ddim_alphasCumprod_sqrt: torch.Tensor = self.alphasCumprod_sqrt[
            self.ddim_steps_list
        ].clone()
        self.ddim_alphasCumprod_oneMinus_sqrt: torch.Tensor = self.alphasCumprod_oneMinus_sqrt[
            self.ddim_steps_list
        ].clone()

        # posterior std dev
        self.ddim_posteriorStdDev = (
                self.ddim_eta *
                (
                        (
                                (1 - self.ddim_alphasCumprod_prev) / (1 - self.ddim_alphasCumprod)
                        )
                        *
                        (
                                1 - self.ddim_alphasCumprod / self.ddim_alphasCumprod_prev
                        )
                ) ** .5
        )

    @torch.no_grad()
    @torch.inference_mode()
    def p_sample(
            self,
            model: torch.nn.Module,
            x_s: torch.Tensor,  # [B, C, H, W]
            s: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        # 考虑迭代起点是否是从纯噪声开始
        extract = functools.partial(self.extract, t=s, x_shape=x_s.shape)

        # 1.extract the parameters at time t
        beta__t = extract(self.betas)
        alphasCumprod_oneMinus_sqrt__t = extract(self.alphasCumprod_oneMinus_sqrt)
        alphas_recip_sqrt__t = extract(self.alphas_sqrt_recip)
        posteriorVariance__t = extract(self.ddim_posteriorStdDev)

        # 2.get the distribution of x_{t-1} given x_t
        mean = alphas_recip_sqrt__t * (x_s - (beta__t / alphasCumprod_oneMinus_sqrt__t) * model(x_s, s))
        var = posteriorVariance__t ** 0.5

        # 3.sample from the x_{t-1} distribution
        noise = torch.randn_like(x_s)
        x_t_prev = mean + var * noise  # parameterization sampling method
        return x_t_prev


if __name__ == "__main__":
    schedule = ScheduleDDPM()
