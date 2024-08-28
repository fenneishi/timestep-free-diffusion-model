import functools
import os
from typing import Callable, Tuple, Any

import numpy as np
from tqdm.auto import tqdm

import torch
import matplotlib.pyplot as plt


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class ScheduleBase:

    @staticmethod
    def linear_beta_schedule(train_steps: int) -> torch.Tensor:
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, train_steps)

    @staticmethod
    def cosine_beta_schedule(train_steps: int, s=0.008) -> torch.Tensor:
        raise NotImplementedError("cosine_beta_schedule is not implemented!")
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672 ''Improved Denoising Diffusion Probabilistic Models''
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
    schedule_fn_default = ScheduleBase.linear_beta_schedule
    T_default = 1000

    @torch.no_grad()
    def __init__(
            self,
            schedule_fn: callable,
            T: int,
    ):
        self.schedule_fn: callable = schedule_fn
        self.T: int = T

        # betas
        self.betas: torch.Tensor = self.schedule_fn(train_steps=self.T)
        self.betas = self.betas.to(torch.float64)
        # torch.tensor([0] + self.schedule_fn(train_steps=self.ddpm_steps).tolist()))

        # alphas
        self.alphas: torch.Tensor = 1. - self.betas
        # self.alphas_sqrt: torch.Tensor = self.alphas ** 0.5
        self.alphas_sqrt_recip: torch.Tensor = (1.0 / self.alphas) ** 0.5

        # cumulative product of alphas
        self.alphasCumprod: torch.Tensor = torch.cumprod(self.alphas, dim=0)
        self.alphasCumprod_prev: torch.Tensor = torch.tensor([1] + self.alphasCumprod.tolist()[:-1])
        self.alphasCumprod_sqrt: torch.Tensor = self.alphasCumprod ** 0.5  # self.alphasCumprod_oneMinus: torch.Tensor = 1. - self.alphasCumprod
        self.alphasCumprod_oneMinus_sqrt: torch.Tensor = (1. - self.alphasCumprod) ** 0.5
        self.std_decay = (1. - self.alphasCumprod_prev) / (1. - self.alphasCumprod)

        # posterior std dev
        eta = 1
        self.posteriorStdDev = eta * (self.std_decay * self.betas) ** 0.5

    @torch.no_grad()
    def uniform_t_sample(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(1, self.T, (batch_size,), device=device)

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
    def extract(cls, a, t, x_shape, need_reindex=True):
        batch_size = t.shape[0]
        index = t - 1 if need_reindex else t
        # index = torch.clip(index, 0, a.shape[0] - 1)
        out = a.gather(-1, index.cpu())
        return out.reshape(
            batch_size,
            *((1,) * (len(x_shape) - 1))
        ).to(t.device)  # [B] -> [B, 1, 1, 1]

    """
    forward diffusion (using the nice property)
    """

    @torch.no_grad()
    def q_sample(
            self,
            x_0: torch.Tensor,  # [B, C, H, W]
            t: torch.Tensor,  # [B]
            noise: torch.Tensor  # [B, C, H, W]
    ) -> torch.Tensor:

        extract = functools.partial(self.extract, t=t, x_shape=x_0.shape)

        # extract the parameters at time t
        alphasCumprod_sqrt__t = extract(self.alphasCumprod_sqrt)
        alphasCumprod_oneMinus_sqrt__t = extract(self.alphasCumprod_oneMinus_sqrt)

        # get the distribution of x_t given x_0
        mean = alphasCumprod_sqrt__t * x_0
        std = alphasCumprod_oneMinus_sqrt__t

        # sample from the x_t distribution
        assert noise is not None and noise.shape == x_0.shape
        x_t = mean + std * noise  # parameterization sampling method

        return x_t.to(x_0.dtype)

    """
    reverse diffusion  (denoising)
    """

    @torch.no_grad()
    def p_sample(
            self,
            model: torch.nn.Module,
            x_t: torch.Tensor,  # [B, C, H, W]
            t: torch.Tensor,  # [B]
    ) -> torch.Tensor:

        extract = functools.partial(self.extract, t=t, x_shape=x_t.shape)

        # 1.extract the parameters at time t
        beta__t = extract(self.betas)
        alphas_sqrt_recip__t = extract(self.alphas_sqrt_recip)
        alphasCumprod_oneMinus_sqrt__t = extract(self.alphasCumprod_oneMinus_sqrt)
        posteriorStdDev__t = extract(self.posteriorStdDev)

        # 2.get the distribution of x_{t-1} given x_t
        std = posteriorStdDev__t
        mean = alphas_sqrt_recip__t * (
                x_t - (beta__t / alphasCumprod_oneMinus_sqrt__t) * model(x_t, t)
        )
        mean2 = (
                (
                    coff0 := alphas_sqrt_recip__t
                ) * x_t
                +
                (
                    coff1 := -alphas_sqrt_recip__t * (beta__t / alphasCumprod_oneMinus_sqrt__t)
                ) * model(x_t, t)
        )

        # 3.sample from the x_{t-1} distribution
        noise = torch.randn_like(x_t)
        x_t_prev = mean + std * noise  # parameterization sampling method
        return x_t_prev.to(x_t.dtype), coff0, coff1

    @torch.no_grad()
    def p_sample_loop(
            self,
            model: torch.nn.Module | Callable,
            shape=(16, 3, 256, 256),
            device: torch.device | str = "cuda"
    ) -> list[torch.Tensor]:
        batch_size = shape[0]
        steps = (torch.arange(0, self.T) + 1).flip(0).repeat(batch_size, 1).t().to(device)
        # x = noise_(shape, device=device)
        x = torch.randn(shape, device=device)
        res = []
        coff0_s, coff1_s = [], []
        for t in tqdm(steps, 'sampling loop time step'):
            x, coff0, coff1 = self.p_sample(model, x, t)
            coff0_s.append(coff0[0].item())
            coff1_s.append(coff1[0].item())
            res.append(x)

        # epio = 1e-3
        # while True:
        #     x,t = self.p_sample(model, x)
        #     if t<epio:
        #         break

        # plt the coefficient

        return res, coff0_s, coff1_s

    @classmethod
    def plot_noise_levels(cls):
        import matplotlib.pyplot as plt

        save_dir = f'doc/{cls.__name__}'
        os.makedirs(save_dir, exist_ok=True)

        train_schedules = {
            'cosine': cls(cls.cosine_beta_schedule, ddpm_T=1000).alphasCumprod,
            'quadratic': cls(cls.quadratic_beta_schedule, ddpm_T=1000).alphasCumprod,
            'sigmoid': cls(cls.sigmoid_beta_schedule, ddpm_T=1000).alphasCumprod,
            'linear': cls(cls.linear_beta_schedule, ddpm_T=1000).alphasCumprod,
        }

        colours = {
            'cosine': 'red',
            'quadratic': 'green',
            'sigmoid': 'blue',
            'linear': 'grey',
        }

        steps_range = {
            'start10': (None, 10),
            'start30': (None, 30),
            'start60': (None, 60),
            'start100': (None, 100),
            'end10': (-10, None),
            'end30': (-30, None),
            'end60': (-60, None),
            'end100': (-100, None),
            'all': (None, None),
        }

        for step_range_name, steps_range in steps_range.items():
            # plot
            for name, schedule in train_schedules.items():
                schedule = schedule[steps_range[0]:steps_range[1]]
                plt.plot(schedule, label=name, color=colours[name])
                # print(f'{name} alphasCumprod:\n {schedule.tolist()}')

            # save
            plt.legend()
            plt.savefig(f'{save_dir}/alphasCumprod_{step_range_name}.png')
            plt.close()

    @classmethod
    def debug_p_sample_loop(cls):
        def model(x, t):
            return torch.randn_like(x)

        schedule = cls(cls.linear_beta_schedule, T=1000)

        res, coefficient0_s, coefficient1_s = schedule.p_sample_loop(model, shape=(16, 3, 256, 256), device='cuda')

        #
        std_update = [(c0 ** 2 + c1 ** 2) ** 0.5 for c0, c1 in zip(coefficient0_s, coefficient1_s)]
        # cumulative product of std_update
        std_update_cum = torch.cumprod(torch.tensor(std_update), dim=0).tolist()
        std_update_total = std_update_cum[-1]
        plt.plot(coefficient0_s, label='coefficient0')
        plt.plot(coefficient1_s, label='coefficient1')
        # plt.plot(std_update, label='std_update')
        plt.legend()
        # save plt to disk
        plt.savefig(f'coefficients_{cls.__name__}.png')
        plt.show()
        plt.close()

        return res, coefficient0_s, coefficient1_s, std_update_cum


class ScheduleDDIM(ScheduleDDPM):
    schedule_fn_default = ScheduleBase.linear_beta_schedule
    T_default = 4000
    sample_T_default = 100
    sample_T_method_default = "linear"
    eta_default = 0.0

    @torch.no_grad()
    def __init__(
            self,
            schedule_fn: callable,
            T: int,
    ):
        super().__init__(schedule_fn, T)

        # cumulative product of alphas
        self.timesteps_ddim = torch.arange(0, self.T + 1, 1)  # [0,1,             2,                 T             ]
        self.alphas_ddim = torch.tensor(
            [1] + self.alphasCumprod.tolist())  # [1,alphaCumprod_1,alphaCumprod_2,...,alphaCumprod_T]
        assert len(self.timesteps_ddim) == len(self.alphas_ddim) == self.T + 1
        """
        timesteps_ddim is one-to-one mapping to alphas_ddim
        """
        """
        Only used the notation α_t( ̄αt in ddpm ) for three reasons: 
        First, 
        it makes it more clear that we only need to choose one set of hyperparameters, reducing possible 
        cross-references of the derived variables. 
        Second, 
        it allows us to introduce the generalization as well as 
        the acceleration case easier, because the inference process is no longer motivated by a diffusion. 
        Third, 
        there exists an isomorphism between α1:T and 1, . . . , T , which is not the case for βt.(different t may map 
        to the same βt)
        Fourth,
        1-a_t(or 1 -  ̄αt in ddpm) represents the noise level at time t
        """

    @torch.no_grad()
    def p_sample(
            self,
            model: torch.nn.Module,
            x_t: torch.Tensor,  # [B, C, H, W]
            step: torch.Tensor,  # [B]
            prev_step: torch.Tensor,  # [B]
            eta: float = 1.0,
    ) -> torch.Tensor:

        extract = functools.partial(self.extract, x_shape=x_t.shape, need_reindex=False)

        # 1.extract the parameters at time t
        """
        self.alphas_ddim is a list saving [1,a1,a2,a3,a4,a5 .... at,at+1 ... aT]
        so the right index for alphasCumprod__t or alphasCumprod_prev__t just is t
        e.g.
        if we want to get a3, the index should be 3,so need_reindex=False
        e.g.
        T=1000, S=30, T%S=10, T-T%S=990, T-T%S+1=991 ,T-T%S+1-S=961
        alphasCumprod__t - 1 is blow:
        [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,600,630,660,690,720,750,780,810,840,870,900,930,960,990]
        alphasCumprod__t is blow:
        [1,31,61,91,121,151,181,211,241,271,301,331,361,391,421,451,481,511,541,571,601,631,661,691,721,751,781,811,841,871,901,931,961,991]
        alphasCumprod_prev__t is blow:
        [0, 1,31,61, 91,121,151,181,211,241,271,301,331,361,391,421,451,481,511,541,571,601,631,661,691,721,751,781,811,841,871,901,931,961]
        """
        alpha = extract(a=self.alphas_ddim, t=step)  # 1 <= step <= (T-T%S + 1)
        alphas_prev = extract(a=self.alphas_ddim, t=prev_step)  # 0 <= prev_step <= (T-T%S + 1-S)
        alpha, alphas_prev = alpha.to(torch.float64), alphas_prev.to(torch.float64)

        # 2. calculate x_{t-1}
        std = eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alpha)  # variance decay
            *
            (1 - alpha / alphas_prev)  # beta_t
        )  # posterior standard deviation

        mean = (
                (
                    coefficient0 := torch.sqrt(alphas_prev / alpha)
                ) * x_t
                +
                (
                    coefficient1 := (
                            torch.sqrt(1 - alphas_prev - std ** 2)
                            -
                            torch.sqrt(
                                (alphas_prev * (1 - alpha))
                                /
                                alpha
                            )
                    )
                ) * model(x_t, step)
        )
        noise = torch.randn_like(x_t)
        x_t_prev = mean + std * noise  # parameterization sampling method
        return x_t_prev.to(x_t.dtype), coefficient0, coefficient1

    @torch.no_grad()
    def build_sub_steps(self, steps: int = 50, method="linear") -> tuple[Any, Any]:
        # todo:考虑迭代起点是否是从纯噪声开始
        if method == "linear":
            sub_indexes = torch.arange(0, self.T, self.T // steps).to(torch.int) + 1
        elif method == "quadratic":
            sub_indexes = (torch.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).to(torch.int) + 1
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        sub_indexes_prev = torch.tensor([0] + sub_indexes.tolist()[:-1])

        return self.timesteps_ddim[sub_indexes_prev], self.timesteps_ddim[sub_indexes]

    @torch.no_grad()
    def p_sample_loop(
            self,
            model: torch.nn.Module | Callable,
            time_steps_prev: torch.Tensor | None = None,
            time_steps: torch.Tensor | None = None,
            eta: float | None = None,
            device: torch.device | str = "cuda",
            shape=(16, 3, 256, 256),
    ) -> list[torch.Tensor]:
        batch_size = functools.partial(
            lambda _: _.flip(0).repeat(shape[0], 1).t().to(device)
        )  # [S] --> [B, S] --> [S, B]

        eta = self.eta_default if eta is None else eta

        if time_steps is None or time_steps_prev is None:
            time_steps_prev, time_steps = self.build_sub_steps(steps=self.sample_T_default,
                                                               method=self.sample_T_method_default)

        time_steps_prev, time_steps = batch_size(time_steps_prev).to(device), batch_size(time_steps).to(device)

        x = torch.randn(shape, device=device)
        res = []
        coefficient0_s, coefficient1_s = [], []
        for prev_step, step in tqdm(zip(time_steps_prev, time_steps), 'sampling loop time step'):
            x, coefficient0, coefficient1 = self.p_sample(
                model=model,
                x_t=x,
                step=step,
                prev_step=prev_step,
                eta=eta
            )
            res.append(x)
            coefficient0_s.append(coefficient0[0].item())
            coefficient1_s.append(coefficient1[0].item())

        return res, coefficient0_s, coefficient1_s

    @classmethod
    def plot_noise_levels(cls):
        import matplotlib.pyplot as plt

        save_dir = f'doc/{cls.__name__}'
        os.makedirs(save_dir, exist_ok=True)

        train_schedules = {
            'cosine': cls(cls.cosine_beta_schedule, T=1000),
            'quadratic': cls(cls.quadratic_beta_schedule, T=1000),
            'sigmoid': cls(cls.sigmoid_beta_schedule, T=1000),
            'linear': cls(cls.linear_beta_schedule, T=1000),
        }

        colours = {
            'cosine': 'red',
            'quadratic': 'green',
            'sigmoid': 'blue',
            'linear': 'grey',
        }

        steps_range = {
            'start10': (None, 10),
            'start30': (None, 30),
            'start60': (None, 60),
            'start100': (None, 100),
            'end10': (-10, None),
            'end30': (-30, None),
            'end60': (-60, None),
            'end100': (-100, None),
            'all': (None, None),
        }

        for step_range_name, steps_range in steps_range.items():
            # plot
            for name, schedule in train_schedules.items():
                steps = schedule.timesteps_ddim[steps_range[0]:steps_range[1]]
                signal_levels = schedule.alphas_ddim[steps_range[0]:steps_range[1]]
                plt.plot(steps, signal_levels, label=name, color=colours[name])
                # print(f'{name} alphasCumprod:\n {schedule.tolist()}')

            # save
            plt.legend()
            plt.savefig(f'{save_dir}/alphasCumprod_{step_range_name}.png')
            plt.close()

    @classmethod
    def debug_p_sample_loop(cls, sub_steps=1000, eta=1.0):
        def model(x, t):
            return torch.randn_like(x)

        schedule = cls(cls.linear_beta_schedule, T=1000)

        time_steps_prev, time_steps = schedule.build_sub_steps(steps=sub_steps, method="linear")

        res, coefficient0_s, coefficient1_s = schedule.p_sample_loop(
            model,
            time_steps_prev=time_steps_prev,
            time_steps=time_steps,
            eta=eta,
            shape=(16, 3, 256, 256),
            device='cuda'
        )
        std_update = [(c0 ** 2 + c1 ** 2) ** 0.5 for c0, c1 in zip(coefficient0_s, coefficient1_s)]
        std_update_cum = torch.cumprod(torch.tensor(std_update), dim=0).tolist()
        std_update_total = std_update_cum[-1]
        plt.plot(coefficient0_s, label='coefficient0')
        plt.plot(coefficient1_s, label='coefficient1')
        plt.legend()
        # save plt to disk
        plt.savefig(f'coefficients_{cls.__name__}.png')
        plt.show()
        plt.close()

        global res_ddpm, coeff0_s_ddpm, coeff1_s_ddpm, std_update_cum_ddpm
        if res_ddpm is None:
            res_ddpm, coeff0_s_ddpm, coeff1_s_ddpm, std_update_cum_ddpm = ScheduleDDPM.debug_p_sample_loop()

        diff_coeff0 = [abs(a - b) for a, b in zip(coeff0_s_ddpm, coefficient0_s)]
        diff_coeff1 = [abs(a - b) for a, b in zip(coeff1_s_ddpm, coefficient1_s)]
        diff_std_update = [abs(a - b) for a, b in zip(std_update_cum_ddpm, std_update_cum)]

        is_zero = functools.partial(lambda f, epsilon=1e-6: abs(f) < epsilon)

        for i, (c0, c1, std) in enumerate(zip(diff_coeff0, diff_coeff1, diff_std_update)):
            if not is_zero(c0):
                print(f'c0 diff at {i} is {c0}')
            if not is_zero(c1):
                print(f'c1 diff at {i} is {c1}')
            if not is_zero(std, 1e-4):
                print(f'std diff at {i} is {std}')

        plt.plot([i + 1 for i in range(1000)], std_update_cum_ddpm, label='std_update_cum_ddpm')
        plt.plot(time_steps.tolist(), std_update_cum, label='std_update_cum_ddim')
        plt.legend()
        plt.savefig(f'std_update_cum_{cls.__name__}_{sub_steps}.png')
        plt.show()
        plt.close()

        return res, coefficient0_s, coefficient1_s, std_update_cum, time_steps.tolist()


if __name__ == "__main__":
    # ScheduleDDPM.plot_noise_levels()
    # ScheduleDDIM.plot_noise_levels()
    res_ddpm, coeff0_s_ddpm, coeff1_s_ddpm, std_update_cum_ddpm = None, None, None, None

    for eta in [0.0, 0.5, 1.0]:
        std_update_cum_res = []
        sub_steps_list = [50, 100, 500, 1000]
        for sub_steps in sub_steps_list:
            _, __, ___, std_update_cum, time_steps = ScheduleDDIM.debug_p_sample_loop(sub_steps=sub_steps,eta=eta)
            std_update_cum_res.append((std_update_cum, time_steps))

        for std_update_cum, time_steps in std_update_cum_res:
            plt.plot(time_steps, std_update_cum, label=f'std_update_cum_{len(time_steps)}')
        plt.plot([i + 1 for i in range(1000)], std_update_cum_ddpm, label='std_update_cum_ddpm',color='black', linestyle='--')
        plt.legend()
        plt.savefig(f'std_update_cum_all_{eta}.png')
        plt.show()
        plt.close()
