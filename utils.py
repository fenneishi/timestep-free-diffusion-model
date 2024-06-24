import torch


def noise_(
        size: tuple[int, int, int, int],
        device: torch.device | str = "cuda",
        seed: int | None = None
) -> torch.Tensor:
    if seed is not None:
        noise = torch.randn(size, device=device, generator=torch.Generator().manual_seed(seed))
    else:
        b, c, h, w = size
        seeds = torch.randint(torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, (b,)).tolist()
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        noise = [torch.randn((c, h, w), generator=generator, device=device) for generator in generators]
        noise = torch.stack(noise, dim=0)
    return noise


def noise_like(
        x: torch.Tensor,
        seed: int | None = None
) -> torch.Tensor:
    # torch.randn_like(input) is equivalent to torch.randn(input.size(), dtype=input.dtype, layout=input.layout,
    # device=input.device).
    size = x.size()
    device = x.device
    dtype = x.dtype
    layout = x.layout

    if seed is not None:
        noise = torch.randn(size, dtype=dtype, layout=layout, device=device,
                            generator=torch.Generator().manual_seed(seed))
    else:
        b, *shape = size
        seeds = torch.randint(torch.iinfo(torch.int64).min, torch.iinfo(torch.int64).max, (b,)).tolist()
        generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
        noise = [torch.randn(
            shape, dtype=dtype, layout=layout, device=device,
            generator=generator
        ) for generator in generators]
        noise = torch.stack(noise, dim=0)
    return noise


def num_to_groups(
        num: int,  # 4
        divisor: int  # batch_size
):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def clamp(x, lower_bound=0., upper_bound=1.):
    return max(lower_bound, min(x, upper_bound))
