import contextlib
import math
from typing import Generator, Literal, Optional, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from nerfacc import OccGridEstimator

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.utils import colors
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize
from abc import abstractmethod
from typing import Callable, List, Optional, Protocol, Tuple, Union

from nerfstudio.model_components.ray_samplers import SpacedSampler

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None

@contextlib.contextmanager
def background_color_override_context(mode: Float[Tensor, "3"]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


class RGBRenderer(nn.Module):
    """Standard volumetric rendering.

    Args:
        background_color: Background color as RGB. Uses random colors if None.
    """

    def __init__(self, background_color: BackgroundColor = "random") -> None:
        super().__init__()
        self.background_color: BackgroundColor = background_color

    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        if background_color == "random":
            background_color = torch.rand_like(comp_rgb).to(rgb.device)
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color].to(rgb.device)

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb

    def forward(
        self,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of rgb values.
        """

        if not self.training:
            rgb = torch.nan_to_num(rgb)
        rgb = self.combine_rgb(
            rgb, weights, background_color=self.background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb

class AccumulationRenderer(nn.Module):
    """Accumulated value along a ray."""

    @classmethod
    def forward(
        cls,
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 1"]:
        """Composite samples along ray and calculate accumulation.

        Args:
            weights: Weights for each sample
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of accumulated values.
        """

        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            accumulation = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            accumulation = torch.sum(weights, dim=-2)
        return accumulation

class UniformSampler(SpacedSampler):
    """Sample uniformly along a ray

    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )