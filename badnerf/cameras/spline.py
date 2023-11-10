"""
SE(3) spline trajectory library
"""

from typing import Tuple

import pypose as pp
import torch
from jaxtyping import Float
from pypose import LieTensor
from torch import Tensor


def linear_interpolation(
        start_pose: Float[LieTensor, "*batch_size 7"],
        end_pose: Float[LieTensor, "*batch_size 7"],
        u: Float[Tensor, "interpolations"],
) -> Tuple[
    Float[LieTensor, "*batch_size interpolations 4"],
    Float[Tensor, "*batch_size interpolations 3"]
]:
    """
    Linear interpolation between two SE(3) poses.
    Args:
        start_pose: The start pose.
        end_pose: The end pose.
        u: Normalized positions on the trajectory between two poses. Range: [0, 1].
    Returns:
        q: The interpolated rotation.
        t: The interpolated translation.
    """

    _EPS = 1e-6
    batch_size = start_pose.shape[:-1]
    interpolations = u.shape
    u = u.tile((*batch_size, 1))  # [*batch_size, interpolations]

    t_start = start_pose.translation()
    q_start = start_pose.rotation()
    t_end = end_pose.translation()
    q_end = end_pose.rotation()

    u[torch.isclose(u, torch.zeros(u.shape, device=u.device), rtol=_EPS)] += _EPS
    u[torch.isclose(u, torch.ones(u.shape, device=u.device), rtol=_EPS)] -= _EPS

    t = pp.bvv(1 - u, t_start) + pp.bvv(u, t_end)

    q_tau_0 = q_start.Inv() @ q_end
    r_tau_0 = q_tau_0.Log()
    q_t_0 = pp.Exp(pp.so3(pp.bvv(u, r_tau_0)))
    q = q_start.unsqueeze(-2).tile((*interpolations, 1)) @ q_t_0
    return q, t


def linear_interpolation_mid(
        start_pose: Float[LieTensor, "*batch_size 7"],
        end_pose: Float[LieTensor, "*batch_size 7"],
) -> Tuple[
    Float[LieTensor, "*batch_size 4"],
    Float[Tensor, "*batch_size 3"]
]:
    """
    Get the median between two SE(3) poses by linear interpolation.
    Args:
        start_pose: The start pose.
        end_pose: The end pose.
    Returns:
        q: The median's rotation.
        t: The median's translation.
    """

    t_start = start_pose.translation()
    q_start = start_pose.rotation()
    t_end = end_pose.translation()
    q_end = end_pose.rotation()

    t = (t_start + t_end) / 2

    q_tau_0 = q_start.Inv() @ q_end
    q_t_0 = pp.Exp(pp.so3(q_tau_0.Log() / 2))
    q = q_start @ q_t_0
    return q, t
