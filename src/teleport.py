"""
Functions for solving teleportation problems.
"""

import torch
from tqdm import tqdm
import numpy as np

# global parameters

ALPHA = 0.5
BETA = 0.8
BETA_INV = 1.25
KKT_TOL = 1e-6
CONST_TOL = 1e-6
MAX_BACKTRACKS = 100
GAMMA_SCALE = 0.1


def normalized_slp(
    x,
    obj_fn,
    max_steps,
    rho,
    allow_sublevel=False,
    line_search=False,
    verbose=False,
):
    """Teleport by solving successive linear approximations.

    This version maximizes the logarithm of the squared gradient norm.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the linear SQP method.
        rho: the step-size to use for each step of linear SQP.

    Returns:
        x: approximate solution to teleportation problem.
    """
    x0 = x.clone()
    f0 = obj_fn(x).item()
    rho0 = rho

    teleport_path = []

    f_next = None
    grad_next = None

    f_diff_next = None
    g_next = None
    num_backtracks = 0

    if allow_sublevel:
        penalty_fn = lambda z: torch.maximum(z, torch.tensor([0]))
    else:
        penalty_fn = torch.abs

    for t in tqdm(range(max_steps)):
        teleport_path.append(x.clone().detach().numpy())

        if f_next is None:
            func_out = obj_fn(x)
            grad = torch.autograd.grad(func_out, x)[
                0
            ]  # ,create_graph=True,retain_graph=True
        else:
            # use computations from line-search
            func_out = f_next
            grad = grad_next

        if torch.isnan(func_out) or torch.isinf(func_out):
            tqdm.write("Teleportation failed! Returning initialization...")
            return x0

        q = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            if f_diff_next is None:
                f_diff = func_out - f0
                g = grad @ grad
            else:
                g = g_next
                f_diff = f_diff_next

            Hg = q / g
            gHg = torch.inner(Hg, grad)
            gHg_g = gHg / g

            # check termination conditions
            proj = gHg_g * grad - Hg

            if verbose:
                tqdm.write(
                    f"Iteration {t+1}/{max_steps}: Obj: {g.item()}, Constr. Gap: {f_diff.item()}, KKT Gap: {proj @ proj}, Backtracks: {num_backtracks}, Step-size: {rho}"
                )

            if proj @ proj <= KKT_TOL and penalty_fn(f_diff) <= CONST_TOL:
                tqdm.write(
                    "KKT conditions approximately satisfied. Terminating SLP procedure."
                )
                return x, teleport_path

            # housekeeping for line-search
            x_prev = torch.tensor(x, requires_grad=True)

        for i in range(MAX_BACKTRACKS):
            # evaluate update
            with torch.no_grad():
                x.add_(Hg, alpha=rho)

                # only project if necessary
                v_scale = (rho * gHg + f_diff) / g
                if not allow_sublevel or v_scale > 0:
                    x.sub_(grad, alpha=v_scale.item())

            if not line_search:
                # accept step-size immediately
                break

            # estimate penalty strength for line-search merit function
            gamma = GAMMA_SCALE
            if penalty_fn(f_diff) > 0:
                gamma *= q @ grad * v_scale / (g**2 * penalty_fn(f_diff))

            # proceed with line-search
            f_next = obj_fn(x)
            grad_next = torch.autograd.grad(f_next, x)[0]

            with torch.no_grad():
                # quantities will be re-used for next step
                # if the step-size is accepted.
                g_next = grad_next @ grad_next
                f_diff_next = f_next - f0
                d_t = x - x_prev

                LHS = torch.log(g_next) / 2 - gamma * penalty_fn(f_diff_next)
                RHS = (
                    torch.log(g) / 2
                    - (1 - ALPHA) * gamma * penalty_fn(f_diff)
                    + ALPHA * Hg @ d_t / 2
                )

                if LHS >= RHS:
                    break

                # reset and try with smaller step-size.
                rho = rho * BETA
                x[:] = x_prev[:]

        # report if line-search failed
        num_backtracks = i
        if i == MAX_BACKTRACKS - 1:
            tqdm.write(
                "WARNING: Line-search failed to return feasible step-size."
            )
            rho = rho0

        # try to increase step-size if merit bound isn't too tight.
        if line_search and LHS / RHS >= 5.0:
            rho = rho * BETA_INV

    return x, teleport_path


def identity(x, obj_fn):
    return x
