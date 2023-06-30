"""
Functions for solving teleportation problems.
"""


import torch
from tqdm import tqdm
import numpy as np


def try_update(
    x,
    Hv,
    vHv,
    denom,
    grad,
    f_diff,
    eta,
    normalize=False,
    allow_sublevel=False,
):
    with torch.no_grad():
        eta_step = eta
        if normalize:
            eta_step = eta / denom

        x.add_(Hv, alpha=eta_step)

        if allow_sublevel and eta_step * vHv - f_diff <= 0:
            # skip projection since step is inside half-space.
            tqdm.write("Skipping projection")
            return x

        negstep = (f_diff - eta_step * vHv) / denom

        x.add_(grad, alpha=negstep.item())

    return x


def linear_sqp(
    x,
    obj_fn,
    max_steps,
    eta,
    normalize=False,
    allow_sublevel=False,
    line_search=False,
    verbose=False,
):
    """Teleport by solving successive linear approximations.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the linear SQP method.
        eta: the step-size to use for each step of linear SQP.

    Returns:
        x: approximate solution to teleportation problem.
    """
    f0 = obj_fn(x).item()

    for t in tqdm(range(max_steps)):
        func_out = obj_fn(x)

        grad = torch.autograd.grad(func_out, x)[
            0
        ]  # ,create_graph=True,retain_graph=True

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            f_diff = f0 - func_out
            grad_norm = grad @ grad

            if verbose:
                tqdm.write(
                    f"Iteration {t}/{max_steps}: Function Diff: {f_diff.item()}, Grad norm: {grad_norm.item()}"
                )

            vHv = torch.inner(Hv, grad)

            x_old = x.clone()
            x = try_update(
                x,
                Hv,
                vHv,
                grad_norm,
                grad,
                f_diff,
                eta,
                normalize=normalize,
                allow_sublevel=allow_sublevel,
            )

        if line_search:
            f_next = obj_fn(x)
            g_next = torch.autograd.grad(f_next, x)[0]
            f_diff_next = f0 - f_next
            if normalize:
                mu = 1
                LHS = -torch.log(g_next @ g_next) + 2 * mu * torch.abs(
                    f_diff_next
                )
                RHS = (
                    -torch.log(grad_norm)
                    + 2 * mu * torch.abs(f_diff)
                    - (mu * torch.sign(f_diff) * grad + Hv / grad_norm)
                    @ (x - x_old)
                )
            else:
                mu = 1e3
                LHS = -g_next @ g_next + 2 * mu * torch.abs(f_diff_next)
                RHS = (
                    -grad_norm
                    + 2 * mu * torch.abs(f_diff)
                    - (mu * torch.sign(f_diff) * grad + Hv) @ (x - x_old)
                )

            while LHS > RHS:
                eta = eta * 0.8
                print(eta)
                # reset
                x[:] = x_old
                # try again
                x = try_update(
                    x,
                    Hv,
                    vHv,
                    grad_norm,
                    grad,
                    f_diff,
                    eta,
                    normalize=normalize,
                    allow_sublevel=allow_sublevel,
                )
                f_next = obj_fn(x)
                g_next = torch.autograd.grad(f_next, x)[0]
                f_diff_next = f0 - f_next

                if normalize:
                    LHS = -torch.log(g_next @ g_next) + 2 * mu * torch.abs(
                        f_diff_next
                    )
                    RHS = (
                        -torch.log(grad_norm)
                        + 2 * mu * torch.abs(f_diff)
                        - (mu * torch.sign(f_diff) * grad + Hv / grad_norm)
                        @ (x - x_old)
                    )
                else:
                    LHS = -g_next @ g_next + 2 * mu * torch.abs(f_diff_next)
                    RHS = (
                        -grad_norm
                        + 2 * mu * torch.abs(f_diff)
                        - (mu * torch.sign(f_diff) * grad + Hv) @ (x - x_old)
                    )

            # try to increase step-size
            eta = eta * 1.25

    return x


def al_method(
    x,
    obj_fn,
    max_steps,
    max_inner_steps,
    inner_tol,
    mu,
    eta,
    verbose=False,
):
    """Teleport using augmented Lagrangian method.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the augmented Lagrangian
            method.
        max_inner_steps: the maximum number of steps to run gradient descent
            when minimizing the augmented Lagrangian.
        inner_tol: the tolerance for terminating the inner optimization
            method.
        mu: the penalty strength.
        eta: the step-size for the inner optimization method.
        verbose: whether or not to print iteration statistics.

    Returns:
        x: approximate solution to teleportation problem.
    """
    # level set
    f0 = obj_fn(x).item()

    # dual parameters
    lam = 0

    # deviation from level set
    f_diff = 0

    for al_steps in tqdm(range(max_steps)):
        # minimize augmented Lagrangian to evaluate primal update

        for t in tqdm(range(max_inner_steps)):
            func_out = obj_fn(x)
            grad = torch.autograd.grad(func_out, x)[0]

            Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

            with torch.no_grad():
                f_diff = f0 - func_out.item()

                al_grad = (lam + mu * f_diff) * grad - Hv
                grad_norm = al_grad @ al_grad

                if verbose:
                    tqdm.write(
                        f"Iteration {t}/{max_inner_steps}: Function Diff: {f_diff}, AL Grad norm: {grad_norm.item()}"
                    )

                # termination criterion.
                if grad_norm.item() <= inner_tol:
                    print(
                        "Inner criterion satisfied. Exiting optimization loop."
                    )
                    break

                # gradient descent step
                x.sub_(al_grad, alpha=eta)

        # update dual parameters
        with torch.no_grad():
            func_out = obj_fn(x)
            f_diff = f0 - func_out
            lam = lam - mu * f_diff

    return x


def penalty_method(
    x,
    obj_fn,
    max_steps,
    mu,
    eta,
    verbose=False,
):
    """Teleport by solving a simple penalty method.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run.
        mu: the strength of the penalty parameter.
        eta: the step-size to use for each step.

    Returns:
        x: approximate solution to teleportation problem.
    """

    f0 = obj_fn(x).item()

    for t in range(max_steps):
        func_out = obj_fn(x)

        grad = torch.autograd.grad(func_out, x)[
            0
        ]  # ,create_graph=True,retain_graph=True

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            f_diff = f0 - func_out.item()

            if verbose:
                grad_norm = grad @ grad
                tqdm.write(
                    f"Iteration {t}/{max_steps}: Function Diff: {f_diff}, Grad norm: {grad_norm.item()}"
                )

            penalty_grad = (mu * f_diff) * grad - Hv

            x.sub_(penalty_grad, alpha=eta)

    return x


def identity(x, obj_fn):
    return x


def primal_dual_subgrad(
    x,
    obj_fn,
    max_steps,
    eta,
    dual_eta,
    verbose=False,
):
    """Teleport using augmented Lagrangian method.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the augmented Lagrangian
            method.
        max_inner_steps: the maximum number of steps to run gradient descent
            when minimizing the augmented Lagrangian.
        inner_tol: the tolerance for terminating the inner optimization
            method.
        mu: the penalty strength.
        eta: the step-size for the inner optimization method.
        verbose: whether or not to print iteration statistics.

    Returns:
        x: approximate solution to teleportation problem.
    """
    # level set
    f0 = obj_fn(x).item()

    # dual parameters
    lam = 0

    # deviation from level set
    f_diff = 0

    func_out = obj_fn(x)
    for t in tqdm(range(max_steps)):
        # minimize augmented Lagrangian to evaluate primal update

        grad = torch.autograd.grad(func_out, x)[0]

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            grad = -lam * grad - Hv
            grad_norm = grad @ grad

            if verbose:
                tqdm.write(
                    f"Iteration {t}/{max_steps}: Function Diff: {f_diff}, Grad norm: {grad_norm.item()}, Lambda: {lam}"
                )

            # gradient descent step
            x.sub_(grad, alpha=eta / (t + 1))

        func_out = obj_fn(x)

        with torch.no_grad():
            f_diff = f0 - func_out.item()

            # gradient ascent step
            lam = lam + (dual_eta / (t + 1)) * f_diff

    return x
