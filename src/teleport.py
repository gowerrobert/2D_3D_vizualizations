"""
Functions for solving teleportation problems.
"""


import torch
from tqdm import tqdm
import numpy as np


def linear_sqp(
    x,
    obj_fn,
    max_steps,
    eta,
    normalize=True,
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
            f_diff = f0 - func_out.item()

            if verbose:
                grad_norm = grad @ grad
                tqdm.write(
                    f"Iteration {t}/{max_steps}: Function Diff: {f_diff}, Grad norm: {grad_norm.item()}"
                )

            denom = torch.inner(grad, grad)

            eta_step = eta
            if normalize:
                eta_step = eta / denom

            x.add_(Hv, alpha=eta_step)

            negstep = (f_diff - eta_step * torch.inner(Hv, grad)) / denom

            # unnecessary step-size
            x.add_(grad, alpha=negstep.item())

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
