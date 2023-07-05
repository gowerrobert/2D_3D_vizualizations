"""
Functions for solving teleportation problems.
"""


import torch
from tqdm import tqdm
import numpy as np

# global parameters

ALPHA = 1e-3
BETA = 0.8
BETA_INV = 1.25
GRAD_TOL = 1e-10
CONST_TOL = 1e-10
MAX_BACKTRACKS = 100
MU_SCALE = 2


def slp(
    x,
    obj_fn,
    max_steps,
    lam,
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
        lam: the step-size to use for each step of linear SQP.

    Returns:
        x: approximate solution to teleportation problem.
    """
    x0 = x.clone()
    f0 = obj_fn(x).item()
    lam0 = lam
    mu = torch.tensor([0])

    f_next = None
    grad_next = None

    f_diff_next = None
    grad_norm_next = None

    if allow_sublevel:
        penalty_fn = lambda z: torch.maximum(-z, torch.tensor([0]))
    else:
        penalty_fn = torch.abs

    for t in tqdm(range(max_steps)):
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

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            if f_diff_next is None:
                f_diff = f0 - func_out
                grad_norm = grad @ grad
            else:
                grad_norm = grad_norm_next
                f_diff = f_diff_next

            if verbose:
                tqdm.write(
                    f"Iteration {t+1}/{max_steps}: Function Diff: {f_diff.item()}, Grad norm: {grad_norm.item()}"
                )

            vHv = torch.inner(Hv, grad)
            vHv_g = vHv / grad_norm

            # check termination conditions
            proj = vHv_g * grad - Hv
            if proj @ proj <= GRAD_TOL and penalty_fn(f_diff) <= CONST_TOL:
                tqdm.write(
                    "KKT conditions approximately satisfied. Terminating SLP procedure."
                )
                return x

            # estimate penalty strength for line-search merit function
            mu = MU_SCALE * torch.abs(vHv_g)

            # housekeeping for line-search
            x_prev = torch.tensor(x, requires_grad=True)

        for i in range(MAX_BACKTRACKS):
            # evaluate update
            with torch.no_grad():
                x.add_(Hv, alpha=lam)

                # skip projection since step if allowed
                if not allow_sublevel or lam * vHv - f_diff > 0:
                    negstep = (f_diff - lam * vHv) / grad_norm
                    x.add_(grad, alpha=negstep.item())

            if not line_search:
                # accept step-size immediately
                break

            else:
                # proceed with line-search
                f_next = obj_fn(x)
                grad_next = torch.autograd.grad(f_next, x)[0]

                with torch.no_grad():
                    # quantities will be re-used for next step
                    # if the step-size is accepted.
                    grad_norm_next = grad_next @ grad_next
                    f_diff_next = f0 - f_next
                    x_diff = x - x_prev

                    LHS = -grad_norm_next + 2 * mu * penalty_fn(f_diff_next)
                    RHS = (
                        -grad_norm
                        + 2 * ALPHA * Hv @ x_diff
                        + 2 * (1 - ALPHA) * mu * penalty_fn(f_diff)
                    )

                    if LHS <= RHS:
                        break

                    # reset and try with smaller step-size.
                    lam = lam * BETA
                    x[:] = x_prev[:]

        # report if line-search failed
        if i == MAX_BACKTRACKS - 1:
            tqdm.write(
                "WARNING: Line-search failed to return feasible step-size."
            )
            lam = lam0

        # try to increase step-size if merit bound isn't too tight.
        if line_search and RHS / LHS >= 5.0:
            lam = lam * BETA_INV

    return x


def normalized_slp(
    x,
    obj_fn,
    max_steps,
    lam,
    allow_sublevel=False,
    line_search=False,
    verbose=False,
):
    """Teleport by solving successive linear approximations.

    This version normalizes the

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run the linear SQP method.
        lam: the step-size to use for each step of linear SQP.

    Returns:
        x: approximate solution to teleportation problem.
    """
    x0 = x.clone()
    f0 = obj_fn(x).item()
    lam0 = lam
    mu = torch.tensor([0])

    f_next = None
    grad_next = None

    f_diff_next = None
    grad_norm_next = None

    if allow_sublevel:
        penalty_fn = lambda z: torch.maximum(-z, torch.tensor([0]))
    else:
        penalty_fn = torch.abs

    for t in tqdm(range(max_steps)):
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

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            if f_diff_next is None:
                f_diff = f0 - func_out
                grad_norm = grad @ grad
            else:
                grad_norm = grad_norm_next
                f_diff = f_diff_next

            if verbose:
                tqdm.write(
                    f"Iteration {t+1}/{max_steps}: Function Diff: {f_diff.item()}, Grad norm: {grad_norm.item()}"
                )

            Hv = Hv / grad_norm
            vHv = torch.inner(Hv, grad)
            vHv_g = vHv / grad_norm

            # check termination conditions
            proj = vHv_g * grad - Hv
            if proj @ proj <= GRAD_TOL and penalty_fn(f_diff) <= CONST_TOL:
                tqdm.write(
                    "KKT conditions approximately satisfied. Terminating SLP procedure."
                )
                return x

            # estimate penalty strength for line-search merit function
            mu = MU_SCALE * torch.abs(vHv_g)

            # housekeeping for line-search
            x_prev = torch.tensor(x, requires_grad=True)

        for i in range(MAX_BACKTRACKS):
            # evaluate update
            with torch.no_grad():
                x.add_(Hv, alpha=lam)

                # skip projection since step if allowed
                if not allow_sublevel or lam * vHv - f_diff > 0:
                    negstep = (f_diff - lam * vHv) / grad_norm
                    x.add_(grad, alpha=negstep.item())

            if not line_search:
                # accept step-size immediately
                break

            else:
                # proceed with line-search
                f_next = obj_fn(x)
                grad_next = torch.autograd.grad(f_next, x)[0]

                with torch.no_grad():
                    # quantities will be re-used for next step
                    # if the step-size is accepted.
                    grad_norm_next = grad_next @ grad_next
                    f_diff_next = f0 - f_next
                    x_diff = x - x_prev

                    LHS = -torch.log(grad_norm_next) + 2 * mu * penalty_fn(
                        f_diff_next
                    )
                    RHS = (
                        -torch.log(grad_norm)
                        + 2 * ALPHA * Hv @ x_diff
                        + 2 * (1 - ALPHA) * mu * penalty_fn(f_diff)
                    )

                    if LHS <= RHS:
                        break

                    # reset and try with smaller step-size.
                    lam = lam * BETA
                    x[:] = x_prev[:]

        # report if line-search failed
        if i == MAX_BACKTRACKS - 1:
            tqdm.write(
                "WARNING: Line-search failed to return feasible step-size."
            )
            lam = lam0

        # try to increase step-size if merit bound isn't too tight.
        if line_search and RHS / LHS >= 5.0:
            lam = lam * BETA_INV

    return x


def al_method(
    x,
    obj_fn,
    max_steps,
    max_inner_steps,
    inner_tol,
    mu,
    lam,
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
        lam: the step-size for the inner optimization method.
        verbose: whether or not to print iteration statistics.

    Returns:
        x: approximate solution to teleportation problem.
    """
    # level set
    f0 = obj_fn(x).item()

    # dual parameters
    eta = 0

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

                al_grad = (eta + mu * f_diff) * grad - Hv
                grad_norm = al_grad @ al_grad

                if verbose:
                    tqdm.write(
                        f"Iteration {t+1}/{max_inner_steps}: Function Diff: {f_diff}, AL Grad norm: {grad_norm.item()}"
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
            eta = eta - mu * f_diff

    return x


def penalty_method(
    x,
    obj_fn,
    max_steps,
    mu,
    lam,
    verbose=False,
):
    """Teleport by solving a simple penalty method.

    Params:
        x: the starting point for optimization.
        obj_fn: callable function which evaluates the objective.
            Must support backward passes.
        max_steps: the maximum number of steps to run.
        mu: the strength of the penalty parameter.
        lam: the step-size to use for each step.

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
                    f"Iteration {t+1}/{max_steps}: Function Diff: {f_diff}, Grad norm: {grad_norm.item()}"
                )

            penalty_grad = (mu * f_diff) * grad - Hv

            x.sub_(penalty_grad, alpha=lam)

    return x


def identity(x, obj_fn):
    return x


def primal_dual_subgrad(
    x,
    obj_fn,
    max_steps,
    lam,
    dual_lam,
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
        lam: the step-size for the inner optimization method.
        verbose: whether or not to print iteration statistics.

    Returns:
        x: approximate solution to teleportation problem.
    """
    # level set
    f0 = obj_fn(x).item()

    # dual parameters
    eta = 0

    # deviation from level set
    f_diff = 0

    func_out = obj_fn(x)
    for t in tqdm(range(max_steps)):
        # minimize augmented Lagrangian to evaluate primal update

        grad = torch.autograd.grad(func_out, x)[0]

        Hv = torch.autograd.functional.hvp(obj_fn, x, grad)[1]

        with torch.no_grad():
            grad = -eta * grad - Hv
            grad_norm = grad @ grad

            if verbose:
                tqdm.write(
                    f"Iteration {t+1}/{max_steps}: Function Diff: {f_diff}, Grad norm: {grad_norm.item()}, Eta: {eta}"
                )

            # gradient descent step
            x.sub_(grad, alpha=lam / (t + 1))

        func_out = obj_fn(x)

        with torch.no_grad():
            f_diff = f0 - func_out.item()

            # gradient ascent step
            eta = eta + (dual_lam / (t + 1)) * f_diff

    return x
