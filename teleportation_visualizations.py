# %%
# !git clone https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective.git
# !mv Python_Benchmark_Test_Optimization_Function_Single_Objective/src .
# !mv Python_Benchmark_Test_Optimization_Function_Single_Objective/pybenchfunction/ .
# !ls Python_Benchmark_Test_Optimization_Function_Single_Objective/

# %%

from functools import partial
import time

import torch
from tqdm import tqdm
import numpy as np
import pybenchfunction as bench

from src.stoch_functions import (
    LevyN13_i,
    PermDBeta_i,
    Rastrigin_i,
    RosenBrock_i,
)
from src.torch_functions import (
    Rosenbrock,
    Rastrigin,
    IllQuad,
    Booth,
    BukinN6,
    GoldsteinPrice,
    Himmelblau,
)
from src.teleport import (
    slp,
    normalized_slp,
    al_method,
    penalty_method,
    identity,
    primal_dual_subgrad,
)
from src.algorithms import run_GD_teleport, run_newton
from src.plotting import plot_function_values, plot_level_set_results


def run_methods(
    x0,
    func,
    bench_func,
    stepsize=0.001,
    epochs=500,
    teleport_num=100,
    teleport_lr=10**-1,
    teleport_lr_norm=10**-1,
    teleport_steps=3,
    logscale=False,
):
    # teleport using linear SQP.
    sqp_teleport = partial(
        slp,
        max_steps=teleport_steps,
        lam=teleport_lr,
        verbose=True,
    )
    t0 = time.perf_counter()
    gdtp_x_list, gdtp_fval = run_GD_teleport(
        func,
        sqp_teleport,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
        teleport_num=teleport_num,
    )
    gdtp_time = time.perf_counter() - t0

    # normalized version
    sqp_teleport_norm = partial(
        normalized_slp,
        max_steps=teleport_steps,
        lam=teleport_lr_norm,
        verbose=True,
    )
    t0 = time.perf_counter()
    gd_normtp_x_list, gd_normtp_fval = run_GD_teleport(
        func,
        sqp_teleport_norm,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
        teleport_num=teleport_num,
    )
    gd_normtp_time = time.perf_counter() - t0

    # sub-level set version
    sqp_teleport_sub = partial(
        slp,
        max_steps=teleport_steps,
        lam=teleport_lr,
        verbose=True,
        allow_sublevel=True,
    )
    t0 = time.perf_counter()
    gd_sub_tp_x_list, gd_sub_tp_fval = run_GD_teleport(
        func,
        sqp_teleport_sub,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
        teleport_num=teleport_num,
    )
    gd_sub_tp_time = time.perf_counter() - t0

    # line-search version
    sqp_teleport_ls = partial(
        normalized_slp,
        max_steps=teleport_steps,
        lam=1,
        verbose=True,
        line_search=True,
    )
    t0 = time.perf_counter()
    gd_ls_tp_x_list, gd_ls_tp_fval = run_GD_teleport(
        func,
        sqp_teleport_ls,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
        teleport_num=teleport_num,
    )
    gd_ls_tp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    gd_x_list, gd_fval = run_GD_teleport(
        func,
        identity,
        epochs=epochs,
        x0=x0,
        d=d,
        lr=stepsize,
    )
    gd_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    newt_x_list, newt_fval = run_newton(func, epochs=20, x0=x0, d=d, lr=0.8)
    newt_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    results = {
        "SLP": (gdtp_time, gdtp_fval, gdtp_x_list),
        "SLP (Log Trick)": (gd_normtp_time, gd_normtp_fval, gd_normtp_x_list),
        "SLP (Sub-level)": (gd_sub_tp_time, gd_sub_tp_fval, gd_sub_tp_x_list),
        "SLP (LS)": (gd_ls_tp_time, gd_ls_tp_fval, gd_ls_tp_x_list),
        "GD": (gd_time, gd_fval, gd_x_list),
        "Newton": (newt_time, newt_fval, newt_x_list),
    }
    plot_function_values(bench_func, results, timeplot=False, show=False)
    plot_level_set_results(bench_func, results, show=False, logscale=logscale)


teleport_steps = 500

d = 2
lr = 1
x0 = torch.tensor([-0.1, 2.0], requires_grad=True).double()
run_methods(
    x0,
    IllQuad,
    bench.function.IllQuad(d),
    stepsize=lr,
    epochs=500,
    teleport_num=1000,
    teleport_lr=1e-3,
    teleport_lr_norm=1,
    teleport_steps=teleport_steps,
)

# x0 = torch.tensor([-2.0, 2.0], requires_grad=True).double()  # teleport_steps=1
# run_methods(
#     x0,
#     Rosenbrock,
#     bench.function.Rosenbrock(d),
#     stepsize=lr,
#     epochs=5000,
#     teleport_num=10000,
#     teleport_lr=10**-5,
#     teleport_lr_norm=10000,
#     teleport_steps=teleport_steps,
#     logscale=True,
# )

# x0 = torch.tensor([-0.2, 0.5], requires_grad=True).double()  # teleport_steps=5
# run_methods(
#     x0,
#     Rastrigin,
#     bench.function.Rastrigin(d),
#     stepsize=1,
#     epochs=100,
#     teleport_num=1000,
#     teleport_lr=1e-5,
#     teleport_lr_norm=1e-2,
#     teleport_steps=teleport_steps,
# )


# x0 = torch.tensor([7.5, -4.0], requires_grad=True).double()  # teleport_steps=5
# run_methods(
#     x0,
#     Booth,
#     bench.function.Booth(d),
#     stepsize=1,
#     epochs=100,
#     teleport_num=1000,
#     teleport_lr=1e-2,
#     teleport_lr_norm=1e1,
#     teleport_steps=teleport_steps,
# )


# x0 = torch.tensor([-8.0, 1.0], requires_grad=True).double()  # teleport_steps=5
# run_methods(
#     x0,
#     BukinN6,
#     bench.function.BukinN6(d),
#     stepsize=1,
#     epochs=100,
#     teleport_num=1000,
#     teleport_lr=1e-2,
#     teleport_lr_norm=1e1,
#     teleport_steps=teleport_steps,
# )


# x0 = torch.tensor([0.0, 0.0], requires_grad=True).double()  # teleport_steps=5
# run_methods(
#     x0,
#     GoldsteinPrice,
#     bench.function.GoldsteinPrice(d),
#     stepsize=1,
#     epochs=100,
#     teleport_num=1000,
#     teleport_lr=1e-9,
#     teleport_lr_norm=1e-2,
#     teleport_steps=teleport_steps,
#     logscale=True,
# )


# x0 = torch.tensor([0.0, 2.0], requires_grad=True).double()  # teleport_steps=5
# run_methods(
#     x0,
#     Himmelblau,
#     bench.function.Himmelblau(d),
#     stepsize=1,
#     epochs=100,
#     teleport_num=1000,
#     teleport_lr=1e-4,
#     teleport_lr_norm=1e-0,
#     teleport_steps=teleport_steps,
#     logscale=True,
# )
