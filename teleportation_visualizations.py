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
from src.torch_functions import Rosenbrock, Rastrigin, IllQuad
from src.teleport import (
    linear_sqp,
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
        linear_sqp,
        max_steps=teleport_steps,
        eta=teleport_lr,
        verbose=True,
        normalize=False,
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

    sqp_teleport_norm = partial(
        linear_sqp,
        max_steps=teleport_steps,
        eta=teleport_lr_norm,
        verbose=True,
        normalize=True,
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

    sqp_teleport_sub = partial(
        linear_sqp,
        max_steps=teleport_steps,
        eta=teleport_lr,
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

    # teleport using primal-dual subgrad method
    # primal_dual_teleport = partial(
    #     primal_dual_subgrad,
    #     max_steps=teleport_steps,
    #     eta=1e-4,
    #     dual_eta=1e2,
    #     verbose=False,
    # )
    # t0 = time.perf_counter()
    # pdstp_x_list, pdstp_fval = run_GD_teleport(
    #     func,
    #     primal_dual_teleport,
    #     epochs=epochs,
    #     x0=x0,
    #     d=d,
    #     lr=stepsize,
    #     teleport_num=teleport_num,
    # )
    # pdstp_time = time.perf_counter() - t0

    # teleport using penalty method
    # al_teleport = partial(
    #     penalty_method,
    #     max_steps=teleport_steps,
    #     mu=al_penalty,
    #     eta=1e-10,
    #     verbose=True,
    # )
    # t0 = time.perf_counter()
    # pentp_x_list, pentp_fval = run_GD_teleport(
    #     func,
    #     al_teleport,
    #     epochs=epochs,
    #     x0=x0,
    #     d=d,
    #     lr=stepsize,
    #     teleport_num=teleport_num,
    # )
    # pentp_time = time.perf_counter() - t0

    # teleport using AL method
    # al_teleport = partial(
    #     al_method,
    #     max_steps=teleport_steps,
    #     max_inner_steps=100,
    #     inner_tol=1e-3,
    #     mu=al_penalty,
    #     eta=al_lr * stepsize,
    #     verbose=True,
    # )
    # t0 = time.perf_counter()
    # altp_x_list, altp_fval = run_GD_teleport(
    #     func,
    #     al_teleport,
    #     epochs=epochs,
    #     x0=x0,
    #     d=d,
    #     lr=stepsize,
    #     teleport_num=teleport_num,
    # )
    # altp_time = time.perf_counter() - t0

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
        "SQP_tp": (gdtp_time, gdtp_fval, gdtp_x_list),
        "SQP_Norm_tp": (gd_normtp_time, gd_normtp_fval, gd_normtp_x_list),
        "SQP_Sub_tp": (gd_sub_tp_time, gd_sub_tp_fval, gd_sub_tp_x_list),
        # "PDS_tp": (pdstp_time, pdstp_fval, pdstp_x_list),
        # "AL_tp": (altp_time, altp_fval, altp_x_list),
        # "Pen_tp": (pentp_time, pentp_fval, pentp_x_list),
        "GD": (gd_time, gd_fval, gd_x_list),
        "Newton": (newt_time, newt_fval, newt_x_list),
    }
    plot_function_values(bench_func, results, timeplot=False, show=False)
    plot_level_set_results(bench_func, results, show=False, logscale=logscale)


teleport_steps = 100

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

teleport_steps = 500
x0 = torch.tensor([-2.0, 2.0], requires_grad=True).double()  # teleport_steps=1
run_methods(
    x0,
    Rosenbrock,
    bench.function.Rosenbrock(d),
    stepsize=lr,
    epochs=5000,
    teleport_num=10000,
    teleport_lr=10**-5,
    teleport_lr_norm=10000,
    teleport_steps=teleport_steps,
    logscale=True,
)

x0 = torch.tensor([-0.2, 0.5], requires_grad=True).double()  # teleport_steps=5
run_methods(
    x0,
    Rastrigin,
    bench.function.Rastrigin(d),
    stepsize=1,
    epochs=100,
    teleport_num=1000,
    teleport_lr=1e-5,
    teleport_lr_norm=1e-2,
    teleport_steps=teleport_steps,
)
