# %%
# !git clone https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective.git
# !mv Python_Benchmark_Test_Optimization_Function_Single_Objective/src .
# !mv Python_Benchmark_Test_Optimization_Function_Single_Objective/pybenchfunction/ .
# !ls Python_Benchmark_Test_Optimization_Function_Single_Objective/

# %%
import torch
from tqdm.notebook import tqdm
import numpy as np
import pybenchfunction as bench
import time

from src.stoch_functions import LevyN13_i, PermDBeta_i,  Rastrigin_i, RosenBrock_i
from src.torch_functions import Rosenbrock, Rastrigin, IllQuad
from src.algorithms import *
from src.plotting import plot_function_values, plot_level_set_results


def run_methods(x0, func, bench_func, stepsize=0.001, epochs=500, teleport_num=100, teleport_lr=10**-1, teleport_steps=3):
    t0 = time.perf_counter()
    gdtp_x_list, gdtp_fval =run_GD_teleport(func, epochs=epochs, x0=x0, d=d, lr=stepsize, 
                                            teleport_num=teleport_num ,teleport_lr=teleport_lr, teleport_steps=teleport_steps)
    gdtp_time = time.perf_counter() -t0
    t0 = time.perf_counter()
    gd_x_list, gd_fval  =run_GD_teleport(func, epochs=epochs, x0 = x0, d=d, lr=stepsize)
    gd_time = time.perf_counter() -t0
    t0 = time.perf_counter()
    newt_x_list, newt_fval=run_newton(func , epochs=20, x0 = x0, d=d, lr=0.8)
    newt_time = time.perf_counter() -t0
    t0 = time.perf_counter()
    results = {"GDtp": (gdtp_time, gdtp_fval, gdtp_x_list), "GD": (gd_time, gd_fval, gd_x_list),"Newton": (newt_time, newt_fval, newt_x_list ), }
    plot_function_values(bench_func, results, timeplot=False)
    plot_level_set_results(bench_func, results ) 

d =2
x0 = torch.tensor([-0.2, 0.6], requires_grad=True).double()
run_methods(x0, IllQuad,  bench.function.IllQuad(d), stepsize =0.007, epochs=500, teleport_num=100, teleport_lr=10**-1, teleport_steps=3)

x0 = torch.tensor([-2.0, 2.0], requires_grad=True).double() # teleport_steps=1
run_methods(x0, Rosenbrock,  bench.function.Rosenbrock(d), stepsize =0.00125, epochs=5000, teleport_num=1000, teleport_lr=10**-3, teleport_steps=3)

x0 = torch.tensor([-0.2, 0.5], requires_grad=True).double() # teleport_steps=5
run_methods(x0, Rastrigin,  bench.function.Rastrigin(d), stepsize =0.0005, epochs=100, teleport_num=10, teleport_lr=10**0, teleport_steps=3)