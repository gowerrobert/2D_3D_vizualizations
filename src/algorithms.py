import torch
from tqdm import tqdm
import numpy as np
import functorch

EPS_GLOBAL = 10**-18
max_tries = 100


def armijo_cond(f_next, f, grad, d, alpha=0.5):
    """Check Armijo condition."""
    return f_next <= f + alpha * grad @ d


def run_algorithm(
    computeValue, step_function, epochs=20, d=2, lr=1.0, x0=None
):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)
    np.random.seed(0)

    idx = list(range(d))
    x_list = []
    y_list = []
    fval = []
    for ep in tqdm(range(epochs)):
        np.random.shuffle(idx)

        for i in idx:
            x = step_function(x, i, computeValue, lr)
        fnew = np.mean([computeValue(i, x).item() for i in range(d)])
        if np.isnan(fnew):
            break
        fval.append(fnew)
        x_list.append(x[0].item())
        y_list.append(x[1].item())
        if fval[-1] <= EPS_GLOBAL:
            break

    return [x_list, y_list], fval


def step_SP2plus(x, i, computeValue, lr, nosp1=True):
    funci = lambda x: computeValue(i, x)
    fi = computeValue(i, x)
    grad = torch.autograd.grad(fi, x, create_graph=True, retain_graph=True)[0]
    hessvgrad = torch.autograd.functional.hvp(
        funci, x, grad, create_graph=True
    )[1]
    # print(hessvgrad.shape)
    # hessian_grad = torch.autograd.grad(grads, self.params, grad_outputs=grads)
    with torch.no_grad():
        gradnormsqr = torch.norm(grad) ** 2
        # if gradnormsqr > 1e-18:
        sps_step = fi / gradnormsqr
        x.sub_(sps_step * grad, alpha=lr)
        # gdiffHvg = grad -hessvgrad*fi/gradnormsqr
        if nosp1:
            gdiffHvg = torch.sub(grad, hessvgrad, alpha=sps_step)
            # gdiffHvg = [g - fi*hg/gradnormsqr for g,hg in zip(grad, hessvgrad)]   # Maybe need this instead?
            if torch.norm(gdiffHvg) ** 2 > 1e-10:
                x.sub_(
                    0.5
                    * (sps_step**2)
                    * gdiffHvg
                    * torch.dot(grad, gdiffHvg)
                    / (torch.norm(gdiffHvg) ** 2),
                    alpha=lr,
                )
    return x


def run_SP2plus(computeValue, epochs=20, d=2, lr=1.0, x0=None, nosp1=True):
    if nosp1:
        step_func = lambda x, i, computeValue, lr: step_SP2plus(
            x, i, computeValue, lr, nosp1=True
        )
    else:
        step_func = lambda x, i, computeValue, lr: step_SP2plus(
            x, i, computeValue, lr, nosp1=False
        )
    return run_algorithm(
        computeValue, step_func, epochs=epochs, d=d, lr=lr, x0=x0
    )


def step_SP2(x, i, computeValue, lr, inner_steps):
    funci = lambda x: computeValue(i, x)
    fi = computeValue(i, x)
    grad = torch.autograd.grad(fi, x, create_graph=True, retain_graph=True)[0]

    # hessian = torch.autograd.functional.hessian(pow_reducer, inputs)
    # print(hessvgrad.shape)
    # hessian_grad = torch.autograd.grad(grads, self.params, grad_outputs=grads)
    w = torch.clone(x)
    for j in range(inner_steps):
        wdiff = torch.sub(w, x)
        hessvgrad = torch.autograd.functional.hvp(
            funci, x, wdiff, create_graph=True
        )[1]
        with torch.no_grad():
            q = fi + torch.dot(grad, wdiff) + 0.5 * torch.dot(wdiff, hessvgrad)
            nablaq = torch.add(grad, hessvgrad)
            nablaqnorm = torch.norm(nablaq)
            if nablaqnorm < 1e-14:
                break
            w.sub_(nablaq, alpha=lr * q / nablaqnorm**2)
    with torch.no_grad():
        # x = torch.clone(w)
        x = w
    return x


def run_SP2(computeValue, epochs=20, d=2, lr=1.0, inner_steps=10, x0=None):
    step_func = lambda x, i, computeValue, lr: step_SP2(
        x, i, computeValue, lr, inner_steps=inner_steps
    )
    return run_algorithm(
        computeValue, step_func, epochs=epochs, d=d, lr=lr, x0=x0
    )


def step_SGD(x, i, computeValue, lr):
    fi = computeValue(i, x)
    grad = torch.autograd.grad(fi, x, create_graph=True, retain_graph=True)[0]

    with torch.no_grad():
        x.sub_(grad, alpha=lr)


# def run_SGD(computeValue, epochs=20, d=2, lr =1.0, x0=None):
#     # import pdb; pdb.set_trace()
#     return run_algorithm(computeValue, step_SGD, epochs=epochs, d=d, lr =lr, x0=x0)


def run_GD_teleport(
    computeValue,
    teleport_fn,
    epochs=20,
    d=2,
    lr=1.0,
    x0=None,
    teleport_num=-1,
    beta=0.8,
):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)
    np.random.seed(0)

    x_list = []
    y_list = []
    fval = []
    func_out = computeValue(x)
    fval.append(func_out.item())
    x_list.append(x[0].item())
    y_list.append(x[1].item())

    teleport_path = []

    for ep in tqdm(range(epochs)):
        if fval[-1] <= EPS_GLOBAL:
            break
        
        if teleport_num != -1:
            if (ep) % teleport_num == 0:
                # run teleport procedure
                x, teleport_path = teleport_fn(x, computeValue)

                func_out = computeValue(x)
                fval.append(computeValue(x).item())
                x_list.append(x[0].item())
                y_list.append(x[1].item())

        func_out = computeValue(x)
        grad = torch.autograd.grad(func_out, x)[
            0
        ]  # ,create_graph=True,retain_graph=True

        with torch.no_grad():
            x_next = x.sub(grad, alpha=lr)
            f_next = computeValue(x_next)

            for i in range(max_tries):
                if armijo_cond(f_next, func_out, grad, -lr * grad):
                    break

                lr = lr * beta
                x_next = x.sub(grad, alpha=lr)
                f_next = computeValue(x_next)

            x.sub_(grad, alpha=lr)

            # try to increase step-size.
            lr = lr / beta

        fval.append(func_out.item())
        x_list.append(x[0].item())
        y_list.append(x[1].item())

    print("gd tp-", teleport_num, " loss: ", fval[-1])
    return [x_list, y_list], fval, teleport_path


def run_SGD(computeValue, epochs=20, d=2, lr=1.0, x0=None):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)
    np.random.seed(0)

    idx = list(range(d))
    x_list = []
    y_list = []
    fval = []
    for ep in tqdm(range(epochs)):
        np.random.shuffle(idx)

        for i in idx:
            fi = computeValue(i, x)
            grad = torch.autograd.grad(fi, x)[
                0
            ]  # ,create_graph=True,retain_graph=True

            with torch.no_grad():
                x.sub_(grad, alpha=lr)

        fval.append(np.mean([computeValue(i, x).item() for i in range(d)]))
        x_list.append(x[0].item())
        y_list.append(x[1].item())
        if fval[-1] <= EPS_GLOBAL:
            break
    return [x_list, y_list], fval


def run_newton(computeValue, epochs=20, d=2, lr=1.0, x0=None):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)

    np.random.seed(0)

    idx = list(range(d))
    x_list = []
    y_list = []
    fval = []
    for ep in tqdm(range(epochs)):
        np.random.shuffle(idx)

        grad = torch.zeros(d)
        hess = np.zeros([d, d])

        func_out = computeValue(x)
        grad = torch.autograd.grad(
            func_out, x, create_graph=True, retain_graph=True
        )[0]

        fval.append(func_out.item())
        x_list.append(x[0].item())
        y_list.append(x[1].item())
        if fval[-1] <= EPS_GLOBAL:
            break

        hess = np.zeros([d, d])
        for j in range(d):
            Hj = torch.autograd.grad(grad[j], x, retain_graph=True)[0]
            hess[:, j] = Hj
        hess = 0.5 * (hess + hess.T)  # symmetrize in case of rounding errors?

        for i in idx:
            hess[i][i] = hess[i][i] + 10**-8

        with torch.no_grad():
            newton_step = torch.tensor(np.linalg.solve(hess, grad.numpy()))
            x.sub_(newton_step, alpha=lr)

    return [x_list, y_list], fval


def run_newton_stoch(computeValue, epochs=20, d=2, lr=1.0, x0=None):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)

    np.random.seed(0)

    idx = list(range(d))
    x_list = []
    y_list = []
    fval = []
    for ep in tqdm(range(epochs)):
        np.random.shuffle(idx)

        grad = torch.zeros(d)
        hess = np.zeros([d, d])

        for i in idx:
            func_out = computeValue(i, x)
            grad = torch.autograd.grad(
                func_out, x, create_graph=True, retain_graph=True
            )[0]
            # populate the hessian for fi
            hess = np.zeros([d, d])
            for j in range(d):
                Hj = torch.autograd.grad(grad[j], x, retain_graph=True)[0]
                hess[:, j] = Hj
            hess = 0.5 * (
                hess + hess.T
            )  # symmetrize in case of rounding errors?

        for i in idx:
            hess[i][i] = hess[i][i] + 10**-8

        with torch.no_grad():
            newton_step = torch.tensor(np.linalg.solve(hess, grad.numpy()))
            # newton_step = torch.linalg.solve(grad, hess)
            # newton_step = torch.tensor(np.linalg.inv(hess) @ grad.numpy())
            x.sub_(newton_step, alpha=lr)
        fval.append(func_out.item())
        x_list.append(x[0].item())
        y_list.append(x[1].item())
        if fval[-1] <= EPS_GLOBAL:
            break

    return [x_list, y_list], fval


def adam(
    computeValue,
    epochs=20,
    d=2,
    lr=0.1,
    x0=None,
    beta1=0.9,
    beta2=0.999,
    eps=10 ** (-8.0),
):
    torch.manual_seed(0)
    if x0 is None:
        x = torch.randn(d, requires_grad=True).double() * 1
    else:
        x = torch.clone(x0)
    m = torch.zeros_like(x)
    # Exponential moving average of squared gradient values
    v = torch.zeros_like(x)
    np.random.seed(0)

    idx = list(range(d))
    x_list = []
    y_list = []
    fval = []
    for ep in tqdm(range(epochs)):
        np.random.shuffle(idx)
        count = 1
        for i in idx:
            fi = computeValue(i, x)
            grad = torch.autograd.grad(fi, x)[
                0
            ]  # ,create_graph=True,retain_graph=True

            with torch.no_grad():
                # m = beta1*m + (1-beta1)*grad
                # m = beta1*m + (1-beta1)*grad
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                # m.mul_(beta1).add_(1 - beta1, grad)
                # v.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                iter_count = d * ep + count
                bias_correction1 = 1 - beta1**iter_count
                bias_correction2 = 1 - beta2**iter_count
                denom = (v.sqrt() / bias_correction2).add_(eps)
                step_size = lr / bias_correction1
                x.addcdiv_(m, denom, value=-step_size)
                # step_size = lr * np.sqrt(bias_correction2) / bias_correction1
                # x.sub_(m, alpha=step_size)
                count = count + 1

        fval.append(np.mean([computeValue(i, x).item() for i in range(d)]))
        x_list.append(x[0].item())
        y_list.append(x[1].item())
        if fval[-1] <= EPS_GLOBAL:
            break
    return [x_list, y_list], fval


# def adam(loss, regularizer, data, label, lr, reg, epoch, x_0, tol=None,
#          beta1 =0.9, beta2 =0.999, eps = 10**(-8.0), verbose = False):
#     """Adam method"""
#     n, d = data.shape
#     x = x_0.copy()
#     m = np.zeros(d)
#     v = np.zeros(d)
#     # init loss
#     loss_x0 = np.mean(loss.val(label, data @ x), axis=0) + reg * regularizer.val(x)
#     # init grad
#     g = np.mean(loss.prime(label, data @ x).reshape(-1, 1) * data, axis=0) + reg * regularizer.prime(x)
#     norm_records = [np.sqrt(g @ g)]
#     loss_records = [1.0]

#     iis = np.random.randint(0, n, n * epoch + 1)
#     cnt = 0
#     time_records, epoch_running_time, total_running_time = [0.0], 0.0, 0.0
# #    total_start_time = time.time()
#     for idx in range(len(iis)):
#         i = iis[idx]

#         start_time = time.time()
#         # gradient of (i-1)-th data point
#         g = loss.prime(label[i], data[i, :] @ x) * data[i, :] + reg * regularizer.prime(x)
#         m = beta1*m +(1-beta1)*g
#         v = beta2*v +(1-beta2)*(g*g)
#         mhat= m/(1-beta1**(idx+1))
#         vhat= v/(1-beta2**(idx+1))
#         direction = lr*mhat/(np.sqrt(vhat) +eps)
#         # update
#         x -= direction
#         epoch_running_time += time.time() - start_time

#         if (idx + 1) % n == 0:
#             cnt += 1
#             epoch_running_time = 0.0
#             if tol is not None and norm_records[-1] <= tol:
#                 print("ADAM reaches the tolerance: " + str(tol))
#                 return x, norm_records, loss_records, time_records
#             #    total_running_time += time.time() - total_start_time
# #    print("adam:")
# #    print(total_running_time)
#     return x, norm_records, loss_records, time_records
