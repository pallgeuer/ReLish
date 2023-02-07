#!/usr/bin/env python3
# Test various classification losses and how to best implement them

# Imports
import math
import inspect
import argparse
import itertools
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import loss_funcs  # noqa

# Constants
DEFAULT_EPS = 0.2
DEFAULT_TAU = loss_funcs.DEFAULT_TAU
FIGSIZE = (9.60, 4.55)
FIGDPI = 100

#
# Data types
#

# Common loss term data class
@dataclasses.dataclass(frozen=True)
class LossCommon:
	K: int
	eps: float
	z: torch.Tensor
	p: torch.Tensor
	q: torch.Tensor

# Loss result data class
@dataclasses.dataclass(frozen=True)
class LossResult:
	M: LossCommon
	x: torch.Tensor
	L: torch.Tensor
	dxdt: torch.Tensor
	dzdt: torch.Tensor
	dLdt: torch.Tensor

#
# Main
#

# Main function
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda', help='Device to perform calculations on')
	parser.add_argument('--eps', type=float, default=DEFAULT_EPS, help='Value of epsilon to use (for all but MSE, NLL, Focal)')
	parser.add_argument('--losses', type=str, nargs='+', default=list(loss_funcs.LOSSES.keys()), help='List of losses to consider')
	parser.add_argument('--gradcheck', action='store_true', help='Perform grad check on custom autograd modules')
	parser.add_argument('--evalx', type=float, nargs='+', help='Evaluate case where raw logits are as listed (first is true class)')
	parser.add_argument('--evalp', type=float, nargs='+', help='Evaluate case where probabilities are as listed (first is true class, rescaled to sum to 1)')
	parser.add_argument('--plot', type=str, nargs='+', help='Situation(s) to provide plots for')
	parser.add_argument('--plot_points', type=int, default=401, help='Number of points to use for plotting')
	parser.add_argument('--plot_classes', type=int, default=10, help='Number of classes to use for plotting')

	args = parser.parse_args()
	args.device = torch.device(args.device)

	if args.gradcheck:
		gradcheck(args)

	if args.evalx:
		evalx(args.evalx, args)

	if args.evalp:
		evalx([math.log(item) for item in args.evalp], args)

	if args.plot:
		for situation in args.plot:
			plot_situation(situation, args)

#
# Losses
#

# Create a loss module from a loss factory that can be used as Callable[[logits_tensor, target_tensor], loss_tensor]
def create_loss_module(loss_factory, M):
	params = dict(num_classes=M.K, normed=True, reduction='none', eps=M.eps)
	factory_param_keys = inspect.signature(loss_factory).parameters.keys()
	params = {key: value for key, value in params.items() if key in factory_param_keys}
	return loss_factory(**params)

# Evaluate a loss module on a logits tensor, assuming the first logit is always the true one
def eval_loss_module(loss_module, x):
	return loss_module(x, torch.zeros(size=(x.shape[0], *x.shape[2:]), dtype=torch.long, device=x.device))

# Evaluate a loss module and its gradients on a logits tensor
def eval_loss_module_grad(loss_module, x, M):

	x = x.detach().requires_grad_()
	L = eval_loss_module(loss_module, x)
	x.grad = None
	L.sum().backward()
	dLdx: torch.Tensor = x.grad  # noqa
	dxdt = -dLdx
	dzdt = dxdt[:, 1:] - dxdt[:, :1]
	dLdt = -dLdx.square().sum(dim=1, keepdim=True)

	loss_module.reduction = 'sum'
	xx = x.detach().requires_grad_()
	LL = eval_loss_module(loss_module, xx)
	xx.grad = None
	LL.backward()
	assert torch.allclose(x.grad, xx.grad)  # noqa

	loss_module.reduction = 'mean'
	xx = x.detach().requires_grad_()
	LL = eval_loss_module(loss_module, xx)
	xx.grad = None
	LL.mul(x.shape[0]).backward()
	assert torch.allclose(x.grad, xx.grad, atol=1e-5, rtol=1e-5)  # noqa

	loss_module.reduction = 'none'

	return LossResult(M=M, x=x, L=L, dxdt=dxdt, dzdt=dzdt, dLdt=dLdt)

# Calculate common loss terms
def loss_common(x, eps):
	K = x.shape[1]
	eta = math.log(1 - eps) - math.log(eps / (K - 1))
	z = x[:, 1:] - x[:, :1] + eta
	p = F.softmax(x, dim=1)
	q = p.clone()
	q[:, :1] -= 1 - eps
	q[:, 1:] -= eps / (K - 1)
	return LossCommon(K=K, eps=eps, z=z, p=p, q=q)

#
# Grad check
#

# Action: Perform grad check on custom autograd modules
def gradcheck(args):
	print("Performing grad checks")
	print("Checking DualLogSoftmaxFunction...")
	for dim in range(4):
		assert torch.autograd.gradcheck(lambda x: loss_funcs.DualLogSoftmaxFunction.apply(x, dim), torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double, device=args.device), check_grad_dtypes=True, check_batched_grad=False)
	print("Done")
	print()

#
# Evaluate
#

# Action: Evaluate losses on a list of logits
def evalx(x, args):
	x = torch.tensor([x], device=args.device)
	M = loss_common(x, args.eps)
	for loss_key in args.losses:
		loss_name, loss_factory = loss_funcs.LOSSES[loss_key.lower()]
		print(f"EVALUATE: {loss_name}")
		loss_module = create_loss_module(loss_factory, M)
		print(f"LOSS MODULE: {loss_module}")
		result = eval_loss_module_grad(loss_module, x, M)
		print_vec('   x', result.x[0])
		print_vec('   p', result.M.p[0])
		print_vec('   L', result.L[0])
		print_vec('dxdt', result.dxdt[0])
		print_vec('dzdt', result.dzdt[0])
		print_vec('dLdt', result.dLdt[0])
		print()

# Print a tensor vector
def print_vec(name, vec, fmt='7.4f'):
	if vec.ndim == 0:
		print(f"{name} = {vec:{fmt}}")
	else:
		print(f"{name} = [{', '.join(f'{item:{fmt}}' for item in vec)}]")

#
# Plot
#

# Action: Plot a situation
def plot_situation(situation, args):
	sit_name, sit_desc, sit_var, sit_gen = SITUATION_MAP[situation.lower()]
	print(f"SITUATION:   {sit_name}")
	print(f"DESCRIPTION: {sit_desc}")
	v, x = sit_gen(args)
	generate_plots(v, x, sit_var, sit_name, args)
	print()

# Generate the plots for a situation
def generate_plots(v, x, sit_var, sit_name, args):

	M = loss_common(x, args.eps)

	fig, axs = plt.subplots(2, 2, figsize=FIGSIZE, dpi=FIGDPI)
	fig.suptitle(f"{sit_name}: Logits and probabilities vs {sit_var}")
	for i, (data, label) in enumerate(((x[:, :3], 'x'), (M.z[:, :2], 'z = xf - xT + eta'), (M.p[:, :3], 'p'), (M.q[:, :3], 'q = p - ptarget'))):
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.plot(v, data.detach().cpu().numpy())
		ax.grid(visible=True)
		ax.autoscale(axis='x', tight=True)
		ax.legend([label[0] + sub for sub in ('T', 'F', 'K')[-data.shape[1]:]], loc='best')
		ax.set_title(label)
	fig.tight_layout()

	figX, axsX = plt.subplots(1, 3, figsize=FIGSIZE, dpi=FIGDPI)
	figZ, axsZ = plt.subplots(1, 2, figsize=FIGSIZE, dpi=FIGDPI)
	figL, axsL = plt.subplots(1, 2, figsize=FIGSIZE, dpi=FIGDPI)
	figX.suptitle(f"{sit_name}: Logit update rate vs {sit_var}")
	figZ.suptitle(f"{sit_name}: Relative logit update rate vs {sit_var}")
	figL.suptitle(f"{sit_name}: Loss value and rate vs {sit_var}")
	for loss_key in args.losses:
		loss_name, loss_factory = loss_funcs.LOSSES[loss_key.lower()]
		loss_module = create_loss_module(loss_factory, M)
		print(f"LOSS MODULE: {loss_module}")
		result = eval_loss_module_grad(loss_module, x, M)
		for i in range(3):
			axsX[i].plot(v, result.dxdt[:, i].detach().cpu().numpy(), label=loss_name)
		for i in range(2):
			axsZ[i].plot(v, result.dzdt[:, i].detach().cpu().numpy(), label=loss_name)
		for i, data in enumerate((result.L, result.dLdt)):
			axsL[i].plot(v, data.detach().cpu().numpy(), label=loss_name)
	for ax, title in zip(
			itertools.chain(axsX.flatten(), axsZ.flatten(), axsL.flatten()),
			('dxT/dt', 'dxF/dt', 'dxK/dt', 'dzF/dt', 'dzK/dt', 'L', 'dL/dt'),
	):
		ax.grid(visible=True)
		ax.autoscale(axis='x', tight=True)
		ax.legend(loc='best')
		ax.set_title(title)
	figX.tight_layout()
	figZ.tight_layout()
	figL.tight_layout()

	plt.show()

#
# Situations
#

# Situation: All xf are zero and xT varies
def gen_equal_false(args):
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.zeros((args.plot_points, args.plot_classes), device=args.device)
	x[:, 0] = v
	return v, x

# Situation: xT varies for xf1 zero and all other xf significantly negative
def gen_two_way(args):
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.full((args.plot_points, args.plot_classes), fill_value=-30.0, device=args.device)
	x[:, 0] = v
	x[:, 1] = 0
	return v, x

# Situation: pT = const and the rest is divided amongst pF and equal pK
def gen_split(args, pT):
	K = args.plot_classes
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.log((1 / pT - 1) / (K - 2 + torch.exp(v))).repeat(K, 1).T
	x[:, 0] = 0
	x[:, 1] = torch.log((1 / pT - 1) / (1 + (K - 2) * torch.exp(-v)))
	return v, x

# Situation map
SITUATION_MAP = dict(
	equal_false=('Equal false', 'All xf are zero and xT varies', 'xT', gen_equal_false),
	two_way=('2-way', 'xT varies for xf1 zero and all other xf significantly negative', 'xT', gen_two_way),
	split_high=('High split', 'pT = 1-eps/2 and the rest is divided amongst pF and equal pK', 'xd = xF - xK', lambda args: gen_split(args, 1 - args.eps / 2)),
	split_equil=('Equilibrium split', 'pT = 1-eps and the rest is divided amongst pF and equal pK', 'xd = xF - xK', lambda args: gen_split(args, 1 - args.eps)),
	split_med=('Medium split', 'pT = 0.9*(1-eps) and the rest is divided amongst pF and equal pK', 'xd = xF - xK', lambda args: gen_split(args, 0.9 * (1 - args.eps))),
	split_low=('Low split', 'pT = 1/K and the rest is divided amongst pF and equal pK', 'xd = xF - xK', lambda args: gen_split(args, 1 / args.plot_classes)),
)

# Run main function
if __name__ == "__main__":
	main()
# EOF
