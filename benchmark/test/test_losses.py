#!/usr/bin/env python3
# Test various classification losses and how to best implement them

# Imports
import math
import argparse
import itertools
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Constants
DEFAULT_EPS = 0.20
AUTO_TAU = 0
FIGSIZE = (9.6, 5.15)

#
# Data types
#

@dataclasses.dataclass(frozen=True)
class LossCommon:
	K: int
	eps: float
	tau: float
	eta: float
	x: torch.Tensor
	z: torch.Tensor
	p: torch.Tensor
	q: torch.Tensor

@dataclasses.dataclass(frozen=True)
class LossResult:
	M: LossCommon
	x: torch.Tensor
	L: torch.Tensor
	dxdt: torch.Tensor
	dzdt: torch.Tensor
	dLdt: torch.Tensor

#
# Losses
#

# Common loss components
def loss_common(x, eps=DEFAULT_EPS, tau=AUTO_TAU):
	K = x.shape[1]
	if tau == AUTO_TAU:
		tau = (1 - 1 / (K * (1 - eps))) ** 2
	eta = math.log(1 - eps) - math.log(eps / (K - 1))
	z = x[:, 1:] - x[:, :1] + eta
	p = F.softmax(x, dim=1)
	q = p.clone()
	q[:, :1] -= 1 - eps
	q[:, 1:] -= eps / (K - 1)
	return LossCommon(K=K, eps=eps, tau=tau, eta=eta, x=x, z=z, p=p, q=q)

# Mean-squared error loss (Brier loss)
def mse_loss(x):
	M = loss_common(x)
	C = math.sqrt(M.K / (M.K - 1)) * (27 / 8)
	L = C * (1 - M.p[:, :1]).square()
	return L, M

# Negative log likelihood loss
def nll_loss(x):
	M = loss_common(x)
	C = math.sqrt(M.K / (M.K - 1))
	L = -C * torch.log(M.p[:, :1])
	return L, M

# Focal loss
def focal_loss(x):
	M = loss_common(x)
	C = math.sqrt(M.K / (M.K - 1)) * (M.K ** 2) / ((M.K - 1) * (M.K - 1 + 2 * math.log(M.K)))
	L = -C * (1 - M.p[:, :1]).square() * torch.log(M.p[:, :1])  # Note: gamma = 2
	return L, M

# Kullback-Leibler divergence loss
def kldiv_loss(x, eps):
	M = loss_common(x, eps)
	C = math.sqrt((M.K - 1) / M.K) / (1 - eps - 1 / M.K)
	L = C * ((1 - eps) * (math.log(1 - eps) - torch.log(M.p[:, :1])) + (eps / (M.K - 1)) * torch.sum(math.log(eps / (M.K - 1)) - torch.log(M.p[:, 1:]), dim=1, keepdim=True))
	return L, M

# Label-smoothed negative log likelihood loss
def snll_loss(x, eps):
	M = loss_common(x, eps)
	C = math.sqrt((M.K - 1) / M.K) / (1 - eps - 1 / M.K)
	L = -C * ((1 - eps) * torch.log(M.p[:, :1]) + (eps / (M.K - 1)) * torch.sum(torch.log(M.p[:, 1:]), dim=1, keepdim=True))
	return L, M

# Dual negative log likelihood loss
def dnll_loss(x, eps, cap):
	M = loss_common(x, eps)
	C = math.sqrt((M.K - 1) / M.K) / (1 - eps - 1 / M.K)
	pT = M.p[:, :1]
	if cap:
		pT = pT.clamp(max=1 - eps)
	L = -C * ((1 - eps) * torch.log(pT) + eps * torch.log(1 - pT))
	return L, M

# Relative dual negative log likelihood loss (Inf-norm)
def rdnlli_loss(x, eps, cap):
	M = loss_common(x, eps)
	mu = 1 - eps - eps / (M.K - 1)
	C = math.sqrt((M.K - 1) / M.K) / mu
	targetpT = torch.amax(M.p[:, 1:].detach(), dim=1, keepdim=True) + mu
	pT = M.p[:, :1]
	if cap:
		pT = pT.clamp(max=targetpT)
	L = -C * (targetpT * torch.log(pT) + (1 - targetpT) * torch.log(1 - pT))
	return L, M

# Relative dual negative log likelihood loss (2-norm)
def rdnll2_loss(x, eps, cap, cgrad):
	M = loss_common(x, eps)
	mu = 1 - eps - eps / math.sqrt(M.K - 1)
	C = math.sqrt((M.K - 1) / M.K) / ((1 - eps - 1 / M.K) * (1 + 1 / math.sqrt(M.K - 1)))
	pF = M.p[:, 1:]
	if not cgrad:
		pF = pF.detach()
	targetpT = torch.linalg.norm(pF, dim=1, keepdim=True) + mu
	pT = M.p[:, :1]
	if cap:
		pT = pT.clamp(max=targetpT)
	L = -C * (targetpT * torch.log(pT) + (1 - targetpT) * torch.log(1 - pT))
	return L, M

# Max-logit dual negative log likelihood loss
def mdnll_loss(x, eps, cap, cgrad):
	M = loss_common(x, eps)
	C = 0.5 * math.sqrt((M.K - 1) / M.K) / (1 - eps - 1 / M.K)
	pTD = M.p[:, :1]
	if not cgrad:
		pTD = pTD.detach()
	targetpT = 2 * (1 - eps) - pTD
	pT = M.p[:, :1]
	if cap:
		pT = pT.clamp(max=targetpT)
	L = -C * (targetpT * torch.log(pT) + (1 - targetpT) * torch.log(1 - pT))
	return L, M

# Relative raw logit loss
def rrl_loss(x, eps, cap):
	M = loss_common(x, eps)
	z = x - x[:, :1]
	z[:, 1:] += M.eta
	if cap:
		z.clamp_(min=0)
	J = z.square().sum(dim=1, keepdim=True) - torch.square(z.sum(dim=1, keepdim=True)) / M.K
	C = math.sqrt(M.K / (M.K - 1)) / (2 * M.eta)
	L = C * J
	return L, M

# Manually capped relative raw logit loss autograd function
# noinspection PyMethodOverriding, PyAbstractClass
class MRRLCapFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, M):
		ctx.set_materialize_grads(False)
		C = math.sqrt(M.K / (M.K - 1)) / M.eta
		z = x - x[:, :1]
		z[:, 1:] += M.eta
		z.clamp_(min=0)
		grad = C * z
		grad -= grad.sum(dim=1, keepdim=True) / M.K
		L = grad.square().sum(dim=1, keepdim=True)
		ctx.save_for_backward(grad)
		return L

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, gradL):
		if gradL is None or not ctx.needs_input_grad[0]:
			return None
		grad, = ctx.saved_tensors
		return grad, None

# Manually capped relative raw logit loss
def mrrl_cap_loss(x, eps):
	M = loss_common(x, eps)
	L = MRRLCapFunction.apply(x, M)
	return L, M

# Saturated relative raw logit loss
def srrl_loss(x, eps, tau, cap):
	M = loss_common(x, eps, tau=tau)
	delta = ((1 - M.tau) / M.tau) * ((M.K - 1) / M.K) * M.eta * M.eta
	z = x - x[:, :1]
	z[:, 1:] += M.eta
	if cap:
		z.clamp_(min=0)
	J = z.square().sum(dim=1, keepdim=True) - torch.square(z.sum(dim=1, keepdim=True)) / M.K
	C = 1 / math.sqrt(M.tau)
	L = C * torch.sqrt(J + delta)
	return L, M

# Manually capped saturated raw logit loss autograd function
# noinspection PyMethodOverriding, PyAbstractClass
class MSRRLCapFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, M):
		ctx.set_materialize_grads(False)
		delta = ((1 - M.tau) / M.tau) * ((M.K - 1) / M.K) * M.eta * M.eta
		C = 1 / math.sqrt(M.tau)
		z = x - x[:, :1]
		z[:, 1:] += M.eta
		z.clamp_(min=0)
		J = z.square().sum(dim=1, keepdim=True) - torch.square(z.sum(dim=1, keepdim=True)) / M.K
		hJ = C / torch.sqrt(J + delta)
		grad = hJ * z
		grad -= grad.sum(dim=1, keepdim=True) / M.K
		L = grad.square().sum(dim=1, keepdim=True)
		ctx.save_for_backward(grad)
		return L

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, gradL):
		if gradL is None or not ctx.needs_input_grad[0]:
			return None
		grad, = ctx.saved_tensors
		return grad, None

# Manually capped saturated raw logit loss
def msrrl_cap_loss(x, eps, tau):
	M = loss_common(x, eps, tau=tau)
	L = MSRRLCapFunction.apply(x, M)
	return L, M

# Manually exponentially capped saturated raw logit loss autograd function
# noinspection PyMethodOverriding, PyAbstractClass
class MESRRLCapFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, M):
		ctx.set_materialize_grads(False)
		C = (1 - M.eps) / ((M.K - 1) / M.K - M.eps)
		beta = (M.K / (M.K - 1)) / (C * M.eta) ** 2
		z = x - x[:, :1]
		z[:, 1:] += M.eta
		z.clamp_(min=0)
		J = z.square().sum(dim=1, keepdim=True) - torch.square(z.sum(dim=1, keepdim=True)) / M.K
		hJ = C * torch.sqrt(torch.tanh(beta * J) / J)
		grad = hJ * z
		grad -= grad.sum(dim=1, keepdim=True) / M.K
		L = grad.square().sum(dim=1, keepdim=True)
		ctx.save_for_backward(grad)
		return L

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, gradL):
		if gradL is None or not ctx.needs_input_grad[0]:
			return None
		grad, = ctx.saved_tensors
		return grad, None

# Manually exponentially capped saturated raw logit loss
def mesrrl_cap_loss(x, eps):
	M = loss_common(x, eps)
	L = MESRRLCapFunction.apply(x, M)
	return L, M

# Loss map
LOSS_MAP = dict(
	mse=('MSE', lambda x, eps, tau: mse_loss(x)),
	nll=('NLL', lambda x, eps, tau: nll_loss(x)),
	focal=('Focal', lambda x, eps, tau: focal_loss(x)),

	kldiv=('KLDiv', lambda x, eps, tau: kldiv_loss(x, eps)),  # Note: Identical grads to SNLL
	snll=('SNLL', lambda x, eps, tau: snll_loss(x, eps)),
	dnll=('DNLL', lambda x, eps, tau: dnll_loss(x, eps, cap=False)),
	dnllcap=('DNLLCap', lambda x, eps, tau: dnll_loss(x, eps, cap=True)),

	rdnlli=('RDNLLI', lambda x, eps, tau: rdnlli_loss(x, eps, cap=False)),
	rdnllicap=('RDNLLICap', lambda x, eps, tau: rdnlli_loss(x, eps, cap=True)),
	rdnll2=('RDNLL2', lambda x, eps, tau: rdnll2_loss(x, eps, cap=False, cgrad=False)),
	rdnll2cap=('RDNLL2Cap', lambda x, eps, tau: rdnll2_loss(x, eps, cap=True, cgrad=False)),
	rdnll2grad=('RDNLL2Grad', lambda x, eps, tau: rdnll2_loss(x, eps, cap=False, cgrad=True)),  # Note: No loss scaling relative to non-grad, zero grad is not exactly where required
	rdnll2capgrad=('RDNLL2CapGrad', lambda x, eps, tau: rdnll2_loss(x, eps, cap=True, cgrad=True)),  # Note: No loss scaling relative to non-grad, zero grad is not exactly where required, some uncapped return gradients occur

	mdnll=('MDNLL', lambda x, eps, tau: mdnll_loss(x, eps, cap=False, cgrad=False)),  # Note: Identical grads to DNLL
	mdnllcap=('MDNLLCap', lambda x, eps, tau: mdnll_loss(x, eps, cap=True, cgrad=False)),  # Note: Identical grads to DNLLCap
	mdnllgrad=('MDNLLGrad', lambda x, eps, tau: mdnll_loss(x, eps, cap=False, cgrad=True)),  # Note: No loss scaling relative to non-grad, zero grad is not exactly where required
	mdnllcapgrad=('MDNLLCapGrad', lambda x, eps, tau: mdnll_loss(x, eps, cap=True, cgrad=True)),  # Note: No loss scaling relative to non-grad, zero grad is not exactly where required, some uncapped return gradients occur

	rrl=('RRL', lambda x, eps, tau: rrl_loss(x, eps, cap=False)),
	rrlcap=('RRLCap', lambda x, eps, tau: rrl_loss(x, eps, cap=True)),  # Note: Gradients are not continuous
	mrrlcap=('MRRLCap', lambda x, eps, tau: mrrl_cap_loss(x, eps)),

	srrl=('SRRL', lambda x, eps, tau: srrl_loss(x, eps, tau, cap=False)),
	srrlcap=('SRRLCap', lambda x, eps, tau: srrl_loss(x, eps, tau, cap=True)),  # Note: Gradients are not continuous
	msrrlcap=('MSRRLCap', lambda x, eps, tau: msrrl_cap_loss(x, eps, tau)),

	mesrrlcap=('MESRRLCap', lambda x, eps, tau: mesrrl_cap_loss(x, eps)),
)

#
# Evaluate
#

def evalx(x, args):
	x = torch.tensor([x], device=args.device)
	for loss_name, loss in LOSS_MAP.values():
		if loss_name not in args.losses:
			continue
		print(f"EVALUATE: {loss_name}")
		result = eval_loss(x, loss, args)
		print_vec('   x', result.x[0])
		print_vec('   p', result.M.p[0])
		print_vec('   L', result.L[0])
		print_vec('dxdt', result.dxdt[0])
		print_vec('dzdt', result.dzdt[0])
		print_vec('dLdt', result.dLdt[0])
		print()

def eval_loss(x, loss, args):
	x = x.detach().requires_grad_()
	L, M = loss(x, args.eps, args.tau)
	x.grad = None
	L.sum().backward()
	# noinspection PyTypeChecker
	dLdx: torch.Tensor = x.grad
	dxdt = -dLdx
	dzdt = dxdt[:, 1:] - dxdt[:, :1]
	dLdt = -dLdx.square().sum(dim=1, keepdim=True)
	return LossResult(M=M, x=x, L=L, dxdt=dxdt, dzdt=dzdt, dLdt=dLdt)

def print_vec(name, vec, fmt='7.4f'):
	if vec.ndim == 0:
		print(f"{name} = {vec:{fmt}}")
	else:
		print(f"{name} = [{', '.join(f'{item:{fmt}}' for item in vec)}]")

#
# Plot
#

def plot_situation(situation, args):
	sit_name, sit_desc, sit_var, sit_gen = SITUATION_MAP[situation.lower()]
	print(f"SITUATION:   {sit_name}")
	print(f"Description: {sit_desc}")
	v, x = sit_gen(args)
	generate_plots(v, x, sit_var, sit_name, args)
	print()

def generate_plots(v, x, sit_var, sit_name, args):

	M = loss_common(x, args.eps, tau=args.tau)

	fig, axs = plt.subplots(2, 2, figsize=FIGSIZE)
	fig.suptitle(f"{sit_name}: Logits and probabilities vs {sit_var}")
	for i, (data, label) in enumerate(((x[:, :3], 'x'), (M.z[:, :2], 'z = xf - xT + eta'), (M.p[:, :3], 'p'), (M.q[:, :3], 'q = p - ptarget'))):
		ax = axs[np.unravel_index(i, axs.shape)]
		ax.plot(v, data.detach().cpu().numpy())
		ax.grid(visible=True)
		ax.autoscale(axis='x', tight=True)
		ax.legend([label[0] + sub for sub in ('T', 'F', 'K')[-data.shape[1]:]], loc='best')
		ax.set_title(label)
	fig.tight_layout()

	figX, axsX = plt.subplots(1, 3, figsize=FIGSIZE)
	figZ, axsZ = plt.subplots(1, 2, figsize=FIGSIZE)
	figL, axsL = plt.subplots(1, 2, figsize=FIGSIZE)
	figX.suptitle(f"{sit_name}: Logit update rate vs {sit_var}")
	figZ.suptitle(f"{sit_name}: Relative logit update rate vs {sit_var}")
	figL.suptitle(f"{sit_name}: Loss value and rate vs {sit_var}")
	for loss_name, loss in LOSS_MAP.values():
		if loss_name not in args.losses:
			continue
		result = eval_loss(x, loss, args)
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

def gen_equal_false(args):
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.zeros((args.plot_points, args.plot_classes), device=args.device)
	x[:, 0] = v
	return v, x

def gen_two_way(args):
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.full((args.plot_points, args.plot_classes), fill_value=-30.0, device=args.device)
	x[:, 0] = v
	x[:, 1] = 0
	return v, x

def gen_split(args, pT):
	K = args.plot_classes
	v = torch.linspace(-10, 10, args.plot_points)
	x = torch.log((1 / pT - 1) / (K - 2 + torch.exp(v))).repeat(K, 1).T
	x[:, 0] = 0
	x[:, 1] = torch.log((1 / pT - 1) / (1 + (K - 2) * torch.exp(-v)))
	return v, x

def gen_split_high(args):
	return gen_split(args, 1 - args.eps / 2)

def gen_split_equil(args):
	return gen_split(args, 1 - args.eps)

def gen_split_medium(args):
	return gen_split(args, 0.9 * (1 - args.eps))

def gen_split_low(args):
	return gen_split(args, 1 / args.plot_classes)

SITUATION_MAP = dict(
	equal_false=('Equal false', 'All xf are zero and xT varies', 'xT', gen_equal_false),
	two_way=('2-way', 'xT varies for xf1 zero and all other xf significantly negative', 'xT', gen_two_way),
	split_high=('High split', 'pT = 1-eps/2 and the rest is divided amongst pF and equal pK', 'xd = xF - xK', gen_split_high),
	split_equil=('Equilibrium split', 'pT = 1-eps and the rest is divided amongst pF and equal pK', 'xd = xF - xK', gen_split_equil),
	split_med=('Medium split', 'pT = 0.9*(1-eps) and the rest is divided amongst pF and equal pK', 'xd = xF - xK', gen_split_medium),
	split_low=('Low split', 'pT = 1/K and the rest is divided amongst pF and equal pK', 'xd = xF - xK', gen_split_low),
)

#
# Main
#

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--device', type=str, default='cuda', help='Device to perform calculations on')
	parser.add_argument('--eps', type=float, default=DEFAULT_EPS, help='Value of epsilon to use (for all but NLL)')
	parser.add_argument('--tau', type=float, default=AUTO_TAU, help='Value of tau to use (for SRRL, 0 = Auto calculate)')
	parser.add_argument('--losses', type=str, nargs='+', default=list(LOSS_MAP.keys()), help='List of losses to consider')
	parser.add_argument('--evalx', type=float, nargs='+', help='Evaluate case where raw logits are as listed (first is true class)')
	parser.add_argument('--evalp', type=float, nargs='+', help='Evaluate case where probabilities are as listed (first is true class, rescaled to sum to 1)')
	parser.add_argument('--plot', type=str, nargs='+', help='Situation(s) to provide plots for')
	parser.add_argument('--plot_points', type=int, default=401, help='Number of points to use for plotting')
	parser.add_argument('--plot_classes', type=int, default=10, help='Number of classes to use for plotting')

	args = parser.parse_args()
	args.device = torch.device(args.device)
	args.losses = [LOSS_MAP[loss_key.lower()][0] for loss_key in args.losses]

	if args.evalx:
		evalx(args.evalx, args)

	if args.evalp:
		evalx([math.log(item) for item in args.evalp], args)

	if args.plot:
		for situation in args.plot:
			plot_situation(situation, args)

# Run main function
if __name__ == "__main__":
	main()
# EOF
