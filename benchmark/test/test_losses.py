#!/usr/bin/env python3
# Test various classification losses and how to best implement them

# Imports
import math
import argparse
import functools
import dataclasses
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#
# Dataclasses
#

@dataclasses.dataclass(frozen=True)
class LossEval:
	loss_name: str
	x: torch.Tensor
	p: torch.Tensor
	L: torch.Tensor
	dxdt: torch.Tensor
	dzdt: torch.Tensor
	dLdt: torch.Tensor

#
# Tests
#

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--device', default='cuda', help='Device to perform calculations on')
	parser.add_argument('--eps', type=float, default=0.2, help='Value of epsilon to use')
	parser.add_argument('--evalx', type=float, nargs='+', help='Evaluate case where raw logits are as listed (first is true class)')
	parser.add_argument('--evalp', type=float, nargs='+', help='Evaluate case where probabilities are as listed (first is true class, rescaled to sum to 1)')
	parser.add_argument('--loss', default='nll', help='Which loss to consider')
	args = parser.parse_args()

	device = torch.device(args.device)

	loss_map = dict(
		nll=('NLL', nll_loss),
		kldiv=('KLDiv', functools.partial(kldiv_loss, eps=args.eps)),
		snll=('SNLL', functools.partial(snll_loss, eps=args.eps)),
		dnll=('DNLL', functools.partial(dnll_loss, eps=args.eps, cap=False)),
		dnllcap=('DNLLCap', functools.partial(dnll_loss, eps=args.eps, cap=True)),
		rrl=('RRL', functools.partial(rrl_loss, eps=args.eps, cap=False)),
		srrl=('SRRL', functools.partial(srrl_loss, eps=args.eps, cap=False)),
	)

	if args.evalp:
		args.evalx = [math.log(item) for item in args.evalp]
	if args.evalx:
		if args.loss == 'all':
			for loss_name in loss_map.keys():
				evalx(x=args.evalx, loss_map=loss_map, loss_name=loss_name, device=device)
		else:
			evalx(x=args.evalx, loss_map=loss_map, loss_name=args.loss, device=device)

def evalx(x, loss_map, loss_name, device):

	loss_name, loss = loss_map[loss_name.lower()]
	print(f"EVALUATE: {loss_name}")

	x = torch.tensor(x, device=device, requires_grad=True)
	L, p = loss(x)

	x.grad = None
	L.backward()
	dLdx = x.grad
	dxdt = -dLdx
	dzdt = dxdt[1:] - dxdt[0]
	dLdt = -dLdx.square().sum()

	print_vec('   x', x)
	print_vec('   p', p)
	print_vec('   L', L)
	print_vec('dxdt', dxdt)
	print_vec('dzdt', dzdt)
	print_vec('dLdt', dLdt)
	print()

	return LossEval(loss_name=loss_name, x=x, p=p, L=L, dxdt=dxdt, dzdt=dzdt, dLdt=dLdt)

#
# Losses
#

# Negative log likelihood loss
def nll_loss(x):
	K = x.numel()
	p = F.softmax(x, dim=0)
	C = math.sqrt(K / (K - 1))
	L = -C * torch.log(p[0])
	return L, p

# Kullback-Leibler divergence loss
def kldiv_loss(x, eps):
	K = x.numel()
	p = F.softmax(x, dim=0)
	C = math.sqrt((K - 1) / K) / (1 - eps - 1 / K)
	L = C * ((1 - eps) * (math.log(1 - eps) - torch.log(p[0])) + (eps / (K - 1)) * torch.sum(math.log(eps / (K - 1)) - torch.log(p[1:])))
	return L, p

# Label-smoothed negative log likelihood loss
def snll_loss(x, eps):
	K = x.numel()
	p = F.softmax(x, dim=0)
	C = math.sqrt((K - 1) / K) / (1 - eps - 1 / K)
	L = -C * ((1 - eps) * torch.log(p[0]) + (eps / (K - 1)) * torch.sum(torch.log(p[1:])))
	return L, p

# Dual negative log likelihood loss
def dnll_loss(x, eps, cap):
	K = x.numel()
	p = F.softmax(x, dim=0)
	C = math.sqrt((K - 1) / K) / (1 - eps - 1 / K)
	pT = torch.clamp(p[0], max=1 - eps) if cap else p[0]
	L = -C * ((1 - eps) * torch.log(pT) + eps * torch.log(1 - pT))
	return L, p

# Relative raw logit loss
def rrl_loss(x, eps, cap):  # TODO: Capping?
	K = x.numel()
	p = F.softmax(x, dim=0)
	eta = math.log(1 - eps) - math.log(eps / (K - 1))
	J = (x[0] - eta).square() + x[1:].square().sum() - torch.square(x[0] - eta + x[1:].sum()) / K
	C = math.sqrt(K / (K-1)) / (2 * eta)
	L = C * J
	return L, p

# Saturated raw logit loss
def srrl_loss(x, eps, cap):
	pass  # TODO: IMPLEMENT

#

#
# Utilities
#

def print_vec(name, vec, fmt='7.4f'):
	if vec.ndim == 0:
		print(f"{name} = {vec:{fmt}}")
	else:
		print(f"{name} = [{', '.join(f'{item:{fmt}}' for item in vec)}]")

# Run main function
if __name__ == "__main__":
	main()
# EOF
