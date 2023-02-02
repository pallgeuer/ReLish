#!/usr/bin/env python3
# Test the activation functions implementations

# Imports
from typing import Union
import argparse
import matplotlib.pyplot as plt
import torch
import act_funcs  # noqa

# Plot activation functions
def plot_act_funcs(names=None, device: Union[str, torch.device] = 'cuda'):
	print("Plotting activation functions...")
	x = torch.linspace(-7, 7, 1401, device=device)
	for act_func_name in names or act_funcs.act_funcs:
		act_func_factory = act_funcs.get_act_func_factory(act_func_name)
		act_func = act_func_factory(inplace=False)
		act_func.to(device=device)
		y = act_func(x)
		fig = plt.figure()
		ax = plt.axes()
		ax.plot(x.cpu(), y.cpu())
		ax.set_xlim(x.min().item(), x.max().item())
		ax.grid(True)
		ax.set_xlabel('Input')
		ax.set_ylabel('Output')
		ax.set_title(act_func_name)
		fig.tight_layout()
		plt.show()

# Warm up a particular activation function (makes a difference for gradients of JIT-implemented activation functions)
def warmup_act_func(name, device: Union[str, torch.device] = 'cuda'):
	act_func_factory = act_funcs.get_act_func_factory(name)
	act_func = act_func_factory(inplace=False)
	act_func.to(device=device)
	x = torch.tensor(0.0, device=device)
	torch.autograd.functional.jacobian(act_func, x, strict=True)

# Perform gradient check on activation functions
def gradcheck_act_funcs(names=None, device: Union[str, torch.device] = 'cuda'):
	print("Performing gradcheck on activation functions...")
	for act_func_name in names or act_funcs.act_funcs:
		if act_func_name == 'rrelu':
			print(f"  Gradcheck {act_func_name} (skipped)")
			continue
		print(f"  Gradcheck {act_func_name}")
		warmup_act_func(act_func_name)
		act_func_factory = act_funcs.get_act_func_factory(act_func_name)
		act_func = act_func_factory(inplace=False)
		act_func.to(device=device, dtype=torch.double)
		x = torch.normal(0.0, 4.0, (8, 3, 8, 8), device=device, dtype=torch.double, requires_grad=True)
		torch.autograd.gradcheck(act_func, x, raise_exception=False, check_undefined_grad=True, check_backward_ad=True)
		torch.autograd.gradcheck(act_func, x, raise_exception=True, check_undefined_grad=True, check_backward_ad=True)

# Compare equivalent implementations of activation functions
def compare_act_funcs(device: Union[str, torch.device] = 'cuda'):
	print("Comparing equivalent activation functions...")
	compare_act_func_pair('relu', 'threshold', kwargs_b=dict(threshold=0.0, value=0.0), device=device)
	compare_act_func_pair('relu', 'leakyrelu-0', device=device)
	compare_act_func_pair('prelu', 'leakyrelu-0.25', device=device)
	compare_act_func_pair('elu', 'celu', kwargs_b=dict(alpha=1.0), device=device)
	compare_act_func_pair('eswish-1', 'swish-beta', device=device)
	compare_act_func_pair('pswish-ff', 'eswish-1.5', device=device)
	compare_act_func_pair('pswish-pp', 'eswish-1.5', device=device)
	compare_act_func_pair('silu', 'eswish-1', device=device)
	compare_act_func_pair('silu', 'swish-beta', device=device)
	compare_act_func_pair('mish', 'mish-jit', device=device, dtol_factor=200)  # Note: mish has significantly better true numerical accuracy
	compare_act_func_pair('relisha', 'relishg1-1-1-1', device=device)
	compare_act_func_pair('relishb', 'relishg2-2-1-1', device=device, dtol_factor=4)  # Note: relishb has slightly better true numerical accuracy
	compare_act_func_pair('relishc', 'relishg2-1-1-1', device=device, dtol_factor=4)
	compare_act_func_pair('relishp1-ppp', 'relishg1-1-1-1', device=device)
	compare_act_func_pair('relishp2-ppp', 'relishg2-1-1-1', device=device, dtol_factor=4)
	compare_act_func_pair('relishp1-ppp', 'relisha', device=device)
	compare_act_func_pair('relishp2-ppp', 'relishc', device=device)

# Compare a pair of activation function implementations that should be equivalent
def compare_act_func_pair(name_a, name_b, kwargs_a=None, kwargs_b=None, device: Union[str, torch.device] = 'cuda', dtol_factor=1.0):
	warmup_act_func(name_a, device=device)
	warmup_act_func(name_b, device=device)
	act_func_a = act_funcs.get_act_func_factory(name_a)(inplace=False, **(kwargs_a or {}))
	act_func_b = act_funcs.get_act_func_factory(name_b)(inplace=False, **(kwargs_b or {}))
	act_func_a.to(device=device)
	act_func_b.to(device=device)
	print(f"  Comparing {act_func_a} <--> {act_func_b}")
	x = torch.linspace(-12.0, 12.0, 2401, device=device)
	ya = act_func_a(x)
	yb = act_func_b(x)
	print(f"    Max error y     = {(ya - yb).abs().max().item():.4g}")
	if not (torch.allclose(ya, yb) and torch.allclose(yb, ya)):
		raise ValueError("Activation functions differ in forward calculation")
	dydxa = torch.autograd.functional.jacobian(act_func_a, x, strict=True)
	dydxb = torch.autograd.functional.jacobian(act_func_b, x, strict=True)
	print(f"    Max error dy/dx = {(dydxa - dydxb).abs().max().item():.4g}")
	if not (torch.allclose(dydxa, dydxb, rtol=1e-5 * dtol_factor, atol=1e-8 * dtol_factor) and torch.allclose(dydxb, dydxa, rtol=1e-5 * dtol_factor, atol=1e-8 * dtol_factor)):
		raise ValueError("Activation functions differ in Jacobian calculation")

# Main function
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device', default='cuda', help='Device to perform calculations on')
	parser.add_argument('--plot', nargs='*', help='Plot all or selected activation functions')
	parser.add_argument('--gradcheck', nargs='*', help='Check all or selected activation function gradients')
	parser.add_argument('--compare', action='store_true', help='Compare equivalent activation function implementations')
	args = parser.parse_args()
	device = torch.device(args.device)
	if args.plot is not None:
		with torch.inference_mode():
			plot_act_funcs(args.plot, device=device)
	if args.gradcheck is not None:
		gradcheck_act_funcs(args.gradcheck, device=device)
	if args.compare:
		compare_act_funcs(device=device)

# Run main function
if __name__ == "__main__":
	main()
# EOF
