#!/usr/bin/env python3
# Test the activation functions implementations

# Imports
import argparse
import torch
import act_funcs  # noqa
import matplotlib.pyplot as plt

# Plot all activation functions
def plot_act_funcs():
	print("Plotting all activation functions...")
	x = torch.linspace(-6.5, 6.5, 1301)
	for act_func_name in act_funcs.act_funcs:
		act_func_factory = act_funcs.get_act_func_factory(act_func_name)
		act_func = act_func_factory(inplace=False)
		y = act_func(x)
		fig = plt.figure()
		ax = plt.axes()
		ax.plot(x, y)
		ax.set_xlim(x.min(), x.max())
		ax.grid(True)
		ax.set_xlabel('Input')
		ax.set_ylabel('Output')
		ax.set_title(act_func_name)
		fig.tight_layout()
		plt.show()

# Perform gradient check on all activation functions
def gradcheck_act_funcs():
	print("Performing gradcheck on all activation functions...")
	for act_func_name in act_funcs.act_funcs:
		if act_func_name == 'rrelu':
			print(f"  Gradcheck {act_func_name} (skipped)")
			continue
		print(f"  Gradcheck {act_func_name}")
		act_func_factory = act_funcs.get_act_func_factory(act_func_name)
		try:
			act_func = act_func_factory(inplace=False, dtype=torch.double)
		except TypeError:
			act_func = act_func_factory(inplace=False)
		x = torch.normal(0.0, 4.0, (8, 3, 8, 8), requires_grad=True, dtype=torch.double)
		torch.autograd.gradcheck(act_func, x, raise_exception=True, check_undefined_grad=True, check_backward_ad=True)
		torch.autograd.gradcheck(act_func, x, raise_exception=True, check_undefined_grad=True, check_backward_ad=True)  # Second call because sometimes nondeterminism errors appear erratically only for subsequent calls for whatever reason

# Compare equivalent implementations of activation functions
def compare_act_funcs():
	print("Comparing equivalent activation functions...")
	compare_act_func_pair(act_funcs.get_act_func_factory('relu')(), act_funcs.get_act_func_factory('threshold')(threshold=0.0, value=0.0))
	compare_act_func_pair(act_funcs.get_act_func_factory('relu')(), act_funcs.get_act_func_factory('leakyrelu-0')())
	compare_act_func_pair(act_funcs.get_act_func_factory('prelu')(), act_funcs.get_act_func_factory('leakyrelu-0.25')())
	compare_act_func_pair(act_funcs.get_act_func_factory('elu')(), act_funcs.get_act_func_factory('celu')(alpha=1.0))
	compare_act_func_pair(act_funcs.get_act_func_factory('silu')(), act_funcs.get_act_func_factory('eswish-1')())
	compare_act_func_pair(act_funcs.get_act_func_factory('silu')(), act_funcs.get_act_func_factory('swish-beta')())
	compare_act_func_pair(act_funcs.get_act_func_factory('mish')(), act_funcs.get_act_func_factory('mish-jit')())

# Compare a pair of activation function implementations that should be equivalent
def compare_act_func_pair(act_func_a, act_func_b):
	print(f"  Comparing {act_func_a} <--> {act_func_b}")
	x = torch.linspace(-13.0, 13.0, 2601)
	ya = act_func_a(x)
	yb = act_func_b(x)
	print(f"    Max error y     = {(ya - yb).abs().max():.4g}")
	if not (torch.allclose(ya, yb) and torch.allclose(yb, ya)):
		raise ValueError("Activation functions differ in forward calculation")
	dydxa = torch.autograd.functional.jacobian(act_func_a, x, strict=True)
	dydxb = torch.autograd.functional.jacobian(act_func_b, x, strict=True)
	print(f"    Max error dy/dx = {(dydxa - dydxb).abs().max():.4g}")
	if not (torch.allclose(dydxa, dydxb) and torch.allclose(dydxb, dydxa)):
		raise ValueError("Activation functions differ in Jacobian calculation")

# Main
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot', action='store_true', help='Plot all activation functions')
	parser.add_argument('--no_compare', dest='compare', action='store_false', help='Compare equivalent activation function implementations')
	parser.add_argument('--no_gradcheck', dest='gradcheck', action='store_false', help='Check all gradients')
	args = parser.parse_args()
	with torch.inference_mode():
		if args.plot:
			plot_act_funcs()
	if args.compare:
		compare_act_funcs()
	if args.gradcheck:
		gradcheck_act_funcs()
# EOF
