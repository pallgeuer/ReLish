#!/usr/bin/env python3
# Test the activation functions implementations

# Imports
import argparse
import itertools
import torch
import act_funcs  # noqa
import matplotlib.pyplot as plt

# Plot all activation functions
def plot_act_funcs():
	print("Plotting all activation functions...")
	x = torch.linspace(-6.5, 6.5, 1301)
	for act_func_name in itertools.chain(act_funcs.act_func_factory_map.keys(), act_funcs.act_func_extra):
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
	for act_func_name in itertools.chain(act_funcs.act_func_factory_map.keys(), act_funcs.act_func_extra):
		if act_func_name == 'rrelu':
			print(f"  Gradcheck {act_func_name} (skipped)")
			continue
		print(f"  Gradcheck {act_func_name}")
		act_func_factory = act_funcs.get_act_func_factory(act_func_name)
		try:
			act_func = act_func_factory(inplace=False, dtype=torch.double)
		except TypeError:
			act_func = act_func_factory(inplace=False)
		x = torch.normal(0.0, 3.0, (3, 4, 5, 6), requires_grad=True, dtype=torch.double)
		torch.autograd.gradcheck(act_func, x, raise_exception=True, check_undefined_grad=True, check_batched_grad=True, check_backward_ad=True)

# TODO: Ensure that two activation function implementations are identical (e.g. mish/mish-jit, elu/celu with alpha 1)

# Main
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--plot', action='store_true', help='Plot all activation functions')
	parser.add_argument('--no_gradcheck', dest='gradcheck', action='store_false', help='Check all gradients')
	args = parser.parse_args()
	with torch.inference_mode():
		if args.plot:
			plot_act_funcs()
	if args.gradcheck:
		gradcheck_act_funcs()
# EOF
