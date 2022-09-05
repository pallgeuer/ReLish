# Utilities

# Import
import fractions
import traceback
import contextlib
import torch.nn as nn
import wandb

#
# Model util
#

# Execute pending apply actions
def execute_apply_actions(actions):
	for func, *args in actions:
		func(*args)

# Scale the channels of a network by a certain fractional factor
def apply_scale_channels(module, actions, factor: fractions.Fraction, skip_inputs, skip_outputs):
	# noinspection PyProtectedMember
	for attr_key in module._modules.keys():
		submodule = getattr(module, attr_key)
		submodule_class = type(submodule)
		if submodule_class == nn.Conv2d:
			scale_input = submodule not in skip_inputs
			scale_output = submodule not in skip_outputs
			if scale_input or scale_output:
				actions.append(replace_conv2d(module, attr_key, submodule, dict(
					in_channels=apply_factor(submodule.in_channels, factor) if scale_input else submodule.in_channels,
					out_channels=apply_factor(submodule.out_channels, factor) if scale_output else submodule.out_channels,
				)))
		elif submodule_class == nn.BatchNorm2d:
			if submodule not in skip_inputs:
				device, dtype = (submodule.weight.device, submodule.weight.dtype) if submodule.weight is not None else (submodule.running_mean.device, submodule.running_mean.dtype) if submodule.running_mean is not None else (None, None)
				actions.append((replace_submodule, module, attr_key, submodule_class, (), dict(
					num_features=apply_factor(submodule.num_features, factor),
					eps=submodule.eps,
					momentum=submodule.momentum,
					affine=submodule.affine,
					track_running_stats=submodule.track_running_stats,
					device=device,
					dtype=dtype,
				)))
		elif submodule_class == nn.Linear:
			scale_input = submodule not in skip_inputs
			scale_output = submodule not in skip_outputs
			if scale_input or scale_output:
				actions.append((replace_submodule, module, attr_key, submodule_class, (), dict(
					in_features=apply_factor(submodule.in_features, factor) if scale_input else submodule.in_features,
					out_features=apply_factor(submodule.out_features, factor) if scale_output else submodule.out_features,
					bias=submodule.bias is not None,
					device=submodule.weight.device,
					dtype=submodule.weight.dtype,
				)))

# Replace certain activation functions with another activation function type
def apply_replace_act_func(module, actions, act_func_classes, factory, klass):
	for attr_key in dir(module):
		attr_value = getattr(module, attr_key)
		for act_func_class in act_func_classes:
			if isinstance(attr_value, act_func_class):
				if attr_value.__class__ != klass:
					actions.append((replace_submodule, module, attr_key, factory, (), dict(inplace=attr_value.inplace)))
				break

# Replace a nn.Conv2D with a new one
def replace_conv2d(module, attr_key, submodule, replace_kwargs, pending=True):
	factory_kwargs = dict(
		in_channels=submodule.in_channels,
		out_channels=submodule.out_channels,
		kernel_size=submodule.kernel_size,
		stride=submodule.stride,
		padding=submodule.padding,
		dilation=submodule.dilation,
		groups=submodule.groups,
		bias=submodule.bias is not None,
		padding_mode=submodule.padding_mode,
		device=submodule.weight.device,
		dtype=submodule.weight.dtype,
	)
	factory_kwargs.update(replace_kwargs)
	if pending:
		return replace_submodule, module, attr_key, type(submodule), (), factory_kwargs
	else:
		replace_submodule(module, attr_key, type(submodule), (), factory_kwargs)

# Replace a submodule with another new one
def replace_submodule(module, attr_key, factory, factory_args, factory_kwargs):
	setattr(module, attr_key, factory(*factory_args, **factory_kwargs))

# Helper for applying a fractional scale factor to an integer
def apply_factor(value: int, factor: fractions.Fraction):
	scaled_value = value * factor
	if scaled_value.denominator != 1:
		raise ValueError(f"Scaling {value} by {factor} does not result in an integer output")
	return scaled_value.numerator

#
# Wandb util
#

# Print exception traceback but propagate exception nonetheless
class ExceptionPrinter(contextlib.AbstractContextManager):

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		traceback.print_exception(exc_type, exc_val, exc_tb)
		return False

# Print wandb configuration
def print_wandb_config(C=None, newline=True):
	if C is None:
		C = wandb.config
	print("Configuration:")
	# noinspection PyProtectedMember
	for key, value in C._items.items():
		if key == '_wandb':
			if value:
				print("  wandb:")
				for wkey, wvalue in value.items():
					print(f"    {wkey}: {wvalue}")
			else:
				print("  wandb: -")
		else:
			print(f"  {key}: {value}")
	if newline:
		print()
# EOF
