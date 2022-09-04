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

# Make a network thinner by a certain factor
def apply_make_thinner(module, actions, factor: fractions.Fraction, skip_inputs, skip_outputs):
	for attr_key in dir(module):
		attr_value = getattr(module, attr_key)
		attr_value_type = type(attr_value)
		if attr_value_type == nn.modules.conv.Conv2d:
			print(f"Found Conv2d: {attr_value}")
		elif attr_value_type == nn.modules.batchnorm.BatchNorm2d:
			print(f"Found BatchNorm2d: {attr_value}")
		elif attr_value_type == nn.modules.linear.Linear:
			print(f"Found Linear: {attr_value}")
	# TODO: If module is in iterable skip_inputs (be None-careful) then don't transform the input planes
	# TODO: If module is in iterable skip_outputs (be None-careful) then don't transform the output planes

# Replace certain activation functions with another activation function type
def apply_replace_act_func(module, actions, act_func_classes, factory, klass):
	for attr_key in dir(module):
		attr_value = getattr(module, attr_key)
		for act_func_class in act_func_classes:
			if isinstance(attr_value, act_func_class):
				if attr_value.__class__ != klass:
					actions.append((replace_submodule, module, attr_key, factory, (), dict(inplace=attr_value.inplace)))
				break

# Replace a submodule with another new one
def replace_submodule(module, attr_key, factory, factory_args, factory_kwargs):
	setattr(module, attr_key, factory(*factory_args, **factory_kwargs))

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
