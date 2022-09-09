# Models

# Import
import fractions
from typing import Union, Type, Tuple
import torch.nn as nn

#
# Classification models
#

# Fully connected classification network
class FCNet(nn.Module):

	def __init__(self, in_features, num_classes, num_layers, layer_features=384, act_func_factory=None, dropout_prob=0.2):
		super().__init__()
		if act_func_factory is None:
			act_func_factory = nn.ReLU
		self.layers = nn.Sequential(
			self._layer_block(in_features, layer_features, act_func_factory, dropout_prob),
			*(self._layer_block(layer_features, layer_features, act_func_factory, dropout_prob) for _ in range(num_layers - 1)),
			nn.Linear(in_features=layer_features, out_features=num_classes, bias=True),
		)

	@classmethod
	def _layer_block(cls, in_features, out_features, act_func_factory, dropout_prob):
		return nn.Sequential(
			nn.Linear(in_features=in_features, out_features=out_features, bias=False),
			nn.BatchNorm1d(num_features=out_features),
			act_func_factory(inplace=True),
			nn.Dropout1d(p=dropout_prob),
		)

	def forward(self, x):
		return self.layers(x.view(x.shape[0], -1))

# WideResNet classification network (version from original paper)
class WideResNet(nn.Module):

	def __init__(self, num_classes, in_channels=3, depth=28, width=10, dropout_prob=0, act_func_factory=None):
		super().__init__()
		if act_func_factory is None:
			act_func_factory = nn.ReLU
		if (depth - 4) % 6 != 0:
			raise ValueError("Depth must be of the format 6n+4 for integer n")
		num_blocks = (depth - 4) // 6
		widths = tuple(round(v * width) for v in (16, 32, 64))
		self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
		self.group0 = self.create_group(in_channels=16, out_channels=widths[0], num_blocks=num_blocks, stride=1, dropout_prob=dropout_prob, act_func_factory=act_func_factory)
		self.group1 = self.create_group(in_channels=widths[0], out_channels=widths[1], num_blocks=num_blocks, stride=2, dropout_prob=dropout_prob, act_func_factory=act_func_factory)
		self.group2 = self.create_group(in_channels=widths[1], out_channels=widths[2], num_blocks=num_blocks, stride=2, dropout_prob=dropout_prob, act_func_factory=act_func_factory)
		self.bn = nn.BatchNorm2d(num_features=widths[2], affine=True, track_running_stats=True)
		self.act_func = act_func_factory(inplace=True)
		self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
		self.fc = nn.Linear(in_features=widths[2], out_features=num_classes, bias=True)

	def forward(self, x):
		x = self.conv0(x)
		x = self.group0(x)
		x = self.group1(x)
		x = self.group2(x)
		x = self.act_func(self.bn(x))
		x = self.pool(x)
		x = self.fc(x.view(x.shape[0], -1))
		return x

	@classmethod
	def create_group(cls, in_channels, out_channels, num_blocks, stride, dropout_prob, act_func_factory):
		blocks = [cls.Block(in_channels=in_channels, out_channels=out_channels, stride=stride, dropout_prob=dropout_prob, act_func_factory=act_func_factory)]
		blocks.extend(cls.Block(in_channels=out_channels, out_channels=out_channels, stride=1, dropout_prob=dropout_prob, act_func_factory=act_func_factory) for _ in range(1, num_blocks))
		return nn.Sequential(*blocks)

	class Block(nn.Module):

		def __init__(self, in_channels, out_channels, stride, dropout_prob, act_func_factory):
			super().__init__()
			self.bn0 = nn.BatchNorm2d(num_features=in_channels, affine=True, track_running_stats=True)
			self.act_func = act_func_factory(inplace=True)
			self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
			self.dropout = None if dropout_prob <= 0 else nn.Dropout2d(p=dropout_prob, inplace=True)
			self.bn1 = nn.BatchNorm2d(num_features=out_channels, affine=True, track_running_stats=True)
			self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
			self.convdim = None if in_channels == out_channels else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, bias=False)

		def forward(self, x):
			y = self.act_func(self.bn0(x))
			o = self.conv0(y)
			if self.dropout:
				o = self.dropout(o)
			o = self.act_func(self.bn1(o))
			o = self.conv1(o)
			if self.convdim:
				return o + self.convdim(y)
			else:
				return o + x

#
# Modules
#

# Module that simply passes through a tensor
class Identity(nn.Module):

	# noinspection PyMethodMayBeStatic
	def forward(self, x):
		return x

# Module that simply clones a tensor
class Clone(nn.Module):

	# noinspection PyMethodMayBeStatic
	def forward(self, x):
		return x.clone()

#
# Utilities
#

# Execute pending actions
def execute_pending_actions(actions):
	for func, *args in actions:
		func(*args)

# Enqueue pending actions to scale the channels of a network by a certain fractional factor
def pending_scale_channels(module, actions, factor: fractions.Fraction, skip_inputs, skip_outputs):
	# noinspection PyProtectedMember
	for attr_key in module._modules.keys():
		submodule = getattr(module, attr_key)
		submodule_class = type(submodule)
		if submodule_class == nn.Conv2d:
			scale_input = submodule not in skip_inputs
			scale_output = submodule not in skip_outputs
			if scale_input or scale_output:
				replace_conv2d(module, attr_key, submodule, dict(
					in_channels=scale_by_factor(submodule.in_channels, factor) if scale_input else submodule.in_channels,
					out_channels=scale_by_factor(submodule.out_channels, factor) if scale_output else submodule.out_channels,
				), actions=actions)
		elif submodule_class == nn.BatchNorm2d:
			if submodule not in skip_inputs:
				device, dtype = (submodule.weight.device, submodule.weight.dtype) if submodule.weight is not None else (submodule.running_mean.device, submodule.running_mean.dtype) if submodule.running_mean is not None else (None, None)
				actions.append((replace_submodule, module, attr_key, submodule_class, (), dict(
					num_features=scale_by_factor(submodule.num_features, factor),
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
					in_features=scale_by_factor(submodule.in_features, factor) if scale_input else submodule.in_features,
					out_features=scale_by_factor(submodule.out_features, factor) if scale_output else submodule.out_features,
					bias=submodule.bias is not None,
					device=submodule.weight.device,
					dtype=submodule.weight.dtype,
				)))

# Enqueue pending actions to replace certain activation functions with another activation function type
def pending_replace_act_func(module, actions, act_func_classes: Union[Type, Tuple[Type, ...]], factory, klass):
	# noinspection PyProtectedMember
	for attr_key in module._modules.keys():
		submodule = getattr(module, attr_key)
		if isinstance(submodule, act_func_classes):
			if submodule.__class__ != klass:
				actions.append((replace_submodule, module, attr_key, factory, (), dict(inplace=getattr(submodule, 'inplace', False))))

# Replace a nn.Conv2D with a new one (pending action if actions is provided)
def replace_conv2d(module, attr_key, submodule, replace_kwargs, actions=None):
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
	if actions is None:
		replace_submodule(module, attr_key, type(submodule), (), factory_kwargs)
	else:
		actions.append((replace_submodule, module, attr_key, type(submodule), (), factory_kwargs))

# Replace a nn.Conv2D with a new one that has an updated number of input channels (pending action if actions is provided)
def replace_conv2d_in_channels(module, attr_key, in_channels, actions=None):
	submodule = getattr(module, attr_key)
	if submodule.in_channels != in_channels:
		replace_conv2d(module, attr_key, submodule, dict(in_channels=in_channels), actions=actions)

# Replace a submodule with another new one
def replace_submodule(module, attr_key, factory, factory_args, factory_kwargs):
	setattr(module, attr_key, factory(*factory_args, **factory_kwargs))

# Helper for applying a fractional scale factor to an integer
def scale_by_factor(value: int, factor: fractions.Fraction):
	scaled_value = value * factor
	if scaled_value.denominator != 1:
		raise ValueError(f"Scaling {value} by {factor} does not result in an integer output")
	return scaled_value.numerator
# EOF
