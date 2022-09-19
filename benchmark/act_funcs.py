# Activation functions

# Imports
import itertools
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

#
# Activation functions
#

# Note: All JIT-scripted activation functions have an inplace argument, but do not actually support it, because scripted kernel fusions
#       often do not work across in-place operation boundaries. Thus, if any in-place operations are used in the implementation, the
#       performance drops to that of the non-scripted versions (or below).

# Mish: https://arxiv.org/pdf/1908.08681.pdf
class MishJIT(nn.Module):

	__constants__ = ['inplace']
	inplace: bool  # Ignored

	def __init__(self, inplace=False):
		super().__init__()
		self.inplace = inplace

	# noinspection PyMethodMayBeStatic
	def forward(self, x):
		return mish_jit(x)

# noinspection PyUnusedLocal
@torch.jit.script
def mish_jit(x, inplace=False):
	return x.mul(F.softplus(x).tanh())

# E-swish: https://arxiv.org/pdf/1801.07145.pdf
class ESwish(nn.Module):

	__constants__ = ['beta', 'inplace']
	beta: float
	inplace: bool  # Ignored

	def __init__(self, beta=1.25, inplace=False):
		super().__init__()
		self.beta = beta
		self.inplace = inplace

	def forward(self, x):
		return eswish(x, beta=self.beta)

	def extra_repr(self):
		return f"beta={self.beta}"

# noinspection PyUnusedLocal
@torch.jit.script
def eswish(x, beta=1.25, inplace=False):
	return x.mul(x.sigmoid().mul(beta))

# SwishBeta: https://arxiv.org/pdf/1710.05941.pdf
class SwishBeta(nn.Module):

	__constants__ = ['inplace']
	inplace: bool  # Ignored

	def __init__(self, init_beta=1.0, inplace=False, device=None, dtype=None):
		super().__init__()
		self.inplace = inplace
		factory_kwargs = {'device': device, 'dtype': dtype}
		self.beta = nn.Parameter(torch.tensor(float(init_beta), **factory_kwargs))

	def forward(self, x):
		return swish_beta(x, self.beta)

# noinspection PyUnusedLocal
@torch.jit.script
def swish_beta(x, beta, inplace=False):
	return x.mul(x.mul(beta).sigmoid())

#
# Utilities
#

# TODO: Update the sweep files with the new palette of activation functions
# TODO: Test all the custom implemented activation functions for correctness
# TODO: ReLish (alpha, beta, gamma being the slope of the positive linear portion)
# TODO: ReLish=xexpx, ReLish=x/coshx, ReLish=x/(2coshx-1)
# TODO: tanh(x)*log(1+exp(x)), x*log(1 + tanh(exp(x))) (CAREFUL WITH POSSIBLE GRADIENT STABILITY ISSUES)
# TODO: Aria-2, Bent's Identity, SQNL, ELisH, Hard ELisH, SReLU, ISRU, ISRLU, Flatten T-Swish, SineReLU, Weighted Tanh, LeCun's Tanh

# Activation function factory map
# noinspection PyArgumentList
act_func_factory_map = {
	'relu': nn.ReLU,
	'relu6': nn.ReLU6,
	'prelu': lambda inplace=False, **kwargs: nn.PReLU(**kwargs),  # Note: Single learnable parameter is shared between all input channels, ideally do not use weight decay with this
	'rrelu': nn.RReLU,
	'threshold': functools.partial(nn.Threshold, threshold=-1.0, value=-1.0),
	'elu': functools.partial(nn.ELU, alpha=1.0),
	'celu': functools.partial(nn.CELU, alpha=0.5),  # Note: alpha = 1.0 would make CELU equivalent to ELU
	'selu': nn.SELU,
	'gelu-exact': lambda inplace=False, **kwargs: nn.GELU(approximate='none', **kwargs),
	'gelu-approx': lambda inplace=False, **kwargs: nn.GELU(approximate='tanh', **kwargs),
	'silu': nn.SiLU,
	'swish-beta': SwishBeta,
	'hardswish': nn.Hardswish,
	'mish': nn.Mish,
	'mish-jit': MishJIT,
	'sigmoid': lambda inplace=False, **kwargs: nn.Sigmoid(**kwargs),
	'hardsigmoid': nn.Hardsigmoid,
	'logsigmoid': lambda inplace=False, **kwargs: nn.LogSigmoid(**kwargs),
	'softshrink': lambda inplace=False, **kwargs: nn.Softshrink(lambd=0.5, **kwargs),
	'hardshrink': lambda inplace=False, **kwargs: nn.Hardshrink(lambd=0.5, **kwargs),
	'tanh': lambda inplace=False, **kwargs: nn.Tanh(**kwargs),
	'tanhshrink': lambda inplace=False, **kwargs: nn.Tanhshrink(**kwargs),
	'hardtanh': nn.Hardtanh,
	'softsign': lambda inplace=False, **kwargs: nn.Softsign(**kwargs),
	'softplus': lambda inplace=False, **kwargs: nn.Softplus(beta=1, **kwargs),
}
act_func_extra_map = {
	'leakyrelu': ('leakyrelu-0.01', 'leakyrelu-0.05', 'leakyrelu-0.25'),
	'eswish': ('eswish-1.25', 'eswish-1.5', 'eswish-1.75'),
}
act_funcs = tuple(itertools.chain(act_func_factory_map.keys(), itertools.chain.from_iterable(act_func_extra_map.values())))

# Get a factory callable for a given activation function
def get_act_func_factory(name):
	# Returns a callable that accepts an 'inplace' keyword argument (and possibly other keyword arguments as well)
	factory = act_func_factory_map.get(name, None)
	if factory is not None:
		return factory
	elif name.startswith('leakyrelu-'):
		negative_slope = util.parse_value(name[10:], default=0.01, error="Invalid leaky ReLU negative slope specification")  # Common values: 0.01, 0.05, 0.25
		return functools.partial(nn.LeakyReLU, negative_slope=negative_slope)
	elif name.startswith('eswish-'):
		beta = util.parse_value(name[7:], default=1.25, error="Invalid E-swish beta specification")  # Common values: 1.25, 1.50, 1.75
		return functools.partial(ESwish, beta=beta)
	else:
		raise ValueError(f"Invalid activation function specification: {name}")
# EOF
