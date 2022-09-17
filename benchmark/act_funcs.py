# Activation functions

# Imports
import functools
import torch
import torch.nn as nn
import util

#
# Activation functions
#

# Note: All JIT-scripted activation functions have an inplace argument, but do not actually support it, because scripted kernel fusions
#       often do not work across in-place operation boundaries. Thus, if any in-place operations are used in the implementation, the
#       performance drops to that of the non-scripted versions (or below).

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
	return x.mul(x.sigmoid()).mul(beta)

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
# TODO: Test all the custom implemented activation functions
# TODO: ReLish (alpha, beta, gamma being the slope of the positive linear portion)
# TODO: ReLish=xexpx, ReLish=x/coshx, ReLish=x/(2coshx-1)
# TODO: Own mish implementation (to be comparable to own implementations of ReLish and other)
# TODO: tanh(x)*log(1+exp(x)), x*log(1 + tanh(exp(x)))
# TODO: Aria-2, Bent's Identity, SQNL, ELisH, Hard ELisH, SReLU, ISRU, ISRLU, Flatten T-Swish, SineReLU, Weighted Tanh, LeCun's Tanh

# Activation function factory map
act_func_factory_map = {
	'relu': nn.ReLU,
	'relu6': nn.ReLU6,
	'prelu': lambda inplace=False: nn.PReLU(),  # Note: Single learnable parameter is shared between all input channels / Ideally do not use weight decay with this
	'rrelu': nn.RReLU,
	'threshold': functools.partial(nn.Threshold, threshold=-1.0, value=-1.0),
	'elu': functools.partial(nn.ELU, alpha=1.0),
	'celu': functools.partial(nn.CELU, alpha=0.5),  # Note: alpha = 1.0 would make CELU equivalent to ELU
	'selu': nn.SELU,
	'gelu-exact': lambda inplace=False: nn.GELU(approximate='none'),
	'gelu-approx': lambda inplace=False: nn.GELU(approximate='tanh'),
	'silu': nn.SiLU,
	'swish-beta': SwishBeta,
	'hardswish': nn.Hardswish,
	'mish': nn.Mish,
	'sigmoid': lambda inplace=False: nn.Sigmoid(),
	'hardsigmoid': nn.Hardsigmoid,
	'logsigmoid': lambda inplace=False: nn.LogSigmoid(),
	'softshrink': lambda inplace=False: nn.Softshrink(lambd=0.5),
	'hardshrink': lambda inplace=False: nn.Hardshrink(lambd=0.5),
	'tanh': lambda inplace=False: nn.Tanh(),
	'tanhshrink': lambda inplace=False: nn.Tanhshrink(),
	'hardtanh': nn.Hardtanh,
	'softsign': lambda inplace=False: nn.Softsign(),
	'softplus': lambda inplace=False: nn.Softplus(beta=1),
}

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
