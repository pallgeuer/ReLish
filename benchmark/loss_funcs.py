# Loss functions

# Imports
import sys
import math
import inspect
import warnings
import functools
import itertools
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
DEFAULT_EPS = 0.1

#
# Classification losses
#

# Generic classification loss
class ClassificationLoss(nn.Module):

	def __init__(self, num_classes, normed, norm_scale, reduction, **kwargs):
		super().__init__()
		self.num_classes = num_classes
		self.normed = normed
		self.norm_scale = norm_scale if self.normed else 1
		self.reduction = reduction
		self.kwargs = kwargs
		for key, value in kwargs.items():
			setattr(self, key, value)

	def extra_repr(self):
		extra_repr_parts = [f"classes={self.num_classes}, normed={self.normed}"]
		if self.normed:
			extra_repr_parts.append(f"scale={self.norm_scale:.4g}")
		extra_repr_parts.extend(f"{key}={value:{'.4g' if isinstance(value, float) else ''}}" for key, value in self.kwargs.items())
		return ', '.join(extra_repr_parts)

	def forward(self, logits, target):
		loss = self.loss(logits, target)
		if self.normed:
			loss.mul_(self.norm_scale)
		return loss

	def loss(self, logits, target):
		# Note: Normalisation is already handled in forward(), but take care to implement self.reduction
		raise NotImplementedError

	def reduce_loss(self, loss):
		if self.reduction == 'mean':
			return torch.mean(loss)
		elif self.reduction == 'sum':
			return torch.sum(loss)
		else:
			return loss

# Mean-squared error loss (Brier loss)
class MSELoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean', all_probs=False):
		norm_scale = (27 / 8) * math.sqrt((num_classes - 1) / num_classes) if all_probs else (27 / 8) * math.sqrt(num_classes / (num_classes - 1))
		super().__init__(num_classes, normed, norm_scale, reduction, all_probs=all_probs)

	def loss(self, logits, target):
		probs = F.softmax(logits, dim=1)
		if self.all_probs:
			probs = probs.scatter_add(dim=1, index=target.unsqueeze(dim=1), src=probs.new_full((probs.shape[0], 1, *probs.shape[2:]), -1))
			return self.reduce_loss(probs.square_().sum(dim=1))
		else:
			probs_true = probs.gather(dim=1, index=target.unsqueeze(dim=1))
			return self.reduce_loss(probs_true.sub_(1).square_())

# Negative log likelihood loss
class NLLLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean'):
		norm_scale = math.sqrt(num_classes / (num_classes - 1))
		super().__init__(num_classes, normed, norm_scale, reduction)

	def loss(self, logits, target):
		return F.cross_entropy(logits, target, reduction=self.reduction)

# Focal loss
class FocalLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean'):
		norm_scale = math.sqrt(num_classes / (num_classes - 1)) * (num_classes ** 2) / ((num_classes - 1) * (num_classes - 1 + 2 * math.log(num_classes)))
		super().__init__(num_classes, normed, norm_scale, reduction)

	def loss(self, logits, target):
		probs = F.softmax(logits, dim=1)
		probs_true = probs.gather(dim=1, index=target.unsqueeze(dim=1)).squeeze(dim=1)
		return self.reduce_loss(F.cross_entropy(logits, target, reduction='none').mul_(probs_true.sub_(1).square_()))

# Kullback-Leibler divergence loss
# Note: Has identical grads to SNLL
class KLDivLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean', eps=DEFAULT_EPS):
		norm_scale = math.sqrt((num_classes - 1) / num_classes) / (1 - eps - 1 / num_classes)
		super().__init__(num_classes, normed, norm_scale, reduction, eps=eps)
		self.target_prob_true = math.log(1 - self.eps)
		self.target_prob_false = math.log(self.eps / (self.num_classes - 1))

	def loss(self, logits, target):
		log_probs = F.log_softmax(logits, dim=1)
		target_log_probs = torch.full_like(log_probs, self.target_prob_false).scatter_(dim=1, index=target.unsqueeze(dim=1), value=self.target_prob_true)  # noqa
		if self.reduction == 'mean':
			return F.kl_div(log_probs, target_log_probs, reduction='sum', log_target=True).div_(target.numel())
		elif self.reduction == 'sum':
			return F.kl_div(log_probs, target_log_probs, reduction='sum', log_target=True)
		else:
			return F.kl_div(log_probs, target_log_probs, reduction='none', log_target=True).sum(dim=1)

# Label-smoothed negative log likelihood loss
class SNLLLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean', eps=DEFAULT_EPS):
		norm_scale = math.sqrt((num_classes - 1) / num_classes) / (1 - eps - 1 / num_classes)
		super().__init__(num_classes, normed, norm_scale, reduction, eps=eps)
		self.label_smoothing = eps * (num_classes / (num_classes - 1))

	def loss(self, logits, target):
		return F.cross_entropy(logits, target, reduction=self.reduction, label_smoothing=self.label_smoothing)

# Dual negative log likelihood loss
class DNLLLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean', eps=DEFAULT_EPS, cap=True):
		norm_scale = math.sqrt((num_classes - 1) / num_classes) / (1 - eps - 1 / num_classes)
		super().__init__(num_classes, normed, norm_scale, reduction, eps=eps, cap=cap)
		self.max_log_prob = math.log(1 - eps)
		self.min_log_prob_comp = math.log(eps)

	def loss(self, logits, target):
		log_probs, log_probs_comp = DualLogSoftmaxFunction.apply(logits, 1)
		if self.cap:
			log_probs.clamp_(max=self.max_log_prob)
			log_probs_comp.clamp_(min=self.min_log_prob_comp)
		item_loss = log_probs.add_(log_probs_comp.sub_(log_probs), alpha=self.eps)
		return F.nll_loss(item_loss, target, reduction=self.reduction)

#
# Loss maps
#

# Auto-generate a map of the implemented losses in the format dict[name_lower, tuple(name, loss_factory)]
def generate_loss_map() -> dict[str, tuple[str, Callable]]:
	loss_map: dict[str, tuple[str, Callable]] = {}
	for cls_name, cls_obj in inspect.getmembers(sys.modules['loss_funcs']):
		if inspect.isclass(cls_obj) and cls_obj is not ClassificationLoss and issubclass(cls_obj, ClassificationLoss):
			loss_name = cls_name.removesuffix('Loss')
			loss_params = inspect.signature(cls_obj).parameters.keys()
			extra_params = []
			if 'all_probs' in loss_params:
				extra_params.append(('all_probs', (('', False), ('All', True))))
			if 'cap' in loss_params:
				extra_params.append(('cap', (('', False), ('Cap', True))))
			for extra_values in itertools.product(*(param[1] for param in extra_params)):
				extra_loss_name = loss_name + ''.join(value[0] for value in extra_values)
				extra_param_dict = dict(zip((param[0] for param in extra_params), (value[1] for value in extra_values)))
				loss_map[extra_loss_name.lower()] = (extra_loss_name, functools.partial(cls_obj, **extra_param_dict) if extra_param_dict else cls_obj)
	# noinspection PyTypeChecker
	return dict(sorted(loss_map.items()))
LOSSES = generate_loss_map()

#
# Helper modules
#

# Dual log softmax autograd function
# noinspection PyMethodOverriding, PyAbstractClass
class DualLogSoftmaxFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, inp, dim):
		ctx.set_materialize_grads(False)
		ctx.dim = dim
		stable_inp = inp.sub(inp.amax(dim=dim, keepdim=True))
		temp_exp = stable_inp.exp()
		temp_sumexp = temp_exp.sum(dim=dim, keepdim=True)
		logsumexp = temp_sumexp.log()
		neg_softmax = temp_exp.div_(temp_sumexp.neg_())
		ctx.save_for_backward(neg_softmax)
		return stable_inp.sub_(logsumexp), neg_softmax.log1p()

	@staticmethod
	@torch.autograd.function.once_differentiable
	def backward(ctx, grad_logsoft, grad_logcompsoft):
		if not ctx.needs_input_grad[0]:
			return None, None
		neg_softmax, = ctx.saved_tensors
		if grad_logsoft is not None:
			grad_inp_logsoft = grad_logsoft.sum(dim=ctx.dim, keepdim=True).mul(neg_softmax).add_(grad_logsoft)
		else:
			grad_inp_logsoft = None
		if grad_logcompsoft is not None:
			grad_scaled = grad_logcompsoft.mul(neg_softmax).div_(neg_softmax.add(1))
			grad_inp_logcompsoft = grad_scaled.sum(dim=ctx.dim, keepdim=True).mul(neg_softmax).add_(grad_scaled)
		else:
			grad_inp_logcompsoft = None
		if grad_inp_logsoft is not None and grad_inp_logcompsoft is not None:
			return grad_inp_logsoft.add_(grad_inp_logcompsoft), None
		elif grad_inp_logsoft is not None:
			return grad_inp_logsoft, None
		elif grad_inp_logcompsoft is not None:
			return grad_inp_logcompsoft, None
		else:
			return None, None

# Dual log softmax functional
def dual_log_softmax(inp: torch.Tensor, dim: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
	if dim is None:
		dim = _get_softmax_dim("dual_log_softmax", inp.ndim)
	return DualLogSoftmaxFunction.apply(inp, dim)

# Automatic dimension choice for softmax
def _get_softmax_dim(name: str, ndim: int) -> int:
	warnings.warn(f"Implicit dimension choice for {name} has been deprecated. Change the call to include dim=X as an argument.")
	return 0 if ndim == 0 or ndim == 1 or ndim == 3 else 1
# EOF
