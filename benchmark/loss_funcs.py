# Loss functions

# Imports
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Classification losses
#

# Generic classification loss
class ClassificationLoss(nn.Module):

	def __init__(self, num_classes, normed, norm_scale, reduction):
		super().__init__()
		self.num_classes = num_classes
		self.normed = normed
		self.norm_scale = norm_scale if self.normed else 1
		self.reduction = reduction

	def extra_repr(self):
		extra_repr = f"classes={self.num_classes}, normed={self.normed}"
		if self.normed:
			extra_repr += f", scale={self.norm_scale:.4g}"
		return extra_repr

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
# TODO: Implement variant where all probabilities are used (mean over K), not just pT (i.e. add pF^2 and divide total by K)
class MSELoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean'):
		norm_scale = math.sqrt(num_classes / (num_classes - 1)) * (27 / 8)
		super().__init__(num_classes, normed, norm_scale, reduction)

	def loss(self, logits, target):
		probs = F.softmax(logits, dim=1)
		probs_true = probs.gather(dim=1, index=target.unsqueeze(dim=1))
		return self.reduce_loss((1 - probs_true).square_())

# Negative log likelihood loss
class NLLLoss(ClassificationLoss):

	def __init__(self, num_classes, normed=True, reduction='mean'):
		norm_scale = math.sqrt(num_classes / (num_classes - 1))
		super().__init__(num_classes, normed, norm_scale, reduction)

	def loss(self, logits, target):
		log_probs = F.log_softmax(logits, dim=1)
		return F.nll_loss(log_probs, target, reduction=self.reduction)
# EOF
