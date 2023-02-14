# Utilities

# Imports
import gc
import math
import time
import os.path
import itertools
import traceback
import contextlib
import collections
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
import torch.nn.functional as F

#
# Wandb util
#

# Print exception traceback but propagate exception nonetheless
class ExceptionPrinter(contextlib.AbstractContextManager):

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_type is not None:
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

#
# Training util
#

# Gradient accumulator
# Note: While this accumulates GRADIENTS across multiple batches 'perfectly', it does NOT accumulate the statistics inside e.g. batch norms or similar, if these are present in the model.
#       As such, it is not necessarily absolutely identical to training with a larger batch size, but if the smaller batch size is still "large enough" the difference might not be measurable.
class GradAccum:

	def __init__(self, loader, accum_size):
		self.raw_loader = loader
		self.loader_batch_size = self.raw_loader.batch_size
		self.accum_size = max(accum_size, 1)
		self.drop_last = self.raw_loader.drop_last
		self.num_batches = len(self.raw_loader)
		if self.drop_last:
			self.num_steps = self.num_batches // self.accum_size
			self.last_full_batch_num = self.accum_size * self.num_steps
			self.last_accum_samples = None
			self.num_batches_used = self.last_full_batch_num
			self.num_samples_used = self.num_batches_used * self.loader_batch_size
		else:
			self.num_samples_used = len(self.raw_loader.dataset)
			self.num_steps = (self.num_batches if self.num_samples_used % self.loader_batch_size == 0 else self.num_batches - 1) // self.accum_size
			self.last_full_batch_num = self.accum_size * self.num_steps
			self.last_accum_samples = self.num_samples_used - self.last_full_batch_num * self.loader_batch_size
			self.num_batches_used = self.num_batches
			if self.last_accum_samples > 0:
				self.num_steps += 1
		self.batch_num = 0

	def loader(self):
		self.batch_num = 0
		return itertools.islice(self.raw_loader, self.last_full_batch_num) if self.drop_last else self.raw_loader

	def accum_loss(self, mean_batch_loss, num_in_batch):
		self.batch_num += 1
		if self.batch_num <= self.last_full_batch_num:
			mean_accum_batch_loss = mean_batch_loss / self.accum_size
		else:
			mean_accum_batch_loss = mean_batch_loss * (num_in_batch / self.last_accum_samples)
		optimizer_step = (self.batch_num % self.accum_size == 0 or self.batch_num == self.num_batches)
		return mean_accum_batch_loss, optimizer_step

# Inference statistics
class InferenceStats:

	num_samples: int
	loss: float
	loss_min: float
	loss_sum: float
	topk: list[float, ...]  # Format: [Top-1, Top-2, ...]
	topk_max: list[float, ...]
	topk_sum: list[float, ...]

	def __init__(self, num_topk=5):
		self.num_topk = num_topk
		self.start_epoch()
		self.loss_min = math.inf
		self.topk_max = [-math.inf] * self.num_topk

	def start_epoch(self):
		self.num_samples = 0
		self.loss = math.nan
		self.loss_sum = 0
		self.topk = [math.nan] * self.num_topk
		self.topk_sum = [0] * self.num_topk

	def update(self, num_in_batch, output, target, mean_batch_loss):
		# Note: Make sure to pass output/target as detached CPU tensors, and mean_batch_loss as a float (non-tensor) item()
		self.num_samples += num_in_batch
		self.loss_sum += mean_batch_loss * num_in_batch
		self.loss = self.loss_sum / self.num_samples
		batch_topk_sum = self.calc_topk_sum(output, target)
		for k in range(self.num_topk):
			self.topk_sum[k] += batch_topk_sum[k]
			self.topk[k] = self.topk_sum[k] / self.num_samples

	def stop_epoch(self):
		assert isinstance(self.loss, float)
		if self.loss < self.loss_min or math.isnan(self.loss):
			self.loss_min = self.loss
		for k in range(self.num_topk):
			self.topk_max[k] = max(self.topk_max[k], self.topk[k])

	def calc_topk_sum(self, output, target):
		num_classes = output.shape[1]
		top_indices = output.topk(min(self.num_topk, num_classes), dim=1, largest=True, sorted=True).indices
		topk_tensor = torch.unsqueeze(target, dim=1).eq(top_indices).cumsum(dim=1).sum(dim=0, dtype=float)
		topk_tuple = tuple(topk.item() for topk in topk_tensor)
		return topk_tuple if num_classes >= self.num_topk else topk_tuple + ((topk_tuple[-1],) * (self.num_topk - num_classes))

# Model checkpoint saver
class ModelCheckpointSaver:

	def __init__(self, num_best=1, maximise=False, save_last=False, upload=False, dirname='models'):
		self.num_best = num_best
		self.maximise = maximise
		self.save_last = save_last
		self.upload = upload
		self.dirname = dirname
		self.models_dir = wandb.run.settings.files_dir if self.upload else os.path.join(wandb.run.settings.sync_dir, self.dirname)
		self.models_dir_created = False
		self.last_model_path = None
		self.best_metric = None
		self.best_model_paths = collections.deque(maxlen=self.num_best) if self.num_best >= 1 else None

	def save_model(self, model, epoch, metric, **extra_data):
		is_best = self.best_model_paths is not None and (self.best_metric is None or (not self.maximise and metric <= self.best_metric) or (self.maximise and metric >= self.best_metric))
		if is_best:
			self.best_metric = metric
		if self.save_last or is_best:
			if not self.models_dir_created:
				with contextlib.suppress(FileExistsError):
					os.mkdir(self.models_dir)
				self.models_dir_created = True
			extra_data['epoch'] = epoch
			extra_data['metric'] = metric
			extra_data['model_state_dict'] = model.state_dict()
		if self.save_last:
			last_model_path = os.path.join(self.models_dir, f'{wandb.run.name}-E{epoch:03d}-last.pt')
			torch.save(extra_data, last_model_path)
			if self.last_model_path is not None and self.last_model_path != last_model_path:
				os.remove(self.last_model_path)
			self.last_model_path = last_model_path
		if is_best:
			best_model_path = os.path.join(self.models_dir, f'{wandb.run.name}-E{epoch:03d}-best.pt')
			torch.save(extra_data, best_model_path)
			if len(self.best_model_paths) == self.best_model_paths.maxlen:
				os.remove(self.best_model_paths[0])
			self.best_model_paths.append(best_model_path)

# NaN monitor
class NaNMonitor:

	def __init__(self, max_batches, max_epochs):
		self.nans = 0
		self.batch_nan_worm = EventWorm(event_count=max_batches)
		self.epoch_nan_worm = EventWorm(event_count=max_epochs)

	def update_batch(self, batch_output):
		batch_output_nans = torch.count_nonzero(batch_output.isnan()).item()
		self.batch_nan_worm.update(batch_output_nans > 0)
		self.nans += batch_output_nans

	def update_epoch(self, train_loss, valid_loss):
		self.epoch_nan_worm.update(math.isnan(train_loss) or math.isnan(valid_loss))
		return self.excessive_nans()

	def excessive_nans(self):
		return self.epoch_nan_worm.had_event() or self.batch_nan_worm.had_event()

	def nan_count(self):
		return self.nans

	def batch_worm_count(self):
		return self.batch_nan_worm.count

	def epoch_worm_count(self):
		return self.epoch_nan_worm.count

# Logit distribution statistics
class LogitDistStats:

	def __init__(self, num_samples, enabled=True):
		self.num_samples = num_samples
		self.enabled = enabled
		self.data = torch.zeros(size=(self.num_samples, 4)) if self.enabled else None
		self.data_count = 0

	def start_epoch(self):
		self.data_count = 0

	def update(self, output, target):
		if not self.enabled:
			return
		new_data_count = self.data_count + output.shape[0]
		probs = F.softmax(output, dim=1)
		target = target.unsqueeze(dim=1)
		self.data[self.data_count:new_data_count, 0:1] = output.gather(dim=1, index=target)                             # 0 => xT
		logits_false = output.scatter(dim=1, index=target, value=-math.inf)
		self.data[self.data_count:new_data_count, 1:2], max_false_index = torch.max(logits_false, dim=1, keepdim=True)  # 1 => max(xF)
		self.data[self.data_count:new_data_count, 2:3] = probs.gather(dim=1, index=target)                              # 2 => pT
		self.data[self.data_count:new_data_count, 3:4] = probs.gather(dim=1, index=max_false_index)                     # 3 => max(pF)
		self.data_count = new_data_count

	def stop_epoch(self):
		if not self.enabled:
			return
		assert self.data_count == self.num_samples
		data = self.data.numpy()
		# TODO
		hist, bin_edges = np.histogram(data[:, 0] - data[:, 1], bins='auto', density=True)
		# table = wandb.Table(data=np.stack((0.5*(bin_edges[:-1] + bin_edges[1:]), hist), axis=1), columns=['xT', 'D'])
		# wandb.log({"my_plot_xt": wandb.plot.line(table, 'xT', 'D', title='My custom plot')})
		# TODO: Only plot if new best valid_top1!
		# TODO: plt.ioff() / fig = plt.figure(figsize=(6.4,4.8),dpi=100) / ax = fig.add_subplot() / ax.stairs(hist, bin_edges, fill=True) / wandb.log({"blah": wandb.Image(fig)})
		# TODO: plt.get_fignums() / ax.cla()

		# TODO: Keep one single plot / axis going in the background (with plt.ioff() when doing anything?) that you keep ax.cla() and wandb.Image(fig)
		# TODO: Whenever epoch is a new best (bool input to stop_epoch tells when this is the case?) generate all the required plots and return a dict of them (or wandb.log ("plot_*") them here with commit=False)
		# TODO: Plot distributions of xT-max(xF), pT, max(pF), pT-max(pF)

# Wait around if paused
def wait_if_paused(pause_files, device):
	paused = False
	while any(os.path.exists(pause_file) for pause_file in pause_files):
		if not paused:
			if torch.cuda.is_initialized() and device.type == 'cuda':
				torch.cuda.synchronize(device)
			gc.collect()
			if torch.cuda.is_initialized():
				torch.cuda.ipc_collect()
				torch.cuda.empty_cache()
			print(f"{'*' * 35}  PAUSED  {'*' * 35}")
			paused = True
		time.sleep(3)
	if paused:
		print(f"{'*' * 34}  UNPAUSED  {'*' * 34}")

#
# Misc util
#

# Format a duration as hours minutes seconds (hours can be arbitrarily large)
def format_duration(duration):
	m, s = divmod(int(duration), 60)
	h, m = divmod(m, 60)
	return f'{h:0>2}:{m:0>2}:{s:0>2}'

# Parse a string to the type of a default value
def parse_value(string, default, error):
	if not string:
		return default
	try:
		return type(default)(string)
	except ValueError:
		raise ValueError(f"{error}: {string}")

# Event worm
class EventWorm:

	def __init__(self, event_count):
		self.event_count = max(event_count, 1)
		self.count = 0
		self.event = False

	def reset(self):
		self.count = 0
		self.event = False

	def update(self, state: bool):
		if state:
			self.count += 1
			if self.is_event():
				self.event = True
		elif self.count > 0:
			self.count -= 1

	def is_event(self):
		return self.count >= self.event_count

	def had_event(self):
		return self.event
# EOF
