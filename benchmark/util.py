# Utilities

# Imports
import os.path
import traceback
import contextlib
import collections
import wandb
import torch

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
