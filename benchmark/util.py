# Utilities

# Imports
import traceback
import contextlib
import wandb

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
