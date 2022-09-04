# Utilities

# Import
import traceback
import contextlib

# Print exception traceback but propagate exception nonetheless
class ExceptionPrinter(contextlib.AbstractContextManager):

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		traceback.print_exception(exc_type, exc_val, exc_tb)
		return False
# EOF
