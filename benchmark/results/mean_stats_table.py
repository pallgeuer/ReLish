#!/usr/bin/env python3
# Obtain run data from wandb and display grouped mean statistics in a table

# Imports
import math
import argparse
import functools
import distutils.util
import collections.abc
import numpy as np
import wandb

# Main function
def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--entity', type=str, metavar='STR', help='Wandb entity to retrieve runs from')
	parser.add_argument('--project', type=str, metavar='STR', required=True, help='Wandb project to retrieve runs from')
	parser.add_argument('--metric', type=str, metavar='STR', required=True, help='Numeric metric of interest (default namespace: summary_metrics.*)')
	parser.add_argument('--group_by', type=str, metavar='STR', help='Key to group runs by (default namespace: config.*)')

	parser_general = parser.add_argument_group('General filters')
	parser_general.add_argument('--sweep', type=str, metavar='STR', nargs='+', help='Sweep names to filter by (can use \'null\' to select runs that are explicitly not part of a sweep)')
	parser_general.add_argument('--all_tags', type=str, metavar='STR', nargs='+', help='Tags that must all be present')
	parser_general.add_argument('--any_tags', type=str, metavar='STR', nargs='+', help='Tags at least one of which must be present')

	parser_config = parser.add_argument_group('Config filters')
	add_filter = functools.partial(add_argparse_filter, ns='config')
	add_filter(parser_config, 'act_func', str)
	add_filter(parser_config, 'batch_size', int)
	add_filter(parser_config, 'dataset', str)
	add_filter(parser_config, 'device', str)
	add_filter(parser_config, 'epochs', int)
	add_filter(parser_config, 'loss', str)
	add_filter(parser_config, 'lr_scale', float)
	add_filter(parser_config, 'model', str)
	add_filter(parser_config, 'no_amp', bool)
	add_filter(parser_config, 'no_auto_augment', bool)
	add_filter(parser_config, 'no_cudnn_bench', bool)
	add_filter(parser_config, 'optimizer', str)
	add_filter(parser_config, 'scheduler', str)

	parser_summary = parser.add_argument_group('Summary metric filters')
	add_filter = functools.partial(add_argparse_filter, ns='summary_metrics')
	add_filter(parser_summary, 'params', int)
	add_filter(parser_summary, 'hostname', str)
	add_filter(parser_summary, 'gpu', str)

	args = parser.parse_args()

	if hasattr(args, 'config') and hasattr(args.config, 'lr_scale'):
		args.config.lr_scale.extend(tuple(int(lrs) for lrs in args.config.lr_scale if lrs.is_integer()))  # Note: Overcome float vs int bug with wandb filters

	args.metric = get_filter_key(args.metric, 'summary_metrics')
	run_filters = [{'state': 'finished'}, {args.metric: {'$exists': True}}]
	if args.group_by:
		args.group_by = get_filter_key(args.group_by, 'config')
		run_filters.append({args.group_by: {'$exists': True}})
	if args.sweep:
		run_filters.append(create_filter('sweep', args.sweep))
	if args.all_tags:
		run_filters.append(create_filter('tags', args.all_tags, use_and=True))
	if args.any_tags:
		run_filters.append(create_filter('tags', args.any_tags, use_and=False))
	for ns in ('config', 'summary_metrics'):
		subns = getattr(args, ns, None)
		if subns is not None:
			for key, value in vars(subns).items():
				run_filters.append(create_filter(key, value, default_key_ns=ns))

	api = wandb.Api()
	runs = api.runs(path=f'{args.entity}/{args.project}' if args.entity else args.project, filters={"$and": run_filters})

	grouped_metrics = {}
	for run in runs:
		metric = get_run_value(run, args.metric)
		group = get_run_value(run, args.group_by or 'name')
		if group in grouped_metrics:
			grouped_metrics[group].append(metric)
		else:
			grouped_metrics[group] = [metric]
	for metrics in grouped_metrics.values():
		metrics.sort()

	table_rows = {}
	model_width = max((len(group) for group in grouped_metrics), default=8)
	for group, metrics_all in grouped_metrics.items():
		num_metrics_all = len(metrics_all)
		num_trim = round(0.1 * num_metrics_all)
		metrics = np.array(metrics_all[num_trim:(len(metrics_all) - num_trim)])
		num_metrics = len(metrics)
		metric_mean = metrics.mean().item()
		metric_std = metrics.std().item()
		c4 = math.sqrt(2 / (num_metrics - 1)) * math.gamma(num_metrics / 2) / math.gamma((num_metrics - 1) / 2)
		metric_stderr = metric_std / (c4 * math.sqrt(num_metrics))
		metric_ci = (metric_mean - 1.96 * metric_stderr, metric_mean + 1.96 * metric_stderr)
		table_rows[metric_mean] = f"{group:<{model_width}s}  {metric_mean:>6.2%}  {metric_std:>6.3%}  {metric_stderr:>6.3%}  {f'{100 * metric_ci[0]:.2f}-{metric_ci[1]:.2%}':>12s}  {num_metrics}/{num_metrics_all}"

	print(f"{'Model':<{model_width}s}  {'Mean':>6s}  {'Std':>6s}  {'StdErr':>6s}  {'95% CI':>12s}  N")
	for _, row in sorted(table_rows.items(), reverse=True):
		print(row)

# Argparse action that stores values into a sub-namespace
class StoreNSAction(argparse.Action):

	def __init__(self, *args, ns=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ns = ns

	def __call__(self, parser, namespace, values, option_string=None):
		if self.ns is None:
			setattr(namespace, self.dest, values)
		else:
			ns = getattr(namespace, self.ns, None)
			if ns is None:
				ns = argparse.Namespace()
				setattr(namespace, self.ns, ns)
			setattr(ns, self.dest, values)

# Argparse helper for creating namespaced filters
def add_argparse_filter(parser, name, typ, ns):
	if typ is bool:
		parser.add_argument(f'--{name}', type=boolean, metavar='BOOL', action=StoreNSAction, ns=ns, default=argparse.SUPPRESS)
	else:
		parser.add_argument(f'--{name}', type=typ, metavar=typ.__name__, nargs='+', action=StoreNSAction, ns=ns, default=argparse.SUPPRESS)

# Argparse a boolean from string
def boolean(x):
	return bool(distutils.util.strtobool(x))

# Create a run filter expression
def create_filter(key, value, default_key_ns=None, use_and=False):
	key = get_filter_key(key, default_key_ns)
	if isinstance(value, str) or not isinstance(value, collections.abc.Sequence):
		return {key: get_filter_value(value)}
	elif len(value) == 1:
		return {key: get_filter_value(value[0])}
	else:
		return {('$and' if use_and else '$or'): tuple({key: get_filter_value(val)} for val in value)}

# Get a run filter expression key
def get_filter_key(key, default_key_ns):
	if key.startswith('.'):
		return key[1:]
	elif default_key_ns and '.' not in key:
		return f'{default_key_ns}.{key}'
	else:
		return key

# Get a run filter expression value
def get_filter_value(value):
	if isinstance(value, str) and ':' in value:
		cmd, val = value.split(':', maxsplit=2)
		return {f'${cmd}': val}
	else:
		return value if value != 'null' else None

# Get a run value
def get_run_value(run, key):
	parts = key.split('.')
	value = run
	for part in parts:
		if part:
			if isinstance(value, dict):
				value = value[part]
			else:
				value = getattr(value, part)
	if value is run:
		raise KeyError(f"Failed to look up run value: {key}")
	return value

# Run main function
if __name__ == "__main__":
	main()
# EOF
