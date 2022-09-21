#!/usr/bin/env python3
# Test processing wandb results

# Imports
import math
from typing import Dict, Any
import dataclasses
import numpy as np
import wandb

# Run info class
@dataclasses.dataclass(frozen=True)
class RunInfo:
	id: str
	config: Dict[str, Any]
	summary: Dict[str, Any]

# Main function
def main():

	api = wandb.Api()
	runs = api.runs("pallgeuer/cls_cifar10")

	sweep_filter = {None}
	config_filter = dict(
		act_func='original',
		batch_size=64,
		dataset='CIFAR10',
		device='cuda',
		epochs=80,
		loss='nllloss',
		lr_scale=1,
		model='wide2_resnet14_g3',
		no_amp=False,
		no_auto_augment=False,
		no_cudnn_bench=False,
		optimizer='adam',
		scheduler='multisteplr',
	)
	group_by = 'act_func'

	grouped_runs = {}
	for run in runs:
		if run.state != 'finished' or (run.sweep and run.sweep.name) not in sweep_filter or group_by not in run.config:
			continue
		elif any(run.config.get(config_key, None) != config_value for config_key, config_value in config_filter.items() if config_key != group_by):
			continue
		# noinspection PyProtectedMember
		run_info = RunInfo(id=run.id, config=run.config, summary=run.summary._json_dict)
		group = run.config[group_by]
		if group in grouped_runs:
			grouped_runs[group].append(run_info)
		else:
			grouped_runs[group] = [run_info]

	table_rows = {}
	model_width = max(len(group) for group in grouped_runs)
	for group, run_infos in grouped_runs.items():
		scores = sorted(run_info.summary['valid_top1_max'] for run_info in run_infos)
		num_scores = len(scores)
		num_outlier = round(0.1 * num_scores)
		cscores = np.array(scores[num_outlier:(len(scores) - num_outlier)])
		num_cscores = len(cscores)
		mean_cscore = cscores.mean().item()
		std_cscore = cscores.std().item()
		c4 = math.sqrt(2 / (num_cscores - 1)) * math.gamma(num_cscores / 2) / math.gamma((num_cscores - 1) / 2)
		stderr_cscore = std_cscore / (c4 * math.sqrt(num_cscores))
		ci_cscore = (mean_cscore - 1.96 * stderr_cscore, mean_cscore + 1.96 * stderr_cscore)
		table_rows[mean_cscore] = f"{group:<{model_width}s}  {mean_cscore:>6.2%}  {std_cscore:>6.3%}  {stderr_cscore:>6.3%}  {f'{100 * ci_cscore[0]:.2f}-{ci_cscore[1]:.2%}':>12s}  {num_cscores}/{num_scores}"

	print(f"{'Model':<{model_width}s}  {'Mean':>6s}  {'Std':>6s}  {'StdErr':>6s}  {'95% CI':>12s}  N")
	for _, row in sorted(table_rows.items(), reverse=True):
		print(row)

# Run main function
if __name__ == "__main__":
	main()
# EOF
