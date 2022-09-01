#!/usr/bin/env python3
# Train a model on a classification task

# Imports
import contextlib
import os
import sys
import argparse
import wandb

# Main function
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--wandb_project', type=str, default='train_cls', metavar='NAME', help='Wandb project name')
	parser.add_argument('--wandb_entity', type=str, default=None, metavar='USER_TEAM', help='Wandb entity')
	parser.add_argument('--wandb_group', type=str, default=None, metavar='GROUP', help='Wandb group')
	parser.add_argument('--wandb_job_type', type=str, default=None, metavar='TYPE', help='Wandb job type')
	parser.add_argument('--wandb_name', type=str, default=None, metavar='NAME', help='Wandb run name')
	parser.add_argument('--dataset', type=str, choices=('CIFAR10', 'CIFAR100'), default=None, metavar='NAME', help='Classification dataset to train on')
	parser.add_argument('--dataset_path', type=str, default=None, metavar='PATH', help='Classification dataset root path')
	parser.add_argument('--dataset_workers', type=int, default=2, metavar='NUM', help='Number of worker processes to use for dataset loading')
	parser.add_argument('--model', type=str, default='resnet18', metavar='MODEL', help='Classification model')
	parser.add_argument('--act_func', type=str, default='relu', metavar='NAME', help='Activation function')
	parser.add_argument('--optimizer', type=str, default='adam', metavar='NAME', help='Optimizer')
	parser.add_argument('--loss', type=str, default='nllloss', metavar='NAME', help='Loss function')
	parser.add_argument('--epochs', type=int, default=50, metavar='NUM', help='Number of epochs to train')
	parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE', help='Training batch size')
	parser.add_argument('--device', type=str, default='cuda', metavar='DEVICE', help='PyTorch device to run on')
	args = parser.parse_args()

	log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
	with contextlib.suppress(OSError):
		os.mkdir(log_dir)

	print()

	with wandb.init(
		project=args.wandb_project,
		entity=args.wandb_entity,
		group=args.wandb_group,
		job_type=args.wandb_job_type,
		name=args.wandb_name,
		config={key: value for key, value in vars(args).items() if not key.startswith('wandb_')},
		dir=log_dir,
	):

		print()

		print("Configuration:")
		# noinspection PyProtectedMember
		for key, value in wandb.config._items.items():
			if key == '_wandb':
				if value:
					print("  wandb:")
					for wkey, wvalue in value.items():
						print(f"    {wkey}: {wvalue}")
				else:
					print("  wandb: -")
			else:
				print(f"  {key}: {value}")
		print()

		train()

	print()

# Train the model
def train():
	import random
	num_epochs = wandb.config.epochs
	val_loss = 0
	wandb.log(dict(epoch=0))
	for epoch in range(1, num_epochs + 1):
		log = {}
		print(f"Epoch {epoch}/{num_epochs}")
		val_loss += (random.random() - 0.5) / 10
		log.update(epoch=epoch, val_loss=val_loss)
		wandb.log(log)
	print()

# Run main function
if __name__ == "__main__":
	sys.exit(main())
# EOF
