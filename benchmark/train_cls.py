#!/usr/bin/env python3
# Train a model on a classification task

# Imports
import os
import re
import sys
import math
import timeit
import argparse
import functools
import contextlib
import collections
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torchvision.models
import torchvision.datasets
import torchvision.transforms as transforms
import wandb
import models
import act_funcs
import loss_funcs
import util

# Main function
def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--wandb_project', type=str, default='train_cls', metavar='NAME', help='Wandb project name (default: %(default)s)')
	parser.add_argument('--wandb_entity', type=str, default=None, metavar='USER_TEAM', help='Wandb entity')
	parser.add_argument('--wandb_group', type=str, default=None, metavar='GROUP', help='Wandb group')
	parser.add_argument('--wandb_job_type', type=str, default=None, metavar='TYPE', help='Wandb job type')
	parser.add_argument('--wandb_name', type=str, default=None, metavar='NAME', help='Wandb run name')
	parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='NAME', help='Classification dataset to train on (default: %(default)s)')
	parser.add_argument('--dataset_path', type=str, default=None, metavar='PATH', help='Classification dataset root path (default: ENV{DATASET_PATH} or ~/Datasets)')
	parser.add_argument('--dataset_workers', type=int, default=4, metavar='NUM', help='Number of worker processes to use for dataset loading (default: %(default)d)')
	parser.add_argument('--no_auto_augment', action='store_true', help='Disable the AutoAugment input data transform (where present)')
	parser.add_argument('--model', type=str, default='resnet18', metavar='MODEL', help='Classification model (default: %(default)s)')
	parser.add_argument('--model_saves', type=int, default=0, metavar='NUM', help='Number of best model checkpoints to maintain during training (default: %(default)d)')
	parser.add_argument('--model_upload', action='store_true', help='Store saved model checkpoints so that they are uploaded to wandb')
	parser.add_argument('--act_func', type=str, default='original', metavar='NAME', help='Activation function (default: %(default)s)')
	parser.add_argument('--optimizer', type=str, default='adam', metavar='NAME', help='Optimizer (default: %(default)s)')
	parser.add_argument('--wd_scale', type=float, default=1.0, metavar='SCALE', help='Weight decay scale relative to the default value')
	parser.add_argument('--scheduler', type=str, default='multisteplr', metavar='NAME', help='Learning rate scheduler (default: %(default)s)')
	parser.add_argument('--lr_scale', type=float, default=1.0, metavar='SCALE', help='Learning rate scale relative to the default value')
	parser.add_argument('--loss', type=str, default='nllloss', metavar='NAME', help='Loss function (default: %(default)s)')
	parser.add_argument('--epochs', type=int, default=80, metavar='NUM', help='Number of epochs to train (default: %(default)s)')
	parser.add_argument('--warmup_epochs', type=int, default=0, metavar='NUM', help='Number of linear learning rate warmup epochs (default: %(default)s)')
	parser.add_argument('--batch_size', type=int, default=64, metavar='SIZE', help='Training batch size (default: %(default)s)')
	parser.add_argument('--max_batch_size', type=int, default=0, metavar='SIZE', help='Maximum batch size to pass into the model at once (0 = No limit)')
	parser.add_argument('--no_batch_drop', action='store_true', help='Do not drop the last incomplete batch during training (batches are never dropped during validation)')
	parser.add_argument('--device', type=str, default='cuda', metavar='DEVICE', help='PyTorch device to run on (default: %(default)s)')
	parser.add_argument('--no_cudnn_bench', action='store_true', help='Disable cuDNN benchmark mode to save memory over speed')
	parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision training')
	parser.add_argument('--aaa', type=int, default=1, metavar='NUM', help='Dummy variable that allows sweeps to do multiple passes of grid searches')
	parser.add_argument('--max_nan_batches', type=int, default=3000, metavar='NUM', help='Abort training if this many batches have a NaN output (as judged by a worm)')
	parser.add_argument('--max_nan_epochs', type=int, default=4, metavar='NUM', help='Abort training if this many epochs have a NaN training or validation loss (as judged by a worm)')
	parser.add_argument('--log_samples', type=int, default=0, metavar='NUM', help='Log a certain number of samples plus info on every new best epoch')
	parser.add_argument('--logit_stats', action='store_true', help='Calculate and log logit statistics on every new best epoch')
	parser.add_argument('--dry', action='store_true', help='Show what would be done but do not actually run the training')
	parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help='Do not use wandb')
	parser.add_argument('--model_details', action='store_true', help='Whether to show model details')
	parser.add_argument('--dataset_details', action='store_true', help='Whether to calculate and show dataset details')
	args = parser.parse_args()

	if args.dataset_path is None:
		args.dataset_path = os.environ.get('DATASET_PATH') or '~/Datasets'
	args.dataset_path = os.path.expanduser(args.dataset_path)

	log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
	with contextlib.suppress(OSError):
		os.mkdir(log_dir)

	print()

	with contextlib.ExitStack() as stack:

		stack.enter_context(wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity,
			group=args.wandb_group,
			job_type=args.wandb_job_type,
			name=args.wandb_name,
			config={key: value for key, value in vars(args).items() if not key.startswith('wandb_') and key not in ('dry', 'use_wandb', 'model_details', 'dataset_details')},
			dir=log_dir,
			mode='online' if args.use_wandb else 'disabled',
		))
		stack.enter_context(util.ExceptionPrinter())
		if args.use_wandb:
			print()

		C = wandb.config
		util.print_wandb_config(C)
		device = torch.device(C.device)
		torch.backends.cudnn.benchmark = not C.no_cudnn_bench

		pause_dir = wandb.run.dir or '/tmp'
		pause_files = []
		for _ in range(3):
			if len(pause_dir) < 4:
				break
			pause_files.append(os.path.join(pause_dir, 'PAUSE'))
			pause_dir = os.path.dirname(pause_dir)
		if pause_files:
			print("Monitoring for pause files:")
			for pause_file in pause_files:
				print(f"  {pause_file}")
			util.wait_if_paused(pause_files, device)
			print()

		train_loader, valid_loader, tfrm_unnormalize, num_classes, in_shape, accum_size = load_dataset(C, device, details=args.dataset_details)
		model = load_model(C, num_classes, in_shape, device, details=args.model_details)
		criterion = load_criterion(C, num_classes, device, details=args.model_details)
		optimizer = load_optimizer(C, model.parameters())
		scheduler = load_scheduler(C, optimizer)

		if args.dry:
			print("Dry run => Would have trained model...")
		else:
			train_model(C, device, train_loader, valid_loader, tfrm_unnormalize, model, criterion, optimizer, scheduler, accum_size, pause_files)

	print()

# Load the dataset
def load_dataset(C, device, details=False):

	tfrm_normalize_rgb = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
	tfrm_normalize = tfrm_normalize_rgb

	is_inaturalist = C.dataset.startswith('iNaturalist')
	if C.dataset in ('MNIST', 'FashionMNIST'):
		num_classes = 10
		in_shape = (1, 28, 28)
		if C.dataset == 'MNIST':
			train_tfrm = valid_tfrm = transforms.Compose([
				transforms.ToTensor(),
				(tfrm_normalize := transforms.Normalize(mean=(0.1307,), std=(0.3081,))),
			])
		elif C.dataset == 'FashionMNIST':
			valid_tfrm = transforms.Compose([
				transforms.ToTensor(),
				(tfrm_normalize := transforms.Normalize(mean=(0.2860,), std=(0.3530,))),
			])
			train_tfrm = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(size=28, padding=4),
				*valid_tfrm.transforms,
				transforms.RandomErasing(inplace=True),
			])
		else:
			raise AssertionError
		dataset_class = getattr(torchvision.datasets, C.dataset)
		train_dataset = dataset_class(root=C.dataset_path, train=True, transform=train_tfrm)
		valid_dataset = dataset_class(root=C.dataset_path, train=False, transform=valid_tfrm)

	elif C.dataset in ('CIFAR10', 'CIFAR100'):
		num_classes = int(C.dataset[5:])
		in_shape = (3, 32, 32)
		valid_tfrm = transforms.Compose([
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		train_tfrm = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(size=32, padding=4),
			*(() if C.no_auto_augment else (transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),)),
			*valid_tfrm.transforms,
			transforms.RandomErasing(inplace=True),
		])
		dataset_class = getattr(torchvision.datasets, C.dataset)
		folder_path = os.path.join(C.dataset_path, 'CIFAR')
		train_dataset = dataset_class(root=folder_path, train=True, transform=train_tfrm)
		valid_dataset = dataset_class(root=folder_path, train=False, transform=valid_tfrm)

	elif C.dataset == 'TinyImageNet':
		num_classes = 200
		in_shape = (3, 64, 64)
		valid_tfrm = transforms.Compose([
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		train_tfrm = transforms.Compose([
			transforms.RandomHorizontalFlip(),
			transforms.RandomCrop(size=64, padding=8),
			*(() if C.no_auto_augment else (transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),)),
			*valid_tfrm.transforms,
			transforms.RandomErasing(inplace=True),
		])
		folder_path = os.path.join(C.dataset_path, C.dataset, 'tiny-imagenet-200')
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'train'), transform=train_tfrm)
		valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'val'), transform=valid_tfrm)

	elif C.dataset in ('Imagenette', 'Imagewoof', 'Food101', 'ImageNet1K') or is_inaturalist:
		in_shape = (3, 224, 224)
		train_tfrm = transforms.Compose([
			transforms.RandomResizedCrop(size=224),
			transforms.RandomHorizontalFlip(),
			*(() if C.no_auto_augment else (transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),)),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		valid_tfrm = transforms.Compose([
			transforms.Resize(size=256),
			transforms.CenterCrop(size=224),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		if C.dataset == 'Food101':
			num_classes = 101
			folder_path = os.path.join(C.dataset_path, C.dataset)
			train_dataset = torchvision.datasets.Food101(root=folder_path, split='train', transform=train_tfrm)
			valid_dataset = torchvision.datasets.Food101(root=folder_path, split='test', transform=valid_tfrm)
		elif is_inaturalist:  # Format: iNaturalist[2021][M][-{genus|6}]
			match = re.fullmatch(r'iNaturalist(\d{4})?(M)?(-(([a-z]+)|(\d+)))?', C.dataset)
			if not match:
				raise ValueError(f"Failed to parse iNaturalist dataset specification: {C.dataset}")
			version_year = match.group(1) or '2021'
			target_types = (('kingdom', 3), ('phylum', 13), ('class', 51), ('order', 273), ('family', 1103), ('genus', 4884), ('full', 10000))
			target_type = match.group(5) or target_types[int(levelstr) - 1 if (levelstr := match.group(6)) else -1][0]
			num_classes = dict(target_types)[target_type]
			folder_path = os.path.join(C.dataset_path, 'iNaturalist')
			train_dataset = torchvision.datasets.INaturalist(root=folder_path, version=f"{version_year}_train{'_mini' if match.group(2) else ''}", target_type=target_type, transform=train_tfrm)
			valid_dataset = torchvision.datasets.INaturalist(root=folder_path, version=f"{version_year}_valid", target_type=target_type, transform=valid_tfrm)
		else:
			if C.dataset == 'Imagenette':
				num_classes = 10
				folder_path = os.path.join(C.dataset_path, C.dataset, 'imagenette2-320')
			elif C.dataset == 'Imagewoof':
				num_classes = 10
				folder_path = os.path.join(C.dataset_path, C.dataset, 'imagewoof2-320')
			elif C.dataset == 'ImageNet1K':
				num_classes = 1000
				folder_path = os.path.join(C.dataset_path, C.dataset, 'ILSVRC-CLS')
			else:
				raise AssertionError
			train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'train'), transform=train_tfrm)
			valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'val'), transform=valid_tfrm)

	else:
		raise ValueError(f"Invalid dataset specification: {C.dataset}")

	if C.batch_size > C.max_batch_size > 0:
		for accum_size in range(math.ceil(C.batch_size / C.max_batch_size), C.batch_size // 2 + 1):
			if C.batch_size % accum_size == 0:
				model_batch_size = C.batch_size // accum_size
				break
		else:
			model_batch_size = 1
			accum_size = C.batch_size
	else:
		model_batch_size = C.batch_size
		accum_size = 1

	dataset_workers = min(C.dataset_workers, model_batch_size)
	pin_memory = (device.type == 'cuda' and C.log_samples <= 0)  # TODO: Make SampleLogger compatible with memory pinning?!
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_batch_size, num_workers=dataset_workers, shuffle=True, pin_memory=pin_memory, drop_last=not C.no_batch_drop)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=model_batch_size, num_workers=dataset_workers, shuffle=False, pin_memory=pin_memory, drop_last=False)

	tfrm_unnormalize = transforms.Normalize(mean=tuple(-mean / std for mean, std in zip(tfrm_normalize.mean, tfrm_normalize.std)), std=tuple(1 / std for std in tfrm_normalize.std))

	if details:
		print("Validation dataset objects:")
		print(valid_dataset)
		print(valid_loader)
		print()
		show_dataset_details(*calc_dataset_details(valid_loader), name='Validation dataset')
		print("Training dataset objects:")
		print(train_dataset)
		print(train_loader)
		print()
		show_dataset_details(*calc_dataset_details(train_loader), name='Training dataset')

	return train_loader, valid_loader, tfrm_unnormalize, num_classes, in_shape, accum_size

# Calculate details of a dataset
def calc_dataset_details(loader):
	data_shapes = collections.Counter()
	class_counts = collections.Counter()
	for data, target_cpu in loader:
		data_shapes[tuple(data.shape[1:])] += data.shape[0]
		class_counts.update(target_cpu.tolist())
	num_samples = len(loader.dataset)
	if num_samples != sum(class_counts.values()):
		raise AssertionError
	num_classes = len(class_counts)
	if sorted(class_counts.keys()) != list(range(num_classes)):
		raise ValueError("Dataset class indices are not a continuous range 0 -> N-1")
	return num_samples, len(loader), num_classes, class_counts, data_shapes

# Show details of a dataset
def show_dataset_details(num_samples, num_batches, num_classes, class_counts, data_shapes, name='Dataset'):
	print(f"{name} details:")
	print(f"  Num samples = {num_samples}")
	print(f"  Num batches = {num_batches}")
	print(f"  Num classes = {num_classes}")
	print("  Data shapes = {0}".format(', '.join(f'{count} \u00D7 {shape}' for shape, count in data_shapes.most_common())))
	print(f"  Class counts = {tuple(class_counts[c] for c in range(num_classes))}")
	print()

# Load the model
def load_model(C, num_classes, in_shape, device, details=False):

	model_type, _, model_variant = C.model.partition('-')
	parse_model_variant = functools.partial(util.parse_value, string=model_variant, error='Invalid model variant')

	is_fcnet = model_type == 'fcnet'
	is_squeezenetp = model_type == 'squeezenetp'
	is_squeezenet = model_type == 'squeezenet1_1'
	is_resnet = model_type in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2')
	is_wideresnet = model_type in ('wide1_resnet14_g3', 'wide2_resnet14_g3', 'wide4_resnet14_g3', 'wide8_resnet14_g3', 'wide1_resnet20_g3', 'wide2_resnet20_g3', 'wide8_resnet20_g3', 'wide10_resnet26_g3', 'wide2_resnet32_g3', 'wide4_resnet38_g3', 'wide10_resnet38_g3', 'wide1_resnet18_g4', 'wide2_resnet18_g4', 'wide4_resnet18_g4', 'wide8_resnet18_g4', 'wide1_resnet26_g4', 'wide2_resnet26_g4', 'wide8_resnet26_g4', 'wide6_resnet34_g4', 'wide6_resnet42_g4', 'wide4_resnet50_g4')
	is_efficientnet = model_type in ('efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l')
	is_convnext = model_type in ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
	is_vit = model_type in ('vit_b', 'vit_l', 'vit_h')
	is_swin = model_type in ('swin_t', 'swin_s', 'swin_b', 'swin_l')

	if C.act_func == 'original':
		act_func_factory = act_func_class = None
	else:
		act_func_factory = act_funcs.get_act_func_factory(C.act_func)
		act_func_class = act_func_factory().__class__
	in_channels = in_shape[0]

	if is_squeezenet or is_resnet or is_efficientnet or is_convnext or is_swin:
		model_factory = getattr(torchvision.models, model_type, getattr(models, model_type, None))
		if model_factory is None or not model_type.islower() or model_type.startswith('_') or not callable(model_factory):
			raise ValueError(f"Invalid torchvision/models model type: {model_type}")
		model = model_factory(num_classes=num_classes)
	elif is_fcnet:
		model = models.FCNet(in_features=math.prod(in_shape), num_classes=num_classes, num_layers=parse_model_variant(default=3), act_func_factory=act_func_factory)
	elif is_squeezenetp:
		model = torchvision.models.SqueezeNet(version="1_1", num_classes=num_classes)
	elif is_wideresnet:
		match = re.fullmatch(r'wide(\d+)_resnet(\d+)_g(\d+)', model_type)
		model = models.WideResNet(num_classes=num_classes, in_channels=in_channels, width=int(match.group(1)), depth=int(match.group(2)), groups=int(match.group(3)), act_func_factory=act_func_factory)
	elif is_vit:
		if in_shape[1] != in_shape[2]:
			raise ValueError("Vision transformer needs square input image size")
		image_size = in_shape[1]
		downscale = parse_model_variant(default=16)
		vit_props = {'vit_b': (12, 12, 768), 'vit_l': (24, 16, 1024), 'vit_h': (32, 16, 1280)}
		num_layers, num_heads, hidden_dim = vit_props[model_type]
		if 4 < downscale <= 8:
			hidden_dim = (3 * hidden_dim) // 4
		elif downscale <= 4:
			hidden_dim //= 2
		model = torchvision.models.VisionTransformer(image_size=image_size, patch_size=downscale, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, mlp_dim=4 * hidden_dim, num_classes=num_classes)
	else:
		raise ValueError(f"Invalid model type: {model_type}")

	actions = []
	if not is_fcnet and not is_wideresnet:
		if is_squeezenetp:  # Version of squeezenet1_1 with customisable maximum downscale (/1, /2, /4, /8, /16) and with each downscale padded to be exactly a factor of 2
			downscale = parse_model_variant(default=16)
			models.replace_conv2d(model.features, '0', dict(in_channels=in_channels, stride=(1, 1) if downscale < 16 else (2, 2), padding=(1, 1)))
			models.replace_maxpool2d(model.features, '2', dict(padding=1, ceil_mode=False), identity=downscale < 8)
			models.replace_maxpool2d(model.features, '5', dict(padding=1, ceil_mode=False), identity=downscale < 4)
			models.replace_maxpool2d(model.features, '8', dict(padding=1, ceil_mode=False), identity=downscale < 2)
			models.replace_submodule(model.classifier, '2', models.Identity, (), {})  # Note: Remove dying ReLU (a ReLU after the last convolution often leads to permanently zero output after a few epochs, especially if there are lots of classes in the dataset)
		elif is_squeezenet:
			models.replace_conv2d(model.features, '0', dict(in_channels=in_channels))
			models.replace_submodule(model.classifier, '2', models.Identity, (), {})  # Note: Remove dying ReLU (a ReLU after the last convolution often leads to permanently zero output after a few epochs, especially if there are lots of classes in the dataset)
		elif is_resnet:
			downscale = parse_model_variant(default=32)
			models.replace_conv2d(model, 'conv1', dict(in_channels=in_channels, stride=(1, 1) if downscale < 32 else (2, 2)))
			if downscale < 16:
				models.replace_submodule(model, 'maxpool', models.Identity, (), {})
			if downscale < 8:
				destride_func = functools.partial(models.pending_destride, actions=actions)
				model.layer2[0].apply(destride_func)
				if downscale < 4:
					model.layer3[0].apply(destride_func)
				if downscale < 2:
					model.layer4[0].apply(destride_func)
		elif is_efficientnet:
			downscale = parse_model_variant(default=32)
			models.replace_conv2d(model.features[0], '0', dict(in_channels=in_channels, stride=(1, 1) if downscale < 32 else (2, 2)))
			if downscale < 16:
				models.replace_conv2d(model.features[2][0].block[0], '0', dict(stride=(1, 1)))
			if downscale < 8:
				models.replace_conv2d(model.features[3][0].block[0], '0', dict(stride=(1, 1)))
			if downscale < 4:
				models.replace_conv2d(model.features[4][0].block[1], '0', dict(stride=(1, 1)))
			if downscale < 2:
				models.replace_conv2d(model.features[6][0].block[1], '0', dict(stride=(1, 1)))
			if model.features[1][0].stochastic_depth.p == 0.0:
				models.replace_submodule(model.features[1][0], 'stochastic_depth', models.Clone, (), {})  # Note: Solves autograd error when using ReLU (ReLU saves output tensor for backward pass, which is modified in-place by '+=' if stochastic depth has p = 0, which the very first stochastic depth does)
		elif is_convnext:
			downscale = parse_model_variant(default=32)
			models.replace_conv2d(model.features[0], '0', dict(in_channels=in_channels, stride=(stride := (1, 1) if downscale < 16 else (2, 2) if downscale < 32 else (4, 4)), kernel_size=stride))
			if downscale < 8:
				models.replace_conv2d(model.features[2], '1', dict(stride=(1, 1), kernel_size=(1, 1)))
			if downscale < 4:
				models.replace_conv2d(model.features[4], '1', dict(stride=(1, 1), kernel_size=(1, 1)))
			if downscale < 2:
				models.replace_conv2d(model.features[6], '1', dict(stride=(1, 1), kernel_size=(1, 1)))
		elif is_swin:
			downscale = parse_model_variant(default=32)
			models.replace_conv2d(model.features[0], '0', dict(in_channels=in_channels, stride=(stride := (1, 1) if downscale < 16 else (2, 2) if downscale < 32 else (4, 4)), kernel_size=stride))
			def adjust_patch_merging(cond, index):  # noqa
				if cond:
					norm, linear = model.features[index].norm, model.features[index].reduction
					setattr(model.features, str(index), nn.Sequential(
						nn.LayerNorm(normalized_shape=linear.out_features // 2, eps=norm.eps, elementwise_affine=norm.elementwise_affine, device=linear.weight.device, dtype=linear.weight.dtype),
						nn.Linear(in_features=linear.out_features // 2, out_features=linear.out_features, bias=linear.bias is not None, device=linear.weight.device, dtype=linear.weight.dtype),
					))
			adjust_patch_merging(cond=downscale < 8, index=2)
			adjust_patch_merging(cond=downscale < 4, index=4)
			adjust_patch_merging(cond=downscale < 2, index=6)
		elif is_vit:
			models.replace_conv2d(model, 'conv_proj', dict(in_channels=in_channels))
		else:
			raise ValueError(f"Invalid model type: {model_type}")
		if act_func_factory:  # Note: EfficientNet keeps its sigmoid activation scalers
			model.apply(functools.partial(models.pending_replace_act_func, actions=actions, act_func_classes=(nn.ReLU, nn.GELU, nn.SiLU), factory=act_func_factory, klass=act_func_class))

	models.execute_pending_actions(actions)

	if details:
		print(model)
		print()
		print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
		print(f"Trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
		print()

	wandb.watch(model)
	model.to(device=device)

	return model

# Load the criterion
def load_criterion(C, num_classes, device, details=False):

	if C.loss == 'nllloss':
		criterion = nn.CrossEntropyLoss(reduction='mean')
	else:
		_, criterion = loss_funcs.resolve_loss_module(C.loss, num_classes, reduction='mean')
	if not criterion:
		raise ValueError(f"Invalid criterion/loss specification: {C.loss}")

	criterion.to(device=device)

	if details:
		print(f"Criterion: {criterion}")
		print()

	return criterion

# Load the optimizer
def load_optimizer(C, model_params):
	if C.optimizer == 'sgd':
		return torch.optim.SGD(model_params, lr=0.1 * C.lr_scale, momentum=0.9, weight_decay=5e-4 * C.wd_scale)
	elif C.optimizer == 'adam':
		return torch.optim.Adam(model_params, lr=0.001 * C.lr_scale)
	elif C.optimizer == 'adamw':
		return torch.optim.AdamW(model_params, lr=0.001 * C.lr_scale, weight_decay=0.05 * C.wd_scale)
	else:
		raise ValueError(f"Invalid optimizer specification: {C.optimizer}")

# Load the learning rate scheduler
def load_scheduler(C, optimizer):
	if C.scheduler == 'fixedlr':
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=C.epochs + 1, gamma=1.0)
	elif C.scheduler == 'multisteplr':
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.6 * C.epochs), round(0.8 * C.epochs)], gamma=0.1)
	elif C.scheduler == 'cosinedecay':
		return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.epochs)
	else:
		raise ValueError(f"Invalid learning rate scheduler specification: {C.scheduler}")

# Train the model
def train_model(C, device, train_loader, valid_loader, tfrm_unnormalize, model, criterion, optimizer, scheduler, accum_size, pause_files):

	cpu_device = torch.device('cpu')
	amp_enabled = C.amp and device.type == 'cuda'
	scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
	grad_accum = util.GradAccum(train_loader, accum_size=accum_size)
	warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1 / (C.warmup_epochs + 1), end_factor=1, total_iters=C.warmup_epochs) if C.warmup_epochs >= 1 else None
	model_saver = util.ModelCheckpointSaver(num_best=C.model_saves, maximise=True, save_last=False, upload=C.model_upload)
	nan_monitor = util.NaNMonitor(max_batches=C.max_nan_batches, max_epochs=C.max_nan_epochs)
	train_sample_logger = util.SampleLogger(num_samples=grad_accum.num_samples_used, key_prefix='train_', sample_size=C.log_samples, data_tfrm=tfrm_unnormalize)
	valid_sample_logger = util.SampleLogger(num_samples=len(valid_loader.dataset), key_prefix='valid_', sample_size=C.log_samples, data_tfrm=tfrm_unnormalize)
	train_logit_stats = util.LogitDistStats(num_samples=grad_accum.num_samples_used, key_prefix='train_', enabled=C.logit_stats)
	valid_logit_stats = util.LogitDistStats(num_samples=len(valid_loader.dataset), key_prefix='valid_', enabled=C.logit_stats)
	train_stats = util.InferenceStats()
	valid_stats = util.InferenceStats()

	wandb.log(dict(
		hostname=os.uname().nodename,
		gpu=re.sub(r'(nvidia|geforce) ', '', torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device), flags=re.IGNORECASE),
		accum_size=grad_accum.accum_size,
		epoch=0,
		params=sum(p.numel() for p in model.parameters()),
		params_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
		output_nans=nan_monitor.nan_count(),
	))

	init_epoch_stamp = epoch_stamp = timeit.default_timer()
	for epoch in range(1, C.epochs + 1):

		util.wait_if_paused(pause_files, device)

		print('-' * 80)
		lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch}/{C.epochs}, LR {lr:.3g}")
		log = dict(epoch=epoch, lr=lr)

		model.train()
		train_stats.start_epoch()
		train_logit_stats.start_epoch()
		train_sample_logger.start_epoch()
		optimizer.zero_grad(set_to_none=True)
		init_detail_stamp = timeit.default_timer()
		last_detail_stamp = -math.inf

		for data_cpu, target_cpu in grad_accum.loader():

			num_in_batch = data_cpu.shape[0]
			data = data_cpu.to(device, non_blocking=True)
			target = target_cpu.to(device, non_blocking=True)

			with torch.autocast(device_type=device.type, enabled=amp_enabled):
				output = model(data)
				mean_batch_loss = criterion(output, target)
				mean_accum_batch_loss, optimizer_step = grad_accum.accum_loss(mean_batch_loss, num_in_batch)
			scaler.scale(mean_accum_batch_loss).backward()

			if optimizer_step:
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad(set_to_none=True)

			output_detach = output.detach()
			train_logit_stats.update(output_detach, target)
			output_cpu = output_detach.to(device=cpu_device, dtype=float)
			nan_monitor.update_batch(output_cpu)
			train_sample_logger.update(data_cpu, output_cpu, target_cpu)
			train_stats.update(num_in_batch, output_cpu, target_cpu, mean_batch_loss.item())

			detail_stamp = timeit.default_timer()
			if detail_stamp - last_detail_stamp >= 2.0:
				last_detail_stamp = detail_stamp
				print(f"\x1b[2K\r --> [{util.format_duration(detail_stamp - init_detail_stamp)}] Trained {grad_accum.batch_num / grad_accum.num_batches_used:.1%}: Mean loss {train_stats.loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(train_stats.topk))})", end='')

		train_stats.stop_epoch()
		log.update(train_loss=train_stats.loss, min_train_loss=train_stats.loss_min)
		for k, topk in enumerate(train_stats.topk, 1):
			log[f'train_top{k}'] = topk

		print(f"\x1b[2K\rTrained {grad_accum.num_samples_used} samples in {grad_accum.num_steps} steps and {grad_accum.num_batches_used} batches in time {util.format_duration(detail_stamp - init_detail_stamp)}")
		print(f"Training results: Mean loss {train_stats.loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(train_stats.topk))})")

		model.eval()
		valid_stats.start_epoch()
		valid_logit_stats.start_epoch()
		valid_sample_logger.start_epoch()
		num_valid_batches = len(valid_loader)
		init_detail_stamp = timeit.default_timer()
		last_detail_stamp = -math.inf

		with torch.inference_mode():
			for batch_num, (data_cpu, target_cpu) in enumerate(valid_loader, 1):

				num_in_batch = data_cpu.shape[0]
				data = data_cpu.to(device, non_blocking=True)
				target = target_cpu.to(device, non_blocking=True)

				with torch.autocast(device_type=device.type, enabled=amp_enabled):
					output = model(data)
					mean_batch_loss = criterion(output, target)

				output_detach = output.detach()
				valid_logit_stats.update(output_detach, target)
				output_cpu = output_detach.to(device=cpu_device, dtype=float)
				nan_monitor.update_batch(output_cpu)
				valid_sample_logger.update(data_cpu, output_cpu, target_cpu)
				valid_stats.update(num_in_batch, output_cpu, target_cpu, mean_batch_loss.item())

				detail_stamp = timeit.default_timer()
				if detail_stamp - last_detail_stamp >= 2.0:
					last_detail_stamp = detail_stamp
					print(f"\x1b[2K\r --> [{util.format_duration(detail_stamp - init_detail_stamp)}] Validated {batch_num / num_valid_batches:.1%}: Mean loss {valid_stats.loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(valid_stats.topk))})", end='')

		valid_stats.stop_epoch()
		log.update(valid_loss=valid_stats.loss, min_valid_loss=valid_stats.loss_min)
		for k, (topk, topk_max) in enumerate(zip(valid_stats.topk, valid_stats.topk_max), 1):
			log[f'valid_top{k}'] = topk
			log[f'valid_top{k}_max'] = topk_max

		print(f"\x1b[2K\rValidated {valid_stats.num_samples} samples in {num_valid_batches} batches in time {util.format_duration(detail_stamp - init_detail_stamp)}")
		print(f"Validation results: Mean loss {valid_stats.loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(valid_stats.topk))})")

		new_best = model_saver.save_model(model, epoch, metric=valid_stats.topk[0])
		train_logit_stats.stop_epoch(log=new_best)
		valid_logit_stats.stop_epoch(log=new_best)
		train_sample_logger.stop_epoch(log=new_best)
		valid_sample_logger.stop_epoch(log=new_best)

		if warmup_scheduler:
			warmup_scheduler.step()
		scheduler.step()

		end_stamp = timeit.default_timer()
		epoch_time = end_stamp - epoch_stamp
		print(f"Completed epoch in {epoch_time:.1f}s = {util.format_duration(epoch_time)} (total {util.format_duration(end_stamp - init_epoch_stamp)})")
		log.update(epoch_time=epoch_time)
		epoch_stamp = end_stamp

		log.update(output_nans=nan_monitor.nan_count())
		wandb.log(log)

		if nan_monitor.update_epoch(train_stats.loss, valid_stats.loss):
			print(f"Aborting training run due to excessive NaNs ({nan_monitor.epoch_worm_count()} worm epochs, {nan_monitor.batch_worm_count()} worm batches)")
			break

# Run main function
if __name__ == "__main__":
	sys.exit(main())
# EOF
