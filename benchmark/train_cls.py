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
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torchvision.models
import torchvision.datasets
import torchvision.transforms as transforms
import wandb
import models
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
	parser.add_argument('--dataset_workers', type=int, default=2, metavar='NUM', help='Number of worker processes to use for dataset loading (default: %(default)d)')
	parser.add_argument('--model', type=str, default='resnet18', metavar='MODEL', help='Classification model (default: %(default)s)')
	parser.add_argument('--model_details', action='store_true', help='Whether to show model details')
	parser.add_argument('--act_func', type=str, default='original', metavar='NAME', help='Activation function (default: %(default)s)')
	parser.add_argument('--optimizer', type=str, default='adam', metavar='NAME', help='Optimizer (default: %(default)s)')
	parser.add_argument('--scheduler', type=str, default='multisteplr', metavar='NAME', help='Learning rate scheduler (default: %(default)s)')
	parser.add_argument('--lr_scale', type=float, default=1.0, metavar='SCALE', help='Learning rate scale relative to the default value')
	parser.add_argument('--loss', type=str, default='nllloss', metavar='NAME', help='Loss function (default: %(default)s)')
	parser.add_argument('--epochs', type=int, default=80, metavar='NUM', help='Number of epochs to train (default: %(default)s)')
	parser.add_argument('--batch_size', type=int, default=64, metavar='SIZE', help='Training batch size (default: %(default)s)')
	parser.add_argument('--device', type=str, default='cuda', metavar='DEVICE', help='PyTorch device to run on (default: %(default)s)')
	parser.add_argument('--no_cudnn_bench', dest='cudnn_bench', action='store_false', help='Disable cuDNN benchmark mode to save memory over speed')
	parser.add_argument('--no_wandb', dest='use_wandb', action='store_false', help='Do not use wandb')
	parser.add_argument('--dry', action='store_true', help='Show what would be done but do not actually run the training')
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
			config={key: value for key, value in vars(args).items() if not key.startswith('wandb_') and key not in ('dry', 'use_wandb', 'model_details')},
			dir=log_dir,
			mode='online' if args.use_wandb else 'disabled',
		))
		stack.enter_context(util.ExceptionPrinter())
		if args.use_wandb:
			print()

		C = wandb.config
		util.print_wandb_config(C)
		torch.backends.cudnn.benchmark = C.cudnn_bench

		train_loader, valid_loader, num_classes, in_shape = load_dataset(C)
		model = load_model(C, num_classes, in_shape, details=args.model_details)
		output_layer, criterion = load_criterion(C)
		optimizer = load_optimizer(C, model.parameters())
		scheduler = load_scheduler(C, optimizer)

		if args.dry:
			print("Dry run => Would have trained model...")
		else:
			train_model(C, train_loader, valid_loader, model, output_layer, criterion, optimizer, scheduler)

	print()

# Load the dataset
def load_dataset(C):

	tfrm_normalize_rgb = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

	if C.dataset in ('MNIST', 'FashionMNIST'):
		num_classes = 10
		in_shape = (1, 28, 28)
		if C.dataset == 'MNIST':
			tfrm = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
			])
		elif C.dataset == 'FashionMNIST':
			tfrm = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
			])
		else:
			raise AssertionError
		dataset_class = getattr(torchvision.datasets, C.dataset)
		train_dataset = dataset_class(root=C.dataset_path, train=True, transform=tfrm)
		valid_dataset = dataset_class(root=C.dataset_path, train=False, transform=tfrm)

	elif C.dataset in ('CIFAR10', 'CIFAR100'):
		num_classes = int(C.dataset[5:])
		in_shape = (3, 32, 32)
		train_tfrm = transforms.Compose([
			transforms.RandomCrop(size=32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		valid_tfrm = transforms.Compose([
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		dataset_class = getattr(torchvision.datasets, C.dataset)
		folder_path = os.path.join(C.dataset_path, 'CIFAR')
		train_dataset = dataset_class(root=folder_path, train=True, transform=train_tfrm)
		valid_dataset = dataset_class(root=folder_path, train=False, transform=valid_tfrm)

	elif C.dataset == 'TinyImageNet':
		num_classes = 200
		in_shape = (3, 64, 64)
		train_tfrm = transforms.Compose([
			transforms.RandomCrop(size=64, padding=8),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		valid_tfrm = transforms.Compose([
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		folder_path = os.path.join(C.dataset_path, C.dataset, 'tiny-imagenet-200')
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'train'), transform=train_tfrm)
		valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'val'), transform=valid_tfrm)

	elif C.dataset in ('Imagenette', 'Imagewoof', 'ImageNet1K'):
		if C.dataset == 'Imagenette':
			num_classes = 10
			folder_path = os.path.join(C.dataset_path, C.dataset, 'imagenette2-320')
		elif C.dataset == 'Imagewoof':
			num_classes = 10
			folder_path = os.path.join(C.dataset_path, C.dataset, 'imagewoof2-320')
		elif C.dataset == 'ImageNet1K':
			num_classes = 1000
			folder_path = os.path.join(C.dataset_path, C.dataset, 'ILSVRC2012')
		else:
			raise AssertionError
		in_shape = (3, 224, 224)
		train_tfrm = transforms.Compose([
			transforms.RandomResizedCrop(size=224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		valid_tfrm = transforms.Compose([
			transforms.Resize(size=256),
			transforms.CenterCrop(size=224),
			transforms.ToTensor(),
			tfrm_normalize_rgb,
		])
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'train'), transform=train_tfrm)
		valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(folder_path, 'val'), transform=valid_tfrm)

	else:
		raise ValueError(f"Invalid dataset specification: {C.dataset}")

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=C.batch_size, num_workers=C.dataset_workers, shuffle=True, pin_memory=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=C.batch_size, num_workers=C.dataset_workers, shuffle=False, pin_memory=True)

	return train_loader, valid_loader, num_classes, in_shape

# Load the model
def load_model(C, num_classes, in_shape, details=False):

	model_type, _, model_variant = C.model.partition('-')

	def parse_model_variant(default):
		if not model_variant:
			return default
		try:
			return type(default)(model_variant)
		except ValueError:
			raise ValueError(f"Invalid model variant: {model_variant}")

	is_fcnet = model_type == 'fcnet'
	is_squeezenetp = model_type == 'squeezenetp'
	is_squeezenet = model_type == 'squeezenet1_1'
	is_resnet = model_type in ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_64x4d', 'wide_resnet50_2', 'wide_resnet101_2')
	is_wideresnet = model_type in ('wide1_resnet14_g3', 'wide2_resnet14_g3', 'wide4_resnet14_g3', 'wide8_resnet14_g3', 'wide1_resnet20_g3', 'wide2_resnet20_g3', 'wide8_resnet20_g3', 'wide10_resnet26_g3', 'wide2_resnet32_g3', 'wide4_resnet38_g3', 'wide10_resnet38_g3', 'wide1_resnet18_g4', 'wide2_resnet18_g4', 'wide4_resnet18_g4', 'wide8_resnet18_g4', 'wide1_resnet26_g4', 'wide2_resnet26_g4', 'wide8_resnet26_g4', 'wide6_resnet34_g4', 'wide6_resnet42_g4', 'wide4_resnet50_g4')
	is_efficientnet = model_type in ('efficientnet_v2_s', 'efficientnet_v2_m', 'efficientnet_v2_l')
	is_convnext = model_type in ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large')
	is_vit = model_type in ('vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14')
	is_swin = model_type in ('swin_t', 'swin_s', 'swin_b', 'swin_l')

	if C.act_func == 'original':
		act_func_factory = act_func_class = None
	else:
		act_func_factory = get_act_func_factory(C)
		act_func_class = act_func_factory().__class__
	in_channels = in_shape[0]

	if is_squeezenet or is_resnet or is_efficientnet or is_convnext or is_vit or is_swin:
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
		elif is_efficientnet or is_convnext or is_swin:
			models.replace_conv2d(model.features[0], '0', dict(in_channels=in_channels))
			if is_efficientnet and model.features[1][0].stochastic_depth.p == 0.0:
				models.replace_submodule(model.features[1][0], 'stochastic_depth', models.Clone, (), {})  # Note: Solves autograd error when using ReLU (ReLU saves output tensor for backward pass, which is modified in-place by '+=' if stochastic depth has p = 0, which the very first stochastic depth does)
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
	model.to(device=C.device)

	return model

# Get the required activation function class
def get_act_func_factory(C):  # TODO: Refactor this into another file along with the implementations of the other activation functions
	# Returns a callable that accepts an 'inplace' keyword argument
	# TODO: ReLish (alpha, beta, gamma being the slope of the positive linear portion)
	# TODO: ReLish=xexpx, ReLish=x/coshx, ReLish=x/(2coshx-1)
	# TODO: Own mish implementation (to be comparable to own implementations of ReLish and other)
	# TODO: swish-beta, eswish-1.25, eswish-1.5, eswish-1.75
	# TODO: tanh(x)*log(1+exp(x)), x*log(1 + tanh(exp(x)))
	# TODO: Aria-2, Bent's Identity, SQNL, ELisH, Hard ELisH, SReLU, ISRU, ISRLU, Flatten T-Swish, SineReLU, Weighted Tanh, LeCun's Tanh
	if C.act_func == 'elu':
		return functools.partial(nn.ELU, alpha=1.0)
	elif C.act_func == 'hardshrink':
		return lambda lambd=0.5, inplace=False: nn.Hardshrink(lambd=lambd)
	elif C.act_func == 'hardsigmoid':
		return nn.Hardsigmoid
	elif C.act_func == 'hardtanh':
		return nn.Hardtanh
	elif C.act_func == 'hardswish':
		return nn.Hardswish
	elif C.act_func == 'leakyrelu-0.01':
		return functools.partial(nn.LeakyReLU, negative_slope=0.01)
	elif C.act_func == 'leakyrelu-0.05':
		return functools.partial(nn.LeakyReLU, negative_slope=0.05)
	elif C.act_func == 'leakyrelu-0.25':
		return functools.partial(nn.LeakyReLU, negative_slope=0.25)
	elif C.act_func == 'logsigmoid':
		return lambda inplace=False: nn.LogSigmoid()
	elif C.act_func == 'prelu':
		return lambda inplace=False: nn.PReLU()  # Note: Single learnable parameter is shared between all input channels / Ideally do not use weight decay with this
	elif C.act_func == 'relu':
		return nn.ReLU
	elif C.act_func == 'relu6':
		return nn.ReLU6
	elif C.act_func == 'rrelu':
		return nn.RReLU
	elif C.act_func == 'selu':
		return nn.SELU
	elif C.act_func == 'celu':
		return functools.partial(nn.CELU, alpha=0.5)  # Note: alpha = 1.0 would make CELU equivalent to ELU
	elif C.act_func == 'gelu-exact':
		return lambda approximate='none', inplace=False: nn.GELU(approximate=approximate)
	elif C.act_func == 'gelu-approx':
		return lambda approximate='tanh', inplace=False: nn.GELU(approximate=approximate)
	elif C.act_func == 'sigmoid':
		return lambda inplace=False: nn.Sigmoid()
	elif C.act_func in ('silu', 'swish-1'):
		return nn.SiLU
	elif C.act_func == 'swish-beta':
		raise NotImplementedError  # TODO: x * sigmoid(beta * x) for trainable parameter beta
	elif C.act_func == 'eswish-1.25':
		raise NotImplementedError  # TODO: 1.25 * x * sigmoid(x)
	elif C.act_func == 'eswish-1.5':
		raise NotImplementedError  # TODO: 1.5 * x * sigmoid(x)
	elif C.act_func == 'eswish-1.75':
		raise NotImplementedError  # TODO: 1.75 * x * sigmoid(x)
	elif C.act_func == 'mish':
		return nn.Mish
	elif C.act_func == 'softplus':
		return lambda beta=1.0, inplace=False: nn.Softplus(beta=beta)
	elif C.act_func == 'softshrink':
		return lambda lambd=0.5, inplace=False: nn.Softshrink(lambd=lambd)
	elif C.act_func == 'softsign':
		return lambda inplace=False: nn.Softsign()
	elif C.act_func == 'tanh':
		return lambda inplace=False: nn.Tanh()
	elif C.act_func == 'tanhshrink':
		return lambda inplace=False: nn.Tanhshrink()
	elif C.act_func == 'threshold':
		return functools.partial(nn.Threshold, threshold=-1.0, value=-1.0)
	else:
		raise ValueError(f"Invalid activation function specification: {C.act_func}")

# Load the criterion
def load_criterion(C):

	if C.loss == 'nllloss':
		output_layer = nn.LogSoftmax(dim=1)
		criterion = nn.NLLLoss(reduction='mean')
	else:
		raise ValueError(f"Invalid criterion/loss specification: {C.loss}")

	if output_layer is not None:
		output_layer.to(device=C.device)
	criterion.to(device=C.device)

	return output_layer, criterion

# Load the optimizer
def load_optimizer(C, model_params):
	if C.optimizer == 'sgd':
		return torch.optim.SGD(model_params, lr=0.1 * C.lr_scale, momentum=0.9, weight_decay=5e-4)
	elif C.optimizer == 'adam':
		return torch.optim.Adam(model_params, lr=0.001 * C.lr_scale)
	else:
		raise ValueError(f"Invalid optimizer specification: {C.optimizer}")

# Load the learning rate scheduler
def load_scheduler(C, optimizer):
	if C.scheduler == 'fixedlr':
		return torch.optim.lr_scheduler.StepLR(optimizer, step_size=C.epochs + 1, gamma=1.0)
	elif C.scheduler == 'multisteplr':
		return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.6 * C.epochs), round(0.8 * C.epochs)], gamma=0.1)
	else:
		raise ValueError(f"Invalid learning rate scheduler specification: {C.scheduler}")

# Train the model
def train_model(C, train_loader, valid_loader, model, output_layer, criterion, optimizer, scheduler):

	valid_topk_max = [0] * 5
	device = torch.device(C.device)

	wandb.log(dict(
		epoch=0,
		params=sum(p.numel() for p in model.parameters()),
		params_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
	))

	init_epoch_stamp = epoch_stamp = timeit.default_timer()
	for epoch in range(1, C.epochs + 1):

		print('-' * 80)
		lr = optimizer.param_groups[0]['lr']
		print(f"Epoch {epoch}/{C.epochs}, LR {lr:.3g}")
		log = dict(epoch=epoch, lr=lr)

		model.train()

		num_train_batches = len(train_loader)
		num_train_samples = 0
		train_loss = 0
		train_topk = [0] * 5
		init_detail_stamp = last_detail_stamp = timeit.default_timer()

		for batch_num, (data, target) in enumerate(train_loader):

			num_in_batch = data.shape[0]
			data = data.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True)

			optimizer.zero_grad()
			output = model(data)
			assert output.shape[0] == num_in_batch
			mean_batch_loss = criterion(output if output_layer is None else output_layer(output), target)
			mean_batch_loss_float = mean_batch_loss.item()
			mean_batch_loss.backward()
			optimizer.step()

			num_train_samples += num_in_batch
			train_loss += mean_batch_loss_float * num_in_batch
			batch_topk_sum = calc_topk_sum(output, target, topn=5)
			for k in range(5):
				train_topk[k] += batch_topk_sum[k]

			detail_stamp = timeit.default_timer()
			if detail_stamp - last_detail_stamp >= 2.0:
				last_detail_stamp = detail_stamp
				print(f"\x1b[2K\r --> [{util.format_duration(detail_stamp - init_detail_stamp)}] Trained {(batch_num + 1) / num_train_batches:.1%}: Mean loss {train_loss / num_train_samples:#.4g}, Top-k ({', '.join(f'{topk / num_train_samples:.2%}' for topk in reversed(train_topk))})", end='')

		train_loss /= num_train_samples
		for k in range(5):
			train_topk[k] /= num_train_samples

		log.update(train_loss=train_loss)
		for k in range(5):
			log[f'train_top{k + 1}'] = train_topk[k]

		print(f"\x1b[2K\rTrained {num_train_samples} samples in {num_train_batches} batches in time {util.format_duration(detail_stamp - init_detail_stamp)}")
		print(f"Training results: Mean loss {train_loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(train_topk))})")

		model.eval()

		num_valid_batches = len(valid_loader)
		num_valid_samples = 0
		valid_loss = 0
		valid_topk = [0] * 5
		init_detail_stamp = last_detail_stamp = timeit.default_timer()

		with torch.inference_mode():
			for batch_num, (data, target) in enumerate(valid_loader):

				num_in_batch = data.shape[0]
				data = data.to(device, non_blocking=True)
				target = target.to(device, non_blocking=True)

				output = model(data)
				assert output.shape[0] == num_in_batch
				mean_batch_loss = criterion(output if output_layer is None else output_layer(output), target)
				mean_batch_loss_float = mean_batch_loss.item()

				num_valid_samples += num_in_batch
				valid_loss += mean_batch_loss_float * num_in_batch
				batch_topk_sum = calc_topk_sum(output, target, topn=5)
				for k in range(5):
					valid_topk[k] += batch_topk_sum[k]

				detail_stamp = timeit.default_timer()
				if detail_stamp - last_detail_stamp >= 2.0:
					last_detail_stamp = detail_stamp
					print(f"\x1b[2K\r --> [{util.format_duration(detail_stamp - init_detail_stamp)}] Validated {(batch_num + 1) / num_valid_batches:.1%}: Mean loss {valid_loss / num_valid_samples:#.4g}, Top-k ({', '.join(f'{topk / num_valid_samples:.2%}' for topk in reversed(valid_topk))})", end='')

		valid_loss /= num_valid_samples
		for k in range(5):
			valid_topk[k] /= num_valid_samples
			valid_topk_max[k] = max(valid_topk_max[k], valid_topk[k])

		log.update(valid_loss=valid_loss)
		for k in range(5):
			log[f'valid_top{k + 1}'] = valid_topk[k]
			log[f'valid_top{k + 1}_max'] = valid_topk_max[k]

		print(f"\x1b[2K\rValidated {num_valid_samples} samples in {num_valid_batches} batches in time {util.format_duration(detail_stamp - init_detail_stamp)}")
		print(f"Validation results: Mean loss {valid_loss:#.4g}, Top-k ({', '.join(f'{topk:.2%}' for topk in reversed(valid_topk))})")

		scheduler.step()

		end_stamp = timeit.default_timer()
		epoch_time = end_stamp - epoch_stamp
		print(f"Completed epoch in {epoch_time:.1f}s = {util.format_duration(epoch_time)} (total {util.format_duration(end_stamp - init_epoch_stamp)})")
		log.update(epoch_time=epoch_time)
		epoch_stamp = end_stamp

		wandb.log(log)

# Calculate summed topk accuracies for a batch
def calc_topk_sum(output, target, topn=5):
	# output = BxC tensor of floats where larger score means higher predicted probability of class
	# target = B tensor of correct class indices
	num_classes = output.shape[1]
	top_indices = output.topk(min(topn, num_classes), dim=1, largest=True, sorted=True).indices
	topk_tensor = torch.unsqueeze(target, dim=1).eq(top_indices).cumsum(dim=1).sum(dim=0, dtype=float)
	topk_tuple = tuple(topk.item() for topk in topk_tensor)
	return topk_tuple if num_classes >= topn else topk_tuple + (topk_tuple[-1] * (topn - num_classes))

# Run main function
if __name__ == "__main__":
	sys.exit(main())
# EOF
