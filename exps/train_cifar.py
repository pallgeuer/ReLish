#!/usr/bin/env python

import os
import os.path
import time
import argparse

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import wandb

try:
    from ranger import Ranger
except ImportError:
    Ranger = None

import resnet

config_defaults = dict(
    version=20,
    act="relu",
    batch_size=32,
    optimizer="adam",
)

best_prec1 = 0
evaluate = True

def main():

    global best_prec1, evaluate

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=os.path.expanduser("~/Datasets/CIFAR"), metavar="PATH", help="Path to the dataset root directory (directory containing cifar-* subdirectories)")
    parser.add_argument("--num_workers", type=int, default=2, metavar="NUM", help="Number of dataset loader workers")
    args = parser.parse_args()

    wandb.init(project="ReLish", entity="pallgeuer", config=config_defaults)

    model = resnet.resnet_model(wandb.config.version, act=wandb.config.act, num_classes=10)
    wandb.watch(model)
    model = model.cuda()

    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=args.dataset_root,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=wandb.config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            root=args.dataset_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=wandb.config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = create_optimizer(wandb.config.optimizer, model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)
    max_epoch = 200

    for epoch in range(0, max_epoch):

        print("current lr {:.5e}".format(optimizer.param_groups[0]["lr"]))
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % 20 == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                },
                filename="train_cifar_checkpoint.pth",
            )

        save_checkpoint(
            {
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
            },
            filename="train_cifar_model.pth",
        )

    wandb.run.finish()


def create_optimizer(name, params):
    if name == "sgd":
        return torch.optim.SGD(params, 0.1, momentum=0.9, weight_decay=5e-4)
    elif name == "adam":
        return torch.optim.Adam(params)
    elif name == "ranger" and Ranger is not None:
        return Ranger(params)  # noqa
    else:
        raise ValueError(f"Invalid optimizer specification: {name}")


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data = data.cuda()
        target = target.cuda()

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )


def validate(val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data, target) in enumerate(val_loader):

        data = data.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))

    wandb.log(
        {
            "epoch": epoch,
            "Top-1 accuracy": top1.avg,
            "loss": losses.avg,
        }
    )

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    """
    Save the training model
    """
    torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
