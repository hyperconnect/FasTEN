from __future__ import print_function

import argparse
import os
import random
import sys
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.optim as optim

from dataloader.FasTEN import dataloader_cifar as dataloader
from dataloader.downloader_cifar import CifarDownloader
from model.cifar.ResNet import *
from utils.value_aggregator import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
# Method
parser.add_argument('--method', default='FasTEN', type=str)
parser.add_argument('--use_correction', default=True, type=bool, help='Use correction.')
parser.add_argument('--thres_upper', default=0.80, type=float, help='threshold')
parser.add_argument('--lambda_n', default=0.5, type=float, help='noisy loss weight')
# Dataset
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--num_clean', default=1000, type=int)
parser.add_argument('--nPerImage', default=10, type=int)
parser.add_argument('--use_valid', default=True, type=bool, help='Use validation set for training.')
parser.add_argument('--root_path', default='nas/workspace/harris/public/cifar', type=str, help='path to dataset')
# Optimization
parser.add_argument('--batch_size', default=100, type=int, help='train batchsize')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=70, type=int)
# Noise setting
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
# Etc.
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--prefetch', default=0, type=int)
parser.add_argument('--exp', default='exp_test', type=str)
args = parser.parse_args()
args.num_classes = {
    "cifar10": 10,
    "cifar100": 100
}[args.dataset]
args.data_path = {
    "cifar10": f"{args.root_path}/cifar-10-batches-py",
    "cifar100": f"{args.root_path}/cifar-100-python"
}[args.dataset]
print(args)

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def save_checkpoint(state, checkpoint_dir, filename='model_best.pth.tar'):
    torch.save(state, checkpoint_dir / filename)


def load_checkpoint(model, checkpoint_dir):
    global best_prec1, start_epoch

    model_path = checkpoint_dir / 'model_best.pth.tar'
    if os.path.isfile(model_path):
        print(f"=> loading checkpoint {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    else:
        print(f"=> no checkpoint found at {model_path}")
        return model


# Training
def train(epoch, net, optimizer, trainloader, trainloader_c, args):
    global best_prec1, start_epoch, save_path, relabels

    net.train()

    num_iter = (len(trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (input_noisy, label_noisy, index) in enumerate(trainloader):
        input_clean, label_clean = next(iter(trainloader_c))

        input_noisy, label_noisy = input_noisy.cuda(), label_noisy.cuda()
        input_clean, label_clean = input_clean.cuda(), label_clean.cuda()

        y_noisy_n, y_clean_n = net(input_noisy)
        y_noisy_c, y_clean_c = net(input_clean)
        pre1 = torch.softmax(y_clean_n, dim=1)  # p(y|x)

        if args.use_correction:
            # Apply cleansing
            thres_upper = args.thres_upper
            max_prob, relabel = torch.max(pre1, dim=1)

            corrected_inst = max_prob.ge(thres_upper)
            # Get new label
            new_label = (~corrected_inst) * label_noisy + corrected_inst * relabel
            # Get noisy label from prev relabels
            label_noisy = torch.tensor(relabels[index]).cuda()

            # Label correction
            new_label_cpu = new_label.cpu().detach().numpy()
            relabels[index] = new_label_cpu.tolist()

        # Estimate transition matrix
        prob_noisy_c = torch.softmax(y_noisy_c, dim=1)
        c_hat = (prob_noisy_c.reshape(args.num_classes, args.nPerImage, -1)).mean(dim=1)
        c_hat = c_hat.detach()
        c_hat_transpose = (c_hat).T

        _c_hat_transpose = c_hat_transpose[label_noisy]  # p(y_hat=j|y, x)
        pre1 = torch.softmax(y_clean_n, dim=1)  # p(y|x)
        pre2 = torch.sum(_c_hat_transpose * pre1, dim=1)  # p(y_hat=j|x) = sum_y[p(y_hat=j|y,x) * p(y|x)]

        eps = 1e-7
        l_c_n = -(torch.log(pre2 + eps))
        l_c_c = F.cross_entropy(y_clean_c, label_clean, reduction="none")
        l_c = torch.cat([l_c_n, l_c_c], dim=0).mean()

        l_n = F.cross_entropy(y_noisy_n, label_noisy, reduction='mean')
        loss = l_c + l_n * args.lambda_n

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode,
                            epoch, args.num_epochs, batch_idx + 1, num_iter, loss.item()))
        sys.stdout.flush()


def test(epoch, net, loader, mode='test'):
    global best_prec1, best_prec1_test, best_epoch

    losses_test = AverageMeter()
    top1_test = AverageMeter()
    net.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            _, outputs = net(inputs)
            loss_test = F.cross_entropy(outputs, targets, reduction='mean')
            prec_test = accuracy(outputs.data, targets.data, topk=(1,))[0]

            losses_test.update(loss_test.data.item(), inputs.size(0))
            top1_test.update(prec_test.item(), inputs.size(0))

    loss_test = losses_test.avg
    top1_test = top1_test.avg

    print(f"\n| {mode} Epoch #%d\t{mode}_loss:%.2f\tAccuracy: %.2f%%\n" % (epoch, loss_test, top1_test))
    test_log.write(f'Epoch:%d {mode}_loss:%.2f\tAccuracy:%.2f\n' % (epoch, loss_test, top1_test))
    test_log.flush()

    if (mode == 'test') and (epoch == best_epoch):
        best_prec1_test = top1_test

    if mode == 'valid':
        best_prec1 = max(top1_test, best_prec1)
        if best_prec1 == top1_test:
            best_epoch = epoch

            checkpoint_dict = {
                "epoch": epoch,
                "arch": "ResNet34",
                "state_dict": net.state_dict(),
                "best_acc1": best_prec1,
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(
                checkpoint_dict,
                checkpoint_dir=save_path,
                filename="model_best.pth.tar".format(epoch)
            )

        print(f"Best Valid Prec@1 is {best_prec1} at epoch {best_epoch}")
    else:
        print(f"Test Prec@1 is {top1_test}\t Best Test Prec@1 is {best_prec1_test}")


def create_model():
    model = ResNet34(num_classes=args.num_classes)
    model = model.cuda()
    return model


if __name__ == '__main__':
    start_epoch = 1
    accum_flops = 0

    best_prec1 = 0
    best_prec1_test = 0
    best_epoch = 0

    save_dir = '../checkpoint'
    save_path = Path(os.path.join(save_dir, args.exp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    test_log = open(f'{save_dir}/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

    cifar_downloader = CifarDownloader(args.root_path, dataset=args.dataset)
    cifar_downloader.download()

    loader = dataloader.cifar_dataloader(args.dataset, r=args.r, noise_mode=args.noise_mode, batch_size=args.batch_size,
                                         num_workers=0, data_dir=args.data_path)

    print('| Building net')
    net = create_model()
    cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.use_valid:
        trainloader, trainloader_c, validloader = loader.run(mode='train', args=args)
        print(f"Size of train noisy set : {len(trainloader.dataset)}")
        print(f"Size of train clean set : {len(trainloader_c.dataset)}")
        print(f"Size of valid set : {len(validloader.dataset)}")
    else:
        trainloader, trainloader_c = loader.run(mode='train', args=args)
        print(f"Size of train noisy set : {len(trainloader.dataset)}")
        print(f"Size of train clean set : {len(trainloader_c.dataset)}")
    testloader = loader.run('test')
    print(f"Size of test set : {len(testloader.dataset)}")

    relabels = np.array(trainloader.dataset.train_label)
    for epoch in range(start_epoch, args.num_epochs + 1):
        lr = args.lr
        if 60 > epoch >= 50:
            lr /= 10
        elif epoch >= 60:
            lr /= 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print('| Train Net')
        train(epoch, net, optimizer, trainloader, trainloader_c, args)

        if args.use_valid:
            print('\n| Valid Net')
            test(epoch, net, validloader, mode='valid')

        print('\n| Test Net')
        test(epoch, net, testloader, mode='test')

    print(f"\n| Test Epoch@{best_epoch}\tAccuracy: %.2f%%\n" % (best_prec1_test))
