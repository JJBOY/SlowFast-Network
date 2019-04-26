import os
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_
from dataset import VideoDataset, get_augmentation
from slowfastnet import SlowFastNet
from transforms import *
from config import parser
from util_ops import record_info, AverageMeter, WarmUpMultiStepLR, accuracy, save_checkpoint

input_size = 224
input_mean = [0.485, 0.456, 0.406]
input_std = [0.229, 0.224, 0.225]
scale_size = input_size * 256 // 224
iter = 0

best_prec1 = 0
def main():

    global args, best_prec1

    args = parser.parse_args()

    if not os.path.exists('./record'):
        os.mkdir('./record')

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'sthsth':
        num_class = 174
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = SlowFastNet(num_class)
    train_augmentation = get_augmentation('RGB', input_size)
    model = torch.nn.DataParallel(model).cuda()

    args.start_epoch=0
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    normalize = torchvision.transforms.Compose([GroupNormalize(input_mean, input_std),f2Dt3D()])

    train_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.train_list,
                     transform=torchvision.transforms.Compose([
                         train_augmentation,
                         Stack(roll=False),
                         ToTorchFormatTensor(div=True),
                         normalize,
                     ]), mode='train', T=args.T, tau=args.tau, dense_sample=not args.no_dense_sample),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        VideoDataset(args.root_path, args.val_list,
                     transform=torchvision.transforms.Compose([
                         GroupScale(int(scale_size)),
                         GroupCenterCrop(input_size),
                         Stack(roll=False),
                         ToTorchFormatTensor(div=True),
                         normalize,
                     ]), mode='test', T=args.T, tau=args.tau, dense_sample=not args.no_dense_sample),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    schduler = WarmUpMultiStepLR(optimizer, [20, 30, 40], 0.1, last_epoch=args.start_epoch-1)

    # the way in the raw paper ,But I do not use it, because I can't estimate how many iter to train
    # max_step = len(train_loader)*args.epochs
    # lr_lambda = lambda step: 0.5 * args.lr* ((np.cos(step / max_step * np.pi)) + 1)
    # scheduler = torch.nn.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_lambda])
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        schduler.step()
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, epoch + 1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input = input.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)
            # if total_norm > args.clip_gradient:
            #    print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    info = {'Epoch': [epoch + 1],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            'lr': optimizer.param_groups[0]['lr']
            }
    record_info(info, args.record_path + 'train.csv', 'train')


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    info = {'Epoch': [epoch + 1],
            'Batch Time': [round(batch_time.avg, 3)],
            'Epoch Time': [round(batch_time.sum, 3)],
            'Data Time': [round(data_time.avg, 3)],
            'Loss': [round(losses.avg, 5)],
            'Prec@1': [round(top1.avg, 4)],
            'Prec@5': [round(top5.avg, 4)],
            }
    record_info(info, args.record_path + 'test.csv', 'test')

    return top1.avg


if __name__ == '__main__':
    main()
