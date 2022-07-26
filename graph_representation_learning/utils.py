import sys

def print_weights(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    sys.stdout.flush()


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    train_acc, test_acc = info['train_acc'], info['test_acc']
    print('{:02d}/{:03d}: Train Acc: {:.7f}, Test Accuracy: {:.7f}'.format(
        fold, epoch, train_acc, test_acc))
    sys.stdout.flush()


