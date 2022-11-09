import os
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet

from loss import TripletMarginLoss
from sampler import PKSampler
from model import ResNet50, DenseNet121


def train_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    running_loss = 0
    running_frac_pos_triplets = 0
    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        samples, targets = data[0].to(device), data[1].to(device)

        embeddings = model(samples)

        loss, frac_pos_triplets = criterion(embeddings, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_frac_pos_triplets += float(frac_pos_triplets)

        if i % print_freq == print_freq - 1:
            i += 1
            avg_loss = running_loss / print_freq
            avg_trip = 100.0 * running_frac_pos_triplets / print_freq
            print('[{:d}, {:d}] | loss: {:.4f} | % avg hard triplets: {:.2f}%'.format(
                epoch, i, avg_loss, avg_trip))
            running_loss = 0
            running_frac_pos_triplets = 0


def retrieval_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = target[pred].t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].any(dim=0).sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
    return res


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)

    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(torch.tensor(float('-inf')))

    # top-k accuracy (i.e. R@K)
    accuracy = retrieval_accuracy(dists, labels)[0].item()

    print('accuracy: {:.3f}%'.format(accuracy))


def save(model, epoch, save_dir, args):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = args.dataset+'_'+args.model
    if args.embedding_dim:
        file_name += '_embed_'+str(args.embedding_dim)
    if args.anomaly:
        file_name += '_anomaly'
    if args.rand_resize:
        file_name += '_randresize'
    file_name += '_seed_'+str(args.seed)+'_epoch_'+str(epoch)+'_ckpt.pth'

    save_path = os.path.join(save_dir, file_name)
    torch.save(model.state_dict(), save_path)


def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = args.labels_per_batch if not args.anomaly else args.labels_per_batch - 1
    k = args.samples_per_label
    batch_size = p * k

    # Choose model
    if args.model == 'densenet121':
        model = DenseNet121(embedding_dim=args.embedding_dim)
    elif args.model == 'resnet50':
        model = ResNet50(embedding_dim=args.embedding_dim)
    else:
        raise NotImplementedError('Model not supported!')

    if os.path.isfile(args.resume):
        print("=> loading checkpoint")
        checkpoint = torch.load(args.resume)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    model.to(device)

    criterion = TripletMarginLoss(margin=args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                          transforms.Resize(256),
                                          transforms.RandomResizedCrop(
                                              224) if args.rand_resize else transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # Set up dataset and dataloader for embedding learning using triplet loss
    if args.dataset == 'covid':
        train_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'train'),
                                         image_list_file=args.train_image_list,
                                         use_covid=not args.anomaly,  # whether or not to use COVID in training
                                         mask_dir=os.path.join(
                                             args.mask_dir, 'train') if args.mask_dir else None,
                                         transform=train_transform)
        test_dataset = ChestXrayDataSet(data_dir=os.path.join(args.dataset_dir, 'test'),
                                        image_list_file=args.test_image_list,
                                        mask_dir=os.path.join(
                                            args.mask_dir, 'test') if args.mask_dir else None,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        train_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Training_Data'),
                                    image_list_file=args.train_image_list,
                                    use_melanoma=not args.anomaly,  # whether or not to use melanoma in training
                                    mask_dir=os.path.join(
                                        args.mask_dir, 'train') if args.mask_dir else None,
                                    transform=train_transform)
        test_dataset = ISICDataSet(data_dir=os.path.join(args.dataset_dir, 'ISIC-2017_Test_v2_Data'),
                                   image_list_file=args.test_image_list,
                                   mask_dir=os.path.join(
                                       args.mask_dir, 'test') if args.mask_dir else None,
                                   transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    # targets is a list where the i_th element corresponds to the label of i_th dataset element.
    # This is required for PKSampler to randomly sample from exactly p classes. You will need to
    # construct targets while building your dataset. Some datasets (such as ImageFolder) have a
    # targets attribute with the same format.
    targets = train_dataset.labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=PKSampler(targets, p, k),
                              num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    for epoch in range(1, args.epochs + 1):
        print('Training...')
        train_epoch(model, optimizer, criterion, train_loader,
                    device, epoch, args.print_freq)

    print('Evaluating...')
    evaluate(model, test_loader, device)
    print('Saving...')
    save(model, epoch, args.save_dir, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--dataset-dir', default='/data/brian.hu/COVID/data/',
                        help='Dataset directory path')
    parser.add_argument('--train-image-list', default='./train_split.txt',
                        help='Train image list')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--rand-resize', action='store_true',
                        help='Use random resizing data augmentation')
    parser.add_argument('--anomaly', action='store_true',
                        help='Train without anomaly class')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121 or resnet50)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('-p', '--labels-per-batch', default=2, type=int,
                        help='Number of unique labels/classes per batch')
    parser.add_argument('-k', '--samples-per-label', default=32, type=int,
                        help='Number of samples per label in a batch')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='Number of training epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--lr', default=0.0001,
                        type=float, help='Learning rate')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Triplet loss margin')
    parser.add_argument('--print-freq', default=5,
                        type=int, help='Print frequency')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='Model save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
