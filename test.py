import os
import numpy as np

import torch
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from read_data import ISICDataSet, ChestXrayDataSet

from model import ResNet50, DenseNet121


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


# Source: https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/utils/evaluate.py
def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.

    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images

    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage: 
           mAP = compute_map (ranks, gnd) 
                 computes mean average precsion (mAP) only

           mAP, aps, pr, prs = compute_map (ranks, gnd, kappas) 
                 computes mean average precision (mAP), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query

         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) If there are no positive images for some query, that query is excluded from the evaluation
    """

    mAP = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.where(gnd == gnd[i])[0]

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        # sorted positions of positive images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        mAP = mAP + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    mAP = mAP / (nq - nempty)
    pr = pr / (nq - nempty)

    return mAP, aps, pr, prs


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    embeds, labels = [], []

    for data in loader:
        samples, _labels = data[0].to(device), data[1]
        out = model(samples)
        embeds.append(out)
        labels.append(_labels)
    print('ouuuuut:',out)
    embeds = torch.cat(embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    dists = -torch.cdist(embeds, embeds)
    dists.fill_diagonal_(torch.tensor(float('-inf')))

    # top-k accuracy (i.e. R@K)
    kappas = [1, 5, 10]
    accuracy = retrieval_accuracy(dists, labels, topk=kappas)
    accuracy = torch.stack(accuracy).numpy()
    print('>> R@K{}: {}%'.format(kappas, np.around(accuracy, 2)))

    # mean average precision and mean precision (i.e. mAP and pr)
    ranks = torch.argsort(dists, dim=0, descending=True)
    mAP, _, pr, _ = compute_map(ranks.cpu().numpy(), labels.numpy(), kappas)
    print('>> mAP: {:.2f}%'.format(mAP * 100.0))
    print('>> mP@K{}: {}%'.format(kappas, np.around(pr * 100.0, 2)))

    # Save results
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        file_name = args.resume.split('/')[-1].split('.')[0]

        save_path = os.path.join(args.save_dir, file_name)
        np.savez(save_path, embeds=embeds.cpu().numpy(),
                 labels=labels.cpu().numpy(), dists=-dists.cpu().numpy(),
                 kappas=kappas, acc=accuracy, mAP=mAP, pr=pr)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        if 'state-dict' in checkpoint:
            checkpoint = checkpoint['state-dict']
        model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    model.to(device)

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_transform = transforms.Compose([transforms.Lambda(lambda image: image.convert('RGB')),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # Set up dataset and dataloader
    if args.dataset == 'covid':
        test_dataset = ChestXrayDataSet(data_dir=args.test_dataset_dir,
                                        image_list_file=args.test_image_list,
                                        mask_dir=args.mask_dir,
                                        transform=test_transform)
    elif args.dataset == 'isic':
        test_dataset = ISICDataSet(data_dir=args.test_dataset_dir,
                                   image_list_file=args.test_image_list,
                                   mask_dir=args.mask_dir,
                                   transform=test_transform)
    else:
        raise NotImplementedError('Dataset not supported!')

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=args.workers)

    print('Evaluating...')
    evaluate(model, test_loader, device, args)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Embedding Learning')

    parser.add_argument('--dataset', default='covid',
                        help='Dataset to use (covid or isic)')
    parser.add_argument('--test-dataset-dir', default='/data/brian.hu/COVID/data/test',
                        help='Test dataset directory path')
    parser.add_argument('--test-image-list', default='./test_COVIDx4.txt',
                        help='Test image list')
    parser.add_argument('--mask-dir', default=None,
                        help='Segmentation masks path (if used)')
    parser.add_argument('--model', default='densenet121',
                        help='Model to use (densenet121 or resnet50)')
    parser.add_argument('--embedding-dim', default=None, type=int,
                        help='Embedding dimension of model')
    parser.add_argument('--eval-batch-size', default=64, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers')
    parser.add_argument('--save-dir', default='./results',
                        help='Result save directory')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
