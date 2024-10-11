import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import datetime
from tqdm import tqdm
import sys
from functools import partial

from models.model_classifier import ResNet, ResidualBlock
from models.utils import EarlyStopping, Tee
from dataset.dataset_ESC50 import ESC50
import config


def set_seeds(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seeds(42)


# Mixup functions
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Evaluate model on different testing data 'dataloader'
def test(model, dataloader, criterion, device):
    model.eval()
    losses = []
    corrects = 0
    samples_count = 0
    probs = {}
    with torch.no_grad():
        for k, x, label in tqdm(dataloader, unit='bat', disable=config.disable_bat_pbar, position=0):
            x = x.float().to(device)
            y_true = label.to(device)

            y_prob = model(x)
            loss = criterion(y_prob, y_true)
            losses.append(loss.item())

            y_pred = torch.argmax(y_prob, dim=1)
            corrects += (y_pred == y_true).sum().item()
            samples_count += y_true.shape[0]
            for w, p in zip(k, y_prob):
                probs[w] = [float(v) for v in p]

    acc = corrects / samples_count
    return acc, losses, probs


def train_epoch(model, train_loader, optimizer, criterion, device, alpha=0.2, use_mixup=True):
    model.train()
    losses = []
    corrects = 0
    samples_count = 0
    for _, x, label in tqdm(train_loader, unit='bat', disable=config.disable_bat_pbar, position=0):
        x = x.float().to(device)
        y_true = label.to(device)

        if use_mixup:
            x, targets_a, targets_b, lam = mixup_data(x, y_true, alpha)
            x, targets_a, targets_b = map(torch.autograd.Variable, (x, targets_a, targets_b))

        y_prob = model(x)
        if use_mixup:
            loss = mixup_criterion(criterion, y_prob, targets_a, targets_b, lam)
        else:
            loss = criterion(y_prob, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        y_pred = torch.argmax(y_prob, dim=1)
        if use_mixup:
            corrects += (lam * y_pred.eq(targets_a.data).cpu().sum().float() + (1 - lam) * y_pred.eq(
                targets_b.data).cpu().sum().float())
        else:
            corrects += (y_pred == y_true).sum().item()
        samples_count += y_true.shape[0]

    acc = corrects / samples_count
    return acc, losses


def fit_classifier():
    num_epochs = config.epochs
    loss_stopping = EarlyStopping(patience=15, delta=0.002, verbose=True, float_fmt=float_fmt,
                                  checkpoint_file=os.path.join(experiment, 'best_val_loss.pt'))
    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    for epoch in range(1, 1 + num_epochs):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion, device, alpha=0.2,
                                            use_mixup=True)

        pbar.update()
        print(end=' ')
        print(f"TrnAcc={train_acc:{float_fmt}}",
              f"TrnLoss={np.mean(train_loss):{float_fmt}}",
              end=' ')
        scheduler.step()
    torch.save(model.state_dict(), os.path.join(experiment, 'terminal.pt'))


def make_model():
    n = config.n_classes
    model_constructor = config.model_constructor
    print(model_constructor)
    model = eval(model_constructor)
    return model


if __name__ == "__main__":
    data_path = config.esc50_path
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{config.device_id}" if use_cuda else "cpu")

    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    runs_path = config.runs_path
    experiment_root = os.path.join(runs_path, str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')))
    if not os.path.exists(experiment_root):
        os.mkdir(experiment_root)

    scores = {}
    for test_fold in config.test_folds:
        experiment = os.path.join(experiment_root, f'{test_fold}')
        if not os.path.exists(experiment):
            os.mkdir(experiment)

        with Tee(os.path.join(experiment, 'train.log'), 'w', 1, encoding='utf-8',
                 newline='\n', proc_cr=True):
            get_fold_dataset = partial(ESC50, test_folds={test_fold}, root=data_path, download=True)

            train_set = get_fold_dataset(subset="train")
            print('*****')
            print(f'train folds are {train_set.train_folds} and test fold is {train_set.test_folds}')
            print('random wave cropping')

            train_loader = DataLoader(train_set,
                                      batch_size=config.batch_size,
                                      shuffle=True,
                                      num_workers=config.num_workers,
                                      drop_last=False,
                                      persistent_workers=config.persistent_workers,
                                      pin_memory=True,
                                      )


            print()
            model = make_model()
            model = model.to(device)
            print('*****')

            criterion = nn.CrossEntropyLoss().to(device)

            # Using AdamW optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=config.step_size,
                                                        gamma=config.gamma)
            print()
            fit_classifier()

            test_loader = DataLoader(get_fold_dataset(subset="test"),
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     drop_last=False,
                                     )

            print(f'\ntest {experiment}')
            test_acc, test_loss, _ = test(model, test_loader, criterion=criterion, device=device)
            scores[test_fold] = pd.Series(dict(TestAcc=test_acc, TestLoss=np.mean(test_loss)))
            print(scores[test_fold])
            print()
    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(['mean', 'std']))))
