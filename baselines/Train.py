import numpy as np
from utils import args_parser, CrpsGaussianLoss, EECRPSGaussianLoss
from models import *
from loader import *
import torch
import os
from torch import nn as nn
from tqdm import tqdm
from datetime import datetime
import xarray as xr

best_crps = 0

scale_dict = {"z500": (52000, 7000), "t850": (265, 45), "t2m": (270, 50)}


# Training
def train(epoch, trainloader, model, optimizer, criterion, args, device):
    model.train()
    train_loss = []
    offset, scale = scale_dict[args.target_var]

    for batch_idx, (inputs, targets, scale_mean, scale_std) in tqdm(enumerate(trainloader), desc=f'Epoch {epoch}: ',
                                                                    unit="Batch", total=len(trainloader)):

        curr_iter = epoch * len(trainloader) + batch_idx

        inputs, targets = inputs.to(device), targets.to(device)
        scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
        output = model(inputs)
        if args.model == 'UNet':
            mu = output[:, 0] * scale_std + scale_mean
            sigma = torch.exp(output[:, 1]) * scale_std
        else:
            mu = output[..., 0] * scale_std + scale_mean
            sigma = torch.exp(output[..., 1]) * scale_std
        loss = criterion(mu, sigma, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    print(f'Epoch {epoch} Avg. Loss: {np.average(train_loss)}')


def test(epoch, testloader, model, criterion, args, device):
    global best_crps
    model.eval()
    test_loss = []
    # test_loss_efi = []

    ds_efi = xr.load_dataarray(f"{args.data_path}/efi_{args.target_var}.nc").stack(space=["latitude", "longitude"])
    crps_efi = EECRPSGaussianLoss()

    with torch.no_grad():
        for batch_idx, (dates, inputs, targets, scale_mean, scale_std) in tqdm(enumerate(testloader),
                                                                               desc=f'[Test] Epoch {epoch}: ',
                                                                               unit="Batch", total=len(testloader)):

            inputs, targets = inputs.to(device), targets.to(device)
            scale_mean, scale_std = scale_mean.to(device), scale_std.to(device)
            output = model(inputs)
            if args.model == 'UNet':
                mu = output[:, 0] * scale_std + scale_mean
                sigma = torch.exp(output[:, 1]) * scale_std
            else:
                mu = output[..., 0] * scale_std + scale_mean
                sigma = torch.exp(output[..., 1]) * scale_std
            loss = criterion(mu, sigma, targets)
            test_loss.append(loss.item())

            # for i in range(len(dates)):
            #     date = dates[i]
            #     try:
            #         efi_tensor = torch.as_tensor(ds_efi.sel(time=date).to_numpy()).to(device)
            #         loss_efi = crps_efi(mu[i, ...].reshape(efi_tensor.shape),
            #                             sigma[i, ...].reshape(efi_tensor.shape),
            #                             targets[i, ...].reshape(efi_tensor.shape),
            #                             efi_tensor)
            #         test_loss_efi.append(loss_efi.item())
            #     except KeyError:
            #         pass

    # Save checkpoint.
    crps_loss = np.average(test_loss)
    # crps_loss_efi = np.average(test_loss_efi)
    print(f'Test CRPS: {crps_loss}')

    if crps_loss < best_crps:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'crps': crps_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, f'checkpoint/{args.model}_{args.target_var}_ens{args.ens_num}_best_ckpt.pth')
        best_crps = crps_loss

    print(
        '\ntest/Epoch_crps: ', crps_loss,
        # '\ntest/Epoch_wcrps: ', crps_loss_efi,
        '\ntest/Epoch: ', epoch,
        '\ntest/Best_crps: ', best_crps
    )


def train_model(args, device):
    global best_crps
    best_crps = 1e10  # best test CRPS
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader = loader_prepare(args)

    model = eval(f'{args.model}_prepare')(args)
    model = model.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_crps = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    if args.loss == 'CRPS':
        criterion = CrpsGaussianLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, line_search_fn="strong_wolfe")

    for epoch in range(start_epoch, args.epochs):
        train(epoch, trainloader, model, optimizer, criterion, args, device)
        test(epoch, testloader, model, criterion, args, device)

    return model


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.make_plot:
       print('no plot!')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {device} device')
        model = train_model(args, device)


if __name__ == '__main__':
    args = args_parser()
    main(args)
