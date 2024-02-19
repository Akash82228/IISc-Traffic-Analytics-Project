import os
import torch
import torch.nn as nn
from datetime import datetime
from torch_geometric.loader import DataLoader
from utils import *
from dataloader import SumoTrafficDataset
from generate_adj_mx import *
from evaluate import MAE_torch
from DAAGCN_Config import args
from DAAGCN_Trainer import Trainer
from generator import DAAGCN as Generator
from discriminator import Discriminator, Discriminator_RF
import sys
sys.path.append('../')

def load_data(args):

    dataset = SumoTrafficDataset(root=args.root, window=args.window, horizon=args.horizon, name_scaler=args.normalizer)
    
    number_of_training_sample = int(args.train_ratio * len(dataset))
    number_of_testing_sample = int(args.test_ratio * len(dataset))
    number_of_validation_sample = int(args.val_ratio * len(dataset))

    train = dataset[:number_of_training_sample]
    val = dataset[number_of_training_sample:number_of_training_sample + number_of_validation_sample]
    test = dataset[number_of_training_sample + number_of_validation_sample:]

    print(f"len of train sample: {len(train)}")
    print(f"len of val sample: {len(val)}")
    print(f"len of test sample: {len(test)}")


    train_dataloader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    return dataset, train_dataloader, val_dataloader, test_dataloader, dataset.scaler

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)
    return model

def generate_model_components(args, dataset):
    init_seed(args.seed)
    generator = Generator(args).to(args.device)
    generator = init_model(generator)

    discriminator = Discriminator(args).to(args.device)
    discriminator = init_model(discriminator)
    
    discriminator_rf = Discriminator_RF(args).to(args.device)
    discriminator_rf = init_model(discriminator_rf)

    def masked_mae_loss(min_val, max_val, mask_value):
        def loss(preds, labels):
            preds = preds * (max_val - min_val) + min_val
            labels = labels * (max_val - min_val) + min_val
            mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
            return mae
        return loss

    min_val = dataset.data.x.min().item()
    max_val = dataset.data.x.max().item()

    if args.loss_func == 'mask_mae':
        loss_G = masked_mae_loss(min_val, max_val, mask_value=0.0)
    elif args.loss_func == 'mae':
        loss_G = torch.nn.L1Loss().to(args.device)
    elif args.loss_func == 'mse':
        loss_G = torch.nn.MSELoss().to(args.device)
    elif args.loss_func == 'smoothloss':
        loss_G = torch.nn.SmoothL1Loss().to(args.device)
    elif args.loss_func == 'huber':
        loss_G = torch.nn.HuberLoss(delta=1.0).to(args.device)
    else:
        raise ValueError
    loss_D = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=args.lr_init, eps=1.0e-8,
                                   weight_decay=args.weight_decay, amsgrad=False)
    optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=args.lr_init * 0.1, eps=1.0e-8,
                                   weight_decay=args.weight_decay, amsgrad=False)
    optimizer_D_RF = torch.optim.Adam(params=discriminator_rf.parameters(), lr=args.lr_init * 0.1, eps=1.0e-8,
                                      weight_decay=args.weight_decay, amsgrad=False)

    lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF = None, None, None
    
    if args.lr_decay:
        lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
        lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)
        lr_scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D,
                                                              milestones=lr_decay_steps,
                                                              gamma=args.lr_decay_rate)
        lr_scheduler_D_RF = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_D_RF,
                                                                 milestones=lr_decay_steps,
                                                                 gamma=args.lr_decay_rate)
        
    return generator, discriminator, discriminator_rf, loss_G, loss_D, optimizer_G, optimizer_D, optimizer_D_RF, lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF

def get_log_dir(model, dataset, debug):
    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    current_dir = os.path.abspath(os.path.join(os.getcwd(), "../"))
    log_dir = os.path.join(current_dir, 'log', model, dataset, current_time)
    if not os.path.isdir(log_dir) and not debug:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

if __name__ == '__main__':

    print(args.window)
    print(args.horizon)

    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    dataset, train_dataloader, val_dataloader, test_dataloader, scaler = load_data(args)
    
    _, trv = next(enumerate(train_dataloader))
    _, vav = next(enumerate(val_dataloader))
    _, tev = next(enumerate(test_dataloader))

    print("Here are the sample of training, validation and testing data")
    print(trv)
    print(vav)
    print(tev)

    args.log_dir = get_log_dir(args.model, args.dataset, args.debug)

    generator, discriminator, discriminator_rf, loss_G, loss_D, optimizer_G, optimizer_D, optimizer_D_RF, lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF = generate_model_components(args, dataset)

    trainer = Trainer(
        args, 
        train_dataloader, 
        val_dataloader, 
        test_dataloader, 
        scaler, 
        generator, discriminator, discriminator_rf, 
        loss_G, loss_D, 
        optimizer_G, optimizer_D, optimizer_D_RF, 
        lr_scheduler_G, lr_scheduler_D, lr_scheduler_D_RF
    )

    if args.mode == "train":
        trainer.train()
    elif args.mode == 'test':
        checkpoint = "../log/DAAGCN/PEMSD4/20230210112045/PEMSD4_DAAGCN_best_model.pth"
        trainer.test(generator, args, test_dataloader, scaler, trainer.logger, save_path=checkpoint)
    else:
        raise ValueError
