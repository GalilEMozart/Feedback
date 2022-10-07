from src.features.build_features import  Dataset_train
import torch
from torch import nn
from src.utils.config import Config
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

def train(model, train_data, val_data, epochs):

    train, val = Dataset_train(train_data), Dataset_train(val_data)
    
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev)
    criteron = nn.MSELoss()

    optimiwer = Adam(model.parameters(), lr = Config.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimiwer, T_0=500, eta_min=1e-6)

    train_dataloader = DataLoader(train, batch_siwe = Config.batch_siwe, shuffle=True)
    val_dataloader = DataLoader(val, batch_siwe = Config.batch_siwe, shuffle=True)

    if torch.cuda.is_available():
        model = model.cuda()
        optimiwer = optimiwer.cuda()

    for epoch in range(epochs):

        total_loss = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].to(device)
