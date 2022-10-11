from src.features.build_features import  Dataset_train
import torch
from torch import nn
from src.utils.config import Config
from predict_model import FeedbackModel
from tqdm import tqdm

from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import pandas as pd
import click
from transformers import AutoTokenizer
import sys

def train(model, tokenizer,train_data, val_data, epochs):

    train, val = Dataset_train(train_data, tokenizer), Dataset_train(val_data, tokenizer)
    
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    criteron = nn.MSELoss()

    optimizer = Adam(model.parameters(), lr = Config.lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=1e-6)

    train_dataloader = DataLoader(train, batch_size = Config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size = Config.batch_size, shuffle=True)

    if torch.cuda.is_available():
        model = model.cuda()
        optimizer = optimizer.cuda()

    for epoch in range(epochs):
        
        total_loss_train, total_loss_val = 0, 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            print(train_input)
            #mask = train_input['attention_mask'].to(device)
            #input_id = train_input['input_ids'].to(device)

            output = model(train_input)

            batch_loss = criteron(output, train_label)
            total_loss_train += batch_loss

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            scheduler.step()

            print('Done')
            sys.exit()

        with torch.no_grad():
            for val_input, val_label in tqdm(val_dataloader):

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].to(device)

                output = model(input_id, mask)

                batch_loss = criteron(output, val_label)
                total_loss_train += batch_loss



@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('kfold', type=click.INT)
def main(data_filepath, kfold):
    
    data = pd.read_csv(data_filepath)
    data_train = data[data['kfold']==kfold].reset_index(drop=True)
    data_val = data[data['kfold']!=kfold].reset_index(drop=True)

    model = FeedbackModel()
    tokenizer = AutoTokenizer.from_pretrained(Config.checkpoint)


    train(model,tokenizer, data_train, data_val, Config.epochs)
    print('Done !')


if __name__ == '__main__':
    #log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()