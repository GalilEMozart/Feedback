import torch
from transformers import AutoModel
from torch import nn 
from torch.optim import Adam
from tqdm import tqdm
from src.utils.config import Config


class FeedbackModel(nn.Module):
    
    def __init__(self, dropout=0.1):

        super(FeedbackModel, self).__init__()

        self.bert = AutoModel.from_pretrained(Config.checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(1024,256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256,6)

    def forward(self, input_id):

        output = self.bert(**input_id)
        print(output)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.out(x)

        return x 

        

