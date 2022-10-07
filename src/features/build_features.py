import torch.utils.data import Dataset 
from transformers import AutoTokenizer


class Dataset_train(Dataset):
    
    def __init__(self, df, checkpoint):
        
        self.texts = df[['full_tex']]
        self.lables = df[['cohesion', 'syntax', 'vocabulary','phraseology', 'grammar', 'conventions']]
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def __len__(self):
        
        return len(self.texts)
        
    def __getitem__(self, idx):

        text = self.tokenizer(self.texts.loc[idx].values[0], padding='max_length', max_length = 512, 
                                truncation=True, return_tensors="pt")
        label = self.label.loc[idx].values.astype(float)

        return text, label



class Dataset_test(Dataset):
    
    def __init__(self, df, path_pretrained):
        
        self.texts = df[['full_tex']]
        self.tokenizer = AutoTokenizer.from_pretrained(path_pretrained)

    def __len__(self):
        
        return len(self.texts)
        
    def __getitem__(self, idx):

        text = self.tokenizer(self.texts.loc[idx].values[0], padding='max_length', max_length = 512, 
                                truncation=True, return_tensors="pt")

        return text

if __name__ == '__main__':
    