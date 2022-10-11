from torch.utils.data import Dataset 


class Dataset_train(Dataset):
    
    def __init__(self, df, tokenizer):
        
        self.texts = df[['full_text']]
        self.labels = df[['cohesion', 'syntax', 'vocabulary','phraseology', 'grammar', 'conventions']]
        self.tokenizer = tokenizer

    def __len__(self):
        
        return len(self.texts)
        
    def __getitem__(self, idx):

        text = self.tokenizer(self.texts.loc[idx].values[0], padding='max_length', max_length = 512, 
                                truncation=True, return_tensors="pt")
        label = self.labels.loc[idx].values.astype(float)

        return text, label



class Dataset_test(Dataset):
    
    def __init__(self, df, path_pretrained):
        
        self.texts = df[['full_text']]
        self.tokenizer = AutoTokenizer.from_pretrained(path_pretrained)

    def __len__(self):
        
        return len(self.texts)
        
    def __getitem__(self, idx):

        text = self.tokenizer(self.texts.loc[idx].values[0], padding='max_length', max_length = 512, 
                                truncation=True, return_tensors="pt")

        return text

if __name__ == '__main__':
    pass