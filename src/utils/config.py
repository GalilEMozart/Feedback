class Config:

    debug = False
    nun_worker = 4
    checkpoint = "models/bert-base-uncased"
    gradient_checkpointing = True
    scheduler = 'cosinus'
    epochs = 4
    batch_size = 1
    max_length = 512
    seeds = 42
    n_fold = 4
    train = True
    target_cols=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    lr = 1e-5
    


