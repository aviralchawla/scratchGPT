import torch

class Tokenizer:
    """
    Simple character level tokenization, can be replaced with any other tokenizer like TikToken

    Parameters:
    data: str
        Text data to be tokenized
    """
    def __init__(self, data: str):
        self.data = data
        unique_chars = list(set(data))
        self.char2idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, text):
        return [self.char2idx[char] for char in text]
    
    def decode(self, tokens):
        return ''.join([self.idx2char[idx] for idx in tokens])
    
    def __len__(self):
        return self.vocab_size
    
    def __call__(self, text):
        return self.encode(text)

class Dataset:
    """
    Dataset class for character level language modeling

    Parameters:
    data: str
        Text data to be tokenized
    tokenizer: Tokenizer
        Tokenizer object to be used for tokenization
    block_size: int
        Number of tokens in each block
    batch_size: int
        Number of blocks in each batch
    """
    def __init__(self, data:str, tokenizer: Tokenizer, block_size:int=32, batch_size:int=4):
        self.data = data
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        text = self.data[start_idx:end_idx]
        tokens = self.tokenizer(text)
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y
    
    def get_batch(self):
        idx = torch.randint(0, len(self) - self.batch_size)
        x_batch, y_batch = [], []
        for i in range(self.batch_size):
            x, y = self[idx + i]
            x_batch.append(x)
            y_batch.append(y)
        return torch.stack(x_batch), torch.stack(y_batch)
        
    def train_val_split(self, train_size=0.9):
        train_size = int(len(self.data) * train_size)
        train_data = self.data[:train_size]
        val_data = self.data[train_size:]
        return Dataset(train_data), Dataset(val_data)
    