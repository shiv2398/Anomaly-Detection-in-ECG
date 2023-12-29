import torch.nn as nn 
class EncoderDecoderModel(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(EncoderDecoderModel, self).__init__()
        
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim,embedding_dim
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=self.hidden_dim, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim*2, num_layers=1, batch_first=True)
        self.out = nn.Linear(self.hidden_dim*2, n_features)

    def forward(self, x):
        x = x.reshape((1, self.seq_len, self.n_features))
        x, (hidden_, _) = self.encoder(x)
        x = hidden_.reshape((self.n_features, self.embedding_dim)).repeat(self.seq_len, self.n_features)
        x = x.reshape((self.n_features, self.seq_len, self.embedding_dim))
        x,(hidden_,_)=self.decoder(x)
        x = x.reshape((self.seq_len, self.hidden_dim*2))
        x=self.out(x)
        return x