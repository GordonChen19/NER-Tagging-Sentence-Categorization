import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# LSTM Model
class NERModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, weights_matrix=None, freeze_weights=True):
        super(NERModel, self).__init__()

        self.hidden_dim = hidden_dim 

        if weights_matrix is not None:
            self.word_embeddings = nn.Embedding.from_pretrained(weights_matrix, freeze=freeze_weights) # initialize word embeddings with pretrained weights and freeze them
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) 

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size) # Linear layer maps from hidden state space to tag space


    def forward(self, sentences):
        embeds = self.word_embeddings(sentences) # Embed the input sentence

        # LSTM input shape: batch_size x max_seq_length x embedding_dim
        lstm_out, _ = self.lstm(embeds)
        
        # LSTM output shape: batch_size x max_seq_length x hidden_dim
        # reshape it for the linear layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) # shape: (batch_size * max_seq_length) x hidden_dim

        tag_space = self.hidden2tag(lstm_out)

        # reshape back to batch_size x max_seq_length x tagset_size
        tag_space = tag_space.contiguous().view(sentences.shape[0], sentences.shape[1], -1)

        # swap dimensions to make it batch_size x tagset_size x max_seq_length
        tag_space = tag_space.permute(0, 2, 1)

        return tag_space # Return output
