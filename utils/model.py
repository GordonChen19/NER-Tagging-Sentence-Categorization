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

        self.lstm = nn.LSTM(embedding_dim, hidden_dim) # The LSTM takes word embeddins as inputs, and outputs hidden states with dimensionality hidden_dim

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size) # Linear layer maps from hidden state space to tag space


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) # Embed the input sentence

        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1)) # LSTM layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # Linear layer output
        tag_scores = F.log_softmax(tag_space, dim=1) # Softmax layer
        return tag_scores # Return output
