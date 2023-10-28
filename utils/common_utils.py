import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# function to get all words not in the word2vec model
def get_words_not_in_model(set, w2v):
    abs_w2v = [] # array to store words that are not in the word2vec model
    abs_w2v_lower = [] # array to store words that are not in the word2vec model, but are in lower case
    for sentences in set:
        for word, type in sentences:
            if word not in w2v.key_to_index:
                abs_w2v.append(word)
                if word.lower() not in w2v.key_to_index:
                    abs_w2v_lower.append(word.lower())
    return abs_w2v, abs_w2v_lower


# takes in a set of sentences in the form of a list of lists of tuples, and 
# returns a list of list of words and tags
def separate_words_tags(set):
    sentences = []
    labels = []

    for sentence in set:
        current_sentence = []
        current_label = []
        for word, tag in sentence:
            current_sentence.append(word)
            current_label.append(tag)
        sentences.append(current_sentence)
        labels.append(current_label)
    return sentences, labels

# class to create a dataset for the NER task
class NERDataset(Dataset):
    def __init__(self, sentences, labels, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.labels = labels
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        original_length = len(sentence)
        
        sentence = [self.word_to_ix.get(word, 0) for word in sentence]  # 0 is for <UNK>
        label = [self.tag_to_ix[tag] for tag in label]
        
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long), original_length


# function to pad the sequences in a batch
def pad_collate(batch):
    (xx, yy, lens) = zip(*batch) # unzip the batch

    x_lens = [len(x) for x in xx] # get lengths of sequences
    y_lens = [len(y) for y in yy] # get lengths of labels

    xx_pad = torch.zeros(len(xx), max(x_lens), dtype=torch.long) # create a matrix of zeros with correct dimensions
    yy_pad = torch.zeros(len(yy), max(y_lens), dtype=torch.long) # create a matrix of zeros with correct dimensions

    for i, (x, y) in enumerate(zip(xx, yy)):
        xx_pad[i, :x_lens[i]] = x
        yy_pad[i, :y_lens[i]] = y
    
    return xx_pad, yy_pad, lens



# remove words with numbers, punctuation, and special characters
# def clean_data(sentence):
#     clean_sentence = []
#     for word, tag in sentence:
#         if word.isalpha():
#             clean_sentence.append([word, tag])
#     return clean_sentence


# word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v.vectors), freeze=False)
# word_embeddings(torch.LongTensor([w2v.key_to_index['<UNK>']]))