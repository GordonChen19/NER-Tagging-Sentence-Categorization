import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm 
import time
import os
import random
import numpy as np
import string


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


# function to remove words that are puncutation, numbers, or special characters
def remove_punc_num_special(s):
    clean_set = []
    for sentence in s:
        clean_sentence = []
        for word, tag in sentence:
            if any(c.isalpha() for c in word):
                clean_sentence.append([word, tag])
        clean_set.append(clean_sentence)
    return clean_set



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
        
        sentence = [self.word_to_ix.get(word, 3000000) for word in sentence] # 3000000 is the index for <UNK>
        label = [self.tag_to_ix[tag] for tag in label]
        
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long), original_length


# function to pad the sequences in a batch
def pad_collate(batch):
    (xx, yy, lens) = zip(*batch)  # unzip the batch

    x_lens = [len(x) for x in xx]  # get lengths of sequences
    y_lens = [len(y) for y in yy]  # get lengths of labels

    max_x_len = max(x_lens)
    max_y_len = max(y_lens)

    xx_pad = torch.full((len(xx), max_x_len), 3000000, dtype=torch.long)  # create a matrix filled with 3000000
    yy_pad = torch.zeros(len(yy), max_y_len, dtype=torch.long)  # create a matrix of zeros with correct dimensions

    for i, (x, y) in enumerate(zip(xx, yy)):
        xx_pad[i, :x_lens[i]] = x
        yy_pad[i, :y_lens[i]] = y
    
    return xx_pad, yy_pad, lens




# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience # how many epochs to wait before stopping when loss is no longer decreasing
        self.min_delta = min_delta # minimum difference between new loss and old loss to be considered as a decrease in loss
        self.counter = 0 # number of epochs since loss was last decreased
        self.min_validation_loss = np.inf # minimum validation loss achieved so far

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss: # new loss is lower than old loss
            self.min_validation_loss = validation_loss # update minimum loss
            self.counter = 0 # reset counter
        elif validation_loss > (self.min_validation_loss + self.min_delta): # new loss is higher than old loss + minimum difference
            self.counter += 1 # increase counter
            if self.counter >= self.patience:
                return True # stop training
        return False # continue training


# set random seed
def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 


# Train step
def train_step(model, trainloader, optimizer, device, lossfn):
    model.train()  # set model to training mode
    total_loss = 0.0

    # Iterate over the training data
    for i, data in trainloader:
        inputs, labels, _ = data  # get the inputs and labels
        inputs, labels = inputs.to(device), labels.to(device)  # move them to the device

        optimizer.zero_grad()  # zero the gradients

        # Forward pass
        outputs = model(inputs)
        loss = lossfn(outputs, labels)

        # Backward pass and optimisation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # accumulate the loss
        trainloader.set_postfix({'Training loss': '{:.4f}'.format(total_loss/(i+1))})  # Update the progress bar with the training loss

    train_loss = total_loss / len(trainloader)
    return train_loss


from sklearn.metrics import f1_score


# Test step
def val_step(model, valloader, lossfn, device):
    model.eval() # set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total_words = 0

    with torch.no_grad(): # disable gradient calculation
        for data in valloader:
            inputs, labels, lens = data # get the inputs and labels and actual lengths
            inputs, labels = inputs.to(device), labels.to(device) # move them to the device

            # Forward pass
            outputs = model(inputs)
            loss = lossfn(outputs, labels)

            total_loss += loss.item()

            # Get the index of the max log-probability along the tagset_size dimension
            _, predicted = torch.max(outputs.permute(0, 2, 1), 2)

            batch_preds = []
            batch_labels = []

            for i in range(len(lens)):
                batch_preds.extend(predicted[i, :lens[i]].cpu().numpy())  # Append predictions but only up to the actual length of the sentence
                batch_labels.extend(labels[i, :lens[i]].cpu().numpy())

            correct += sum(p == l for p, l in zip(batch_preds, batch_labels))  # Accumulate correct predictions for this batch
            total_words += sum(lens)  # Accumulate the actual sentence lengths for this batch

    val_loss = total_loss / len(valloader)
    accuracy = 100 * correct / total_words
    f1 = f1_score(batch_labels, batch_preds, average='macro') 

    return val_loss, accuracy, f1




# Save model
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Training loop
def train(model, tl, vl, opt, loss, device, epochs, early_stopper, path):
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    f1_list = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        start_time = time.time()  # Record the start time of the epoch

        # Wrap the trainloader with tqdm for the progress bar
        pbar = tqdm(enumerate(tl), total=len(tl), desc=f"Epoch {epoch+1}/{epochs}")

        train_loss = train_step(model, pbar, opt, device, loss)  # Pass the tqdm-wrapped loader
        val_loss, val_acc, F1 = val_step(model, vl, loss, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        f1_list.append(F1)

        # Print time taken for epoch
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'Epoch {epoch+1}/{epochs} took {elapsed_time:.2f}s | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.2f}% | F1: {F1:.4f} | EarlyStopper count: {early_stopper.counter}')

        # save as last_model after every epoch
        save_model(model, os.path.join(path, 'last_model.pt'))

        # save as best_model if validation loss is lower than previous best validation loss
        if val_loss < early_stopper.min_validation_loss:
            save_model(model, os.path.join(path, 'best_model.pt'))

        if early_stopper.early_stop(val_loss):
            print('Early stopping')
            break

    return train_loss_list, val_loss_list, val_acc_list, f1_list




# remove words with numbers, punctuation, and special characters
# def clean_data(sentence):
#     clean_sentence = []
#     for word, tag in sentence:
#         if word.isalpha():
#             clean_sentence.append([word, tag])
#     return clean_sentence


# word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(w2v.vectors), freeze=False)
# word_embeddings(torch.LongTensor([w2v.key_to_index['<UNK>']]))