import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 6
EMBEDDING_DIM = 50

with open("../data/back_to_the_future.txt", "r") as text_file:
    text = text_file.read().split()
text = [word.lower() for word in text[:]]
skipgrams = [([text[i], text[i+1], text[i+2], text[i+4], text[i+5], text[i+6]], text[i+3])
             for i in range(len(text) - 7)]

vocab = set(text)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)


loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)
print("Training word embedding...")
for epoch in range(30):
    losses = []
    for context, target in skipgrams:
        context_indexes = torch.tensor([word_to_index[w] for w in context], dtype=torch.long).cuda()
        model.zero_grad()
        log_probs = model(context_indexes)
        loss = loss_function(log_probs, torch.tensor([word_to_index[target]], dtype=torch.long).cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    torch.save(model.embeddings, "word_embeddings.pth")
    print("Loss: {0}".format(sum(losses)/len(losses)))
print()

embeddings = model.embeddings


class WordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(WordPredictor, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear_out = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        l1 = F.relu(self.linear1(inputs))
        l2 = F.relu(self.linear2(l1))
        out = self.linear_out(l2)
        return out


loss_function = nn.NLLLoss()
model = WordPredictor(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001)

print("Training language model...")
for epoch in range(100):
    losses = []
    for context, target in skipgrams:
        context_indexes = embeddings(torch.tensor([word_to_index[context[4]]], dtype=torch.long).cuda())
        model.zero_grad()
        log_probs = F.log_softmax(model(context_indexes), dim=1)
        loss = loss_function(log_probs, torch.tensor([word_to_index[target]], dtype=torch.long).cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    torch.save(model, "language_model.pth")
    print("Loss: {0}".format(sum(losses)/len(losses)))
print()

STORY_LENGTH = 10000
SEED_WORD = random.choice(list(vocab))
print("OUTPUTTING STORY (seed word = {0})...".format(SEED_WORD))
word = SEED_WORD
story_file = open("results/story.txt", "w")
with torch.no_grad():
    for x in range(STORY_LENGTH):
        input = embeddings(torch.tensor([word_to_index[word]], dtype=torch.long).cuda())
        output = F.softmax(model(input), dim=1).cpu().numpy()[0]
        index = random.choices(range(len(vocab)), weights=output)[0]
        story_file.write(index_to_word[index] + " ")
        if x % 20 == 0:
            story_file.write("\n")
story_file.close()
