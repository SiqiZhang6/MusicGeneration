# Author: Robert Guthrie
# directly from https://arxiv.org/pdf/1706.09088 see paper for more details
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import preprocessing

torch.manual_seed(1)

# CONTEXT_SIZE = 2
# EMBEDDING_DIM = 128
# data_path='./data/testmidi.mid'

# Print the first 3, just so you can see what they look like.
# print("ngrams",ngrams)
# print("first 3",ngrams[:3])

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
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def embed(CONTEXT_SIZE,EMBEDDING_DIM,alltok):
    test_sentence = [tuple(s) for s in alltok]
    # test_sentence =list(alltok)
    # test_sentence =[torch.tensor(i) for i in alltok]
    #N-Gram
    ngrams = [
        (
            [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
            test_sentence[i]
        )
        for i in range(CONTEXT_SIZE, len(test_sentence))
    ]
    # print("ngrams",ngrams)
    vocab=[]
    for x in test_sentence:
        if x not in vocab:
            vocab.append(x)
    # print("voc",vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    print("worddict",word_to_ix)
    losses = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(10):
        total_loss = 0
        for context, target in ngrams:

            # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
            # into integer indices and wrap them in tensors)
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

            # Step 2. Recall that torch *accumulates* gradients. Before passing in a
            # new instance, you need to zero out the gradients from the old
            # instance
            model.zero_grad()

            # Step 3. Run the forward pass, getting log probabilities over next
            # words
            log_probs = model(context_idxs)

            # Step 4. Compute your loss function. (Again, Torch wants the target
            # word wrapped in a tensor)
            loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))

            # Step 5. Do the backward pass and update the gradient
            loss.backward()
            optimizer.step()

            # Get the Python number from a 1-element Tensor by calling tensor.item()
            total_loss += loss.item()
        losses.append(total_loss)
    return model.embeddings.weight,word_to_ix
# print("loss",losses)  # The loss decreased every iteration over the training data!

# To get the embedding of a particular note
#test if this works
a=preprocessing.tokenizeSOSEOS('./data/testmidi.mid')
b,c=embed(2,128,a)
# print(model.embeddings.weight[word_to_ix[(41, 16, 5, 4, 5)]])
idx=c[(37,  0,  5, 20, 10)]
print(b[idx])