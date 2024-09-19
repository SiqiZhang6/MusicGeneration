#imports
import math, copy, time
import torch, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch import Tensor,layer_norm
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import preprocessing

#参数
hidden_dim = 512
nhead = 8
nlayer=6
dim_feedforward = 512
dropout = 0.1
embedding_dim=128
epochs = 150
lr = 0.00001
criterion = nn.CrossEntropyLoss()
### main functions###
def getDataLoader(data, batch_size):
  return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=2)
def loadData(allTokensEvents,num_bar):
  X = []
  y = []
  for song in allTokensEvents:
    for i in range(len(song) - num_bar):
      src_bar = []
      for num in range(num_bar):
        src_bar += song[i + num]
      tgt_bar = song[i + num_bar] + [2]
      X.append(src_bar)
      y.append(tgt_bar)

  max_src_bar_len = len(max(X, key = len))
  max_tgt_bar_len = len(max(y, key = len))

  data = []
  for i in range(len(X)):
    src_notes = torch.LongTensor(X[i])
    tgt_notes = torch.LongTensor(y[i])

    src_full = torch.full((max_src_bar_len,), fill_value = 0)
    tgt_full = torch.full((max_tgt_bar_len,), fill_value = 0)

    src_full[:len(src_notes)] = src_notes
    tgt_full[:len(tgt_notes)] = tgt_notes
    data.append([src_full, tgt_full])
  dataloader_combined = getDataLoader(data, 2)
  return dataloader_combined
# clones
def clone(module, N):
  "Produce N identical layers."
  return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
# mask
def subsequent_mask(size):
  attention_shape=(1,size, size)
  subsequent_mask= np.triu(np.ones(attention_shape),k=1).astype('uint8')
  return torch.from_numpy(1-subsequent_mask)
# attention🎶#
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

### main classes###
# layernorm
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
# embedding
class Embedding(nn.Module):
  def __init__(self, embedding_dim, vocab_size):
    super(Embedding,self).__init__()
    self.lut=nn.Embedding(vocab_size, embedding_dim)
    self.embed_dim=embedding_dim
  def forward(self, input):
    return self.lut(input)*math.sqrt(self.embed_dim) 
# positional encoding
class PosEncoding(nn.Module):
  def __init__(self, embedding_dim, dropout,max_len):
    super(PosEncoding,self).__init__()
    self.dropout=nn.Dropout(p=dropout)
    pe=torch.zeros(max_len, embedding_dim)
    position=torch.arange(0,max_len).unsqueeze(1)
    half=torch.exp(torch.arange(0, embedding_dim, 2)* -(math.log(1000.0/embedding_dim)))
    pe[:,0::2]=torch.sin(position*half)
    pe[:,1::2]=torch.cos(position*half)
    pe=pe.unsqueeze(0)
    ####
    self.register_buffer('pe',pe)
  def forward(self,input):
    input=input+Variable(self.pe[:,:input.size(1)],requires_grad=False)
    return self.dropout(input)
# sublayer connection
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
class MultiHeadedAttention(nn.Module):
    def __init__(self, nhead, embedding_dim, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % nhead == 0
        # We assume d_v always equals d_k
        self.d_k = embedding_dim // nhead
        self.h = nhead
        temlinear=nn.Linear(embedding_dim, embedding_dim)
        self.linears = clone(temlinear, 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
#FFW#
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, embedding_dim, dim_feedforward, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(embedding_dim, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# encoder
class Encoder(nn.Module):
  def __init__(self, encoderlayer, nhead):
    super(Encoder,self).__init__()
    self.layers=clone(encoderlayer,nhead)
    self.norm=LayerNorm(encoderlayer.size)
  def forward(self, encoder_input,mask):
    for layer in self.layers:
      encoder_input=layer(encoder_input,mask)
    return self.norm(encoder_input)
class EncoderLayer(nn.Module):
    "Encoder = self-attn + feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
# decoder
class Decoder(nn.Module):
  def __init__(self, decoderlayer, N):
    super(Decoder, self).__init__()
    self.layers = clone(decoderlayer, N)
    self.norm = LayerNorm(decoderlayer.size)
  def forward(self, decoder_input,encoder_input,src_mask,tgt_mask):
    for layer in self.layers:
      decoder_input = layer(decoder_input, encoder_input, src_mask, tgt_mask)
    return self.norm(decoder_input)
class EncoderDecoderLayer(nn.Module):
  def __init__(self,size, self_attn,src_attn,feedforward,dropout):
    super(EncoderDecoderLayer, self).__init__()
    self.size=size
    self.self_attn=self_attn
    self.src_attn=src_attn
    self.feedforward=feedforward
    self.sublayer=clone(SublayerConnection(size, dropout),3)
  def forward(self, decoder_input,encoder_input,src_mask,tgt_mask):
    d=decoder_input
    d = self.sublayer[0](d, lambda d: self.self_attn(d, d, d, tgt_mask))
    d = self.sublayer[1](d, lambda d: self.src_attn(d, encoder_input, encoder_input, src_mask))
    return self.sublayer[2](d, self.feedforward)

# model
class MusicTransformer(nn.Module):
  def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    super(MusicTransformer, self).__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embed = src_embed
    self.tgt_embed = tgt_embed
    self.generator = generator
  #哥偷一下
  def create_mask(self, src, tgt):
    tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1])
    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    return tgt_mask, src_padding_mask, tgt_padding_mask
  #rand mask
  def rand_mask(self, src):
    rand_src_mask = torch.randint(1, (src.shape[0], src.shape[1]))
    return rand_src_mask
  def forward(self, src, tgt,train):
    "Take in and process masked src and target sequences."
    if train:
      src_mask=(src == 0)#ignore padding
      # for i in range(src_mask.size(1)):
      #   src_mask[0][i]=[1 if s=='True' else 0 for s in src_mask[0][i]]
      # src_mask=torch.matmul(src_mask, src_mask.transpose(-2, -1))
      tgt_mask =subsequent_mask(tgt.shape[1])
      # tgt_mask = torch.transformer.generate_square_subsequent_mask(tgt.shape[1])
      return self.decode(self.encode(src, src_mask), src_mask,tgt,tgt_mask)
    else:
      return self.decode(self.encode(src, None), None,tgt,None)
    # return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
  def encode(self, src, src_mask):
    return self.encoder(self.src_embed[0](src), src_mask)
    
  def decode(self, memory, src_mask,tgt,tgt_mask):
    return self.decoder(self.tgt_embed[0](tgt), memory, src_mask, tgt_mask)
  
# generation
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # print("softmax",F.log_softmax(self.proj(x), dim=-1))
        return F.log_softmax(self.proj(x), dim=-1)
def make_model(src_vocab, tgt_vocab,nlayer, embedding_dim,dim_feedforward,nhead,dropout,max_len):
  c = copy.deepcopy
  attn = MultiHeadedAttention(nhead, embedding_dim)
  ff = PositionwiseFeedForward(embedding_dim, dim_feedforward, dropout)
  position = PosEncoding(embedding_dim, dropout, max_len)
  src_embed=nn.Sequential(Embedding(embedding_dim, src_vocab), c(position))
  tgt_embed=nn.Sequential(Embedding(embedding_dim, tgt_vocab), c(position))
  model = MusicTransformer(
        Encoder(EncoderLayer(embedding_dim, c(attn), c(ff), dropout), nlayer),
        Decoder(EncoderDecoderLayer(embedding_dim, c(attn), c(attn), c(ff), dropout), nlayer),
        src_embed,
        tgt_embed,
        Generator(embedding_dim,tgt_vocab))
  return model  
def makeTensor(X,y,max_src_bar_len,max_tgt_bar_len):
  data = []
  for i in range(len(X)):
    src_notes = torch.LongTensor(X[i])
    tgt_notes = torch.LongTensor(y[i])

    src_full = torch.full((max_src_bar_len,), fill_value = 0)
    tgt_full = torch.full((max_tgt_bar_len,), fill_value = 0)

    src_full[:len(src_notes)] = src_notes
    tgt_full[:len(tgt_notes)] = tgt_notes
    data.append([src_full, tgt_full])
  return data
#main loop
def run_epoch(epochs,dataloader, model,modname):
  for epoch in range(1, 1 + epochs):
    train_epoch_loss = 0
    num_correct = 0
    num_total = 0
    model.train()
    for i, (src, tgt) in enumerate(dataloader):
      optimizer.zero_grad()
      # print("a?",src,tgt)
      output = model(src, tgt,True)
      # print("out",output,output.size())
      output=model.generator(output)
      # print("predout",output,output.size())
      pred=torch.argmax(output, dim = 2).data
      loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt.contiguous().view(-1))
      loss.backward()
      # print(loss.item())
      
      optimizer.step()

      train_epoch_loss += loss.item()
      num_correct += torch.sum(pred == tgt)
      num_total+=len(tgt)
    model.eval()
    outputs.append(output)
    train_loss.append(train_epoch_loss)
    train_acc.append(num_correct/num_total)
    if epoch % 2 == 0:
      print('EPOCH %i' % (epoch))
      print("Train loss: %f" % (train_loss[epoch - 1]))
      print("Train acc: %f" % (train_acc[epoch - 1]))
      torch.save(model.state_dict(), './data/modelStates/'+modname)
      if epoch % 4 == 0:
        print("Predictions:",torch.argmax(output, dim = 2).data)#改softmax
        # print("Predictions:", torch.argmax(output, dim = 2))#改softmax
        # encoderHeat(testMod,6,2,tokenEvents)
        # decoderHeat(testMod,6,2,tokenEvents,tokenEvents)
        print("Target:", tgt)
  torch.save(model.state_dict(),'./data/modelStates/'+modname)
#plots
def plotloss(train_loss):
    plt.plot(train_loss)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.show()
    plt.clf()
def plotaccuraccy(train_acc):
    plt.plot(train_acc)
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

#generation
def generate(testMod, iter,numbarXY):
    inputbar=numbarXY[0][0]
    tgtbar=numbarXY[0][1]
    generated_song=[inputbar]
    for i in range(iter):
        prob_tokens = testMod(torch.LongTensor(inputbar).unsqueeze(0), torch.LongTensor(tgtbar).unsqueeze(0), train = False)
        prob=testMod.generator(prob_tokens)
        # final_token = torch.argmax(prob, dim = 2).data
        tok_list=[i for i in range(prob.size(-1))]
        # print("tok_list",tok_list,len(tok_list))
        problist=prob.tolist()[0]
        comp=[]
        for tok in problist:
            #tring avg
            tokavg=sum(tok)/len(tok)
            newtok=[]
            for t in tok:
                if t<=tokavg:
                    newtok.append(0)
                else:
                    newtok.append(t)
            newtokprob=[v/sum(newtok) for v in newtok]
            comp.append(np.random.choice(tok_list,p=newtokprob))
        final_token=comp
        # finallist=final_token[0].tolist()
        # generated_song.append(finallist)
        generated_song.append(final_token)
        # if any([finallist[i] <minpitch or finallist[i] >maxpitch for i in range(0,len(finallist),5)]):
        #     print("debug")
        #     print("src",inputbar)
        #     print("tgt",tgtbar)
        #     print("result",finallist)
        #     print("probs",prob)
        # inputbar=numbarXY[i+1][0]
        tgtbar=numbarXY[i+1][1]
        lasttoklen=len(generated_song[-1])
        inputbar=inputbar[:-lasttoklen]+generated_song[-1]
    return generated_song
def oldgenerate(testMod, iter,numbarXY):
    inputbar=numbarXY[0][0]
    tgtbar=numbarXY[0][1]
    iter=7
    generated_song=[inputbar]
    for i in range(iter):
        prob_tokens = testMod(torch.LongTensor(inputbar).unsqueeze(0), torch.LongTensor(tgtbar).unsqueeze(0), train = False)
        prob=testMod.generator(prob_tokens)
        final_token = torch.argmax(prob, dim = 2).data
        finallist=final_token[0].tolist()
        generated_song.append(finallist)
        inputbar=numbarXY[i+1][0]
        tgtbar=numbarXY[i+1][1]
        lasttoklen=len(generated_song[-1])
        inputbar=inputbar[:-lasttoklen]+generated_song[-1]
    return generated_song
def splitinitbar(rawbar,numnotes):
    allnotes=[]
    splitidx=0
    for i in range(numnotes,len(rawbar),numnotes):
        toadd=rawbar[splitidx:i]
        toadd[1]=16
        allnotes.append(toadd)
        splitidx=i
    return allnotes
def soseosConvert(generated_song,targetname):
    splitinput=[]
    for song in generated_song:
        splitinput+=splitinitbar(song,5)
    testmidi=preprocessing.tokenizer.tokens_to_midi([splitinput])
    testmidi.dump(targetname)
if __name__ == "__main__":
    data_path = './data/vocaloid.mid'
    barnum=2
    tokenEvents=preprocessing.tokenizeSOSEOS(data_path)
    numbarXY=preprocessing.makeNumBarDatasetPad(tokenEvents,barnum,5)
    X=[l[0] for l in numbarXY]
    Y=[l[1] for l in numbarXY]
    # print(numbarXY,X,Y)
    max_src_bar_len = len(max(X, key = len))
    max_tgt_bar_len = len(max(Y, key = len))
    data=[[torch.LongTensor(X[i]),torch.LongTensor(Y[i])]for i in range(len(X))]
    dataloader=getDataLoader(data,1)
    num_events = len(tokenEvents)
    
    minpitch=float('infinity')
    maxpitch=-float('infinity')
    for j in range(len(X)):
      xpitrange=[min([X[j][i] for i in range(0,len(X[j]),5)]),max([X[j][i] for i in range(0,len(X[j]),5)])]
      ypitrange=[min([Y[j][i] for i in range(0,len(Y[j]),5)]),max([Y[j][i] for i in range(0,len(Y[j]),5)])]
      if (xpitrange[0]<minpitch):
        minpitch=xpitrange[0]
      if ypitrange[0]<minpitch:
        minpitch=ypitrange[0]
      if (xpitrange[1]>maxpitch):
        maxpitch=xpitrange[1]
      if ypitrange[1]>maxpitch:
        maxpitch=ypitrange[1]
    print(minpitch,maxpitch)
    # Metrics to evaluate model
    train_loss = []
    train_acc = []
    outputs = []
    testMod=make_model(num_events,num_events,nlayer, embedding_dim,dim_feedforward,nhead,dropout,max_len=num_events)
    optimizer = torch.optim.Adam(testMod.parameters(), lr= lr)
    # run_epoch(epochs, dataloader, testMod, 'vocaloidlowdimff')
    plotloss(train_loss)
    plotloss(train_acc)

    #generation
    testMod.load_state_dict(torch.load('./data/modelStates/vocaloidlowdimff'))
    generated_song=generate(testMod,20,numbarXY)
    print(generated_song)
    soseosConvert(generated_song,'vocaloidlowdimfftesta.mid')