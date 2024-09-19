import os
from miditok import Octuple, TokenizerConfig,REMI;
from miditoolkit import MidiFile
import numpy as np
import pprint,torch

pitch_range = (20, 110)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Rest': True,'sos':True, 'eos':True, 'bar':True}
tokenizerconfig= TokenizerConfig(pitch_range, beat_res, nb_velocities, additional_tokens,sos_eos = True, pad = True)
tokenizer=Octuple(tokenizerconfig)
# tokenizer=REMI(tokenizerconfig) ←used by Daniel

# rest=0, 'sos_None': 1, 'eos_None': 2, 'bar_None': 3
def get_translate(data_path):
    #returns initial velo, pos, barnum
    maxpos=max([i[-2] for i in tokenize(data_path)])
    tok_Event1=tokenize(data_path)[0]
    return tok_Event1[1],tok_Event1[-2],tok_Event1[-1],maxpos
def gapping(allTokensEvents,gapamount):
    """make gaps in the datapoint"""
    gapped=[]
    random0=np.random.choice([0,1],len(allTokensEvents),p=gapamount)
    print(random0)
    for i in range(len(allTokensEvents)):
        if random0[i]==1:
            gapped.append(allTokensEvents[i])
        else:
            gapped.append([0,0,0,0,0])
    print(gapped)
def tokenize(data_path):
    """pitch,velocity,duration,position,bar
    try replacing  velocity and bar"""
    midi = MidiFile(data_path)
    tokens = tokenizer.midi_to_tokens(midi)
    tok_Events=tokens[0].ids
    return tok_Events
def tokenizeNovelo(data_path):
    """pitch,duration,position,bar
    try taking out velocity"""
    midi = MidiFile(data_path)
    tokens = tokenizer.midi_to_tokens(midi)
    #remove velocity
    tok_Events=tokens[0].ids
    for i in range(len(tok_Events)):
        tok_Events[i]=[tok_Events[i][0]]+tok_Events[i][2:5]
    return tok_Events
def tokenizeSOSEOS(data_path):
    """pitch,soseos,duration,position,bar
    try taking out velocity and bar"""
    midi = MidiFile(data_path)
    tokens = tokenizer.midi_to_tokens(midi)
    #remove velocity
    tok_Events=tokens[0].ids
    for i in range(0,len(tok_Events)):
        if tok_Events[i][-1]!=tok_Events[i-1][-1]:
            tok_Events[i-1][1]=2
            tok_Events[i][1]=1
        else:
            tok_Events[i][1]=0
    tok_Events[-1][1]=2
    return tok_Events
def makeBarDataset(allTokensEvents):
    """X=current bar, Y=next bar"""
    allbars=[]
    sliceidx=0
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][-1]!=allTokensEvents[i-1][-1]:
            toadd=[]
            for tok in range(sliceidx,i):
                toadd+=[allTokensEvents[tok][0]]+allTokensEvents[tok][2:4]
            allbars.append(toadd)
            # allbars.append([tok for tok in allTokensEvents[sliceidx:i]])
            sliceidx=i
    # print([(bar,0) for bar in allbars])
    XY=[[allbars[i],allbars[i+1]] for i in range(len(allbars)-1)]
    XY.append([allbars[-1], allbars[-1]]) #last has no next
    return XY
def makeBartoNoteDataset(allTokensEvents,numtok):
    """X=current bar, Y=next bar"""
    allbars=[]
    for i in range(len(allTokensEvents)-numtok):
        toadd=[]
        for tok in range(i,i+numtok):
            toadd+=allTokensEvents[tok]
        allbars.append([toadd,allTokensEvents[i+numtok]])
        # sliceidx=i
    # print([(bar,0) for bar in allbars])
    # XY=[[allbars[i],allbars[i+1]] for i in range(len(allbars)-1)]
    # XY.append([allbars[-1], allbars[-1]]) #last has no next
    return allbars

def makeNumBarDatasetPad(allTokensEvents,num_bar,idperNote):
    """X=current (num_bars-1), Y=next bar, padding included"""
    allbars=[]
    #list of bars
    sliceidx=0
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][-1]!=allTokensEvents[i-1][-1]:
            bar=[]
            for j in range(sliceidx,i):
                bar+=allTokensEvents[j]

            # bar+=[3]#bar tok
            allbars.append(bar)
            sliceidx=i
    maxlen=max([(len(bar))//idperNote for bar in allbars])
    for b in range(len(allbars)):
        barlen=(len(allbars[b]))//idperNote
        if barlen<maxlen:
            for _ in range(maxlen-barlen):
                # allbars[b][-1]=0
                allbars[b]+=[0]
    numbarSplit=[]
    sliceidx=0
    for i in range(num_bar,len(allbars)):
        x=[]
        for j in range(sliceidx,i):
            x+=allbars[j]

        numbarSplit.append([x,allbars[i]])
        sliceidx+=1
    # print([(bar,0) for bar in allbars])
    # XY=[[allbars[i],allbars[i+1]] for i in range(len(allbars)-1)]
    # XY.append([allbars[-1], allbars[-1]]) #last has no next
    return numbarSplit
def makeNumBarDatasetNo3(allTokensEvents,num_bar):
    """X=current (num_bars-1), Y=next bar, padding included"""
    allbars=[]
    #list of bars
    sliceidx=0
    allTokensEvents[0][1]=1
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][-1]!=allTokensEvents[i-1][-1]:
            bar=[]
            for j in range(sliceidx,i):
                bar+=allTokensEvents[j]
            allbars.append(bar)
            sliceidx=i
    numbarSplit=[]
    sliceidx=0
    for i in range(num_bar,len(allbars)):
        x=[]
        for j in range(sliceidx,i):
            x+=allbars[j]

        numbarSplit.append([x,allbars[i]])
        sliceidx+=1
    return numbarSplit
def makeNumBarDatasetNopad(allTokensEvents,num_bar):
    """X=current (num_bars-1), Y=next bar, padding included"""
    allbars=[]
    #list of bars
    sliceidx=0
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][-1]!=allTokensEvents[i-1][-1]:
            bar=[]
            for j in range(sliceidx,i):
                bar+=allTokensEvents[j]

            bar+=[3]#bar tok
            allbars.append(bar)
            sliceidx=i
    numbarSplit=[]
    sliceidx=0
    for i in range(num_bar,len(allbars)):
        x=[]
        for j in range(sliceidx,i):
            x+=allbars[j]

        numbarSplit.append([x,allbars[i]])
        sliceidx+=1
    # print([(bar,0) for bar in allbars])
    # XY=[[allbars[i],allbars[i+1]] for i in range(len(allbars)-1)]
    # XY.append([allbars[-1], allbars[-1]]) #last has no next
    return numbarSplit
def makeNoteDataset(allTokensEvents):
    """X=current note, Y=next note"""
    XY=[[allTokensEvents[i],allTokensEvents[i+1]] for i in range(len(allTokensEvents)-1)]
    XY.append([allTokensEvents[-1], allTokensEvents[-1]]) #last has no next so replace with same token
    return XY
def note2vecNumBarDatasetNo3(allTokensEvents,num_bar):
    """X=current (num_bars-1), Y=next bar, padding included"""
    allbars=[]
    #list of bars
    sliceidx=0
    allTokensEvents[0][1]=1
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][-1]!=allTokensEvents[i-1][-1]:
            bar=[]
            for j in range(sliceidx,i):
                bar.append(tuple(allTokensEvents[j]))
            allbars.append(bar)
            sliceidx=i
    numbarSplit=[]
    sliceidx=0
    for i in range(num_bar,len(allbars)):
        x=[]
        for j in range(sliceidx,i):
            x+=allbars[j]
        numbarSplit.append([torch.tensor(x),torch.tensor(allbars[i])])
        sliceidx+=1
    return numbarSplit
def note2vecNumBarRelative(allTokensEvents,num_bar):
    """X=current (num_bars-1), Y=next bar, padding included"""
    allbars=[]
    sliceidx=0
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][1]==1:
            bar=[]
            for j in range(sliceidx,i):
                bar.append(tuple(allTokensEvents[j]))
            allbars.append(bar)
            sliceidx=i
    numbarSplit=[]
    sliceidx=0
    for i in range(num_bar,len(allbars)):
        x=[]
        for j in range(sliceidx,i):
            x+=allbars[j]
        numbarSplit.append([torch.tensor(x),torch.tensor(allbars[i])])
        sliceidx+=1
    return numbarSplit
def newtok(data_path):
    """pitch,duration,position,bar
    try taking out velocity and bar"""
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 8),  # (half, 8 beats)
                     'nb_tempos': 32,  # nb of tempo bins
                     'tempo_range': (40, 250)}
    tokenizerconfig= TokenizerConfig(pitch_range, beat_res, nb_velocities, additional_tokens,sos_eos = True, pad = True)
    tokenizer=Octuple(tokenizerconfig)

    midi = MidiFile(data_path)
    tokens = tokenizer.midi_to_tokens(midi)
    #remove velocity
    # print(tokens)
    song_tokens = tokens
    # tok_Events=tokens[0].events
    print(tokenizer._vocab_base)
    tok_Events=tokens[0].ids
    # for i in range(len(tok_Events)):
    #     tok_Events[i]=[tok_Events[i][0]]+tok_Events[i][2:4]
    return tok_Events
def note2vecNote(allnotes, sliceamount):
    """returns x=sliceamount of notes y=next note"""
    # allnotes=[tuple(a) for a in allnotes]
    sliced=[]
    slideidx=0
    for i in range(sliceamount,len(allnotes)-sliceamount,sliceamount):
        sliced.append([allnotes[slideidx:i],allnotes[i]])
        slideidx=i
    return sliced
def DPmapping(allTokensEvents):
    """returns mapping dict {duration:pos toadd}"""
    allbars=[]
    sliceidx=0
    DP={}
    for i in range(1,len(allTokensEvents)):
        if allTokensEvents[i][1]==1:
            bar=[]
            for j in range(sliceidx,i):
                bar+=allTokensEvents[j]
            allbars.append(bar)
            sliceidx=i
    # print(allbars)
    for b in allbars:
        for n in range(2,len(b)-2,4):
            if b[n] not in DP:
                DP[b[n]]=b[n+5]-b[n+1]
    return DP

#some tests
# a=tokenizeSOSEOS('./data/testmidi.mid')
# a=[x[:-1] for x in a]
# gapping(a,[0.5,0.5])
# print(a)
# print(DPmapping(a))
# print(note2vecNumBarRelative(a,2))
# allTokensEvents=newtok('./data/testmidi.mid')
# a=tokenizeE('./data/testmidi.mid')
# novelo=tokenizeNovelo('./data/hbd.mid')
# pprint.pprint(novelo)
# pprint.pprint(tokenizeSOSEOS('./data/hbd.mid'))
# pprint.pprint(a)
# print(makeNumBarDatasetPad(a,3,5))
# print("barnoteXY")
# print(makeDict(a))
# print(makeBartoNoteDataset(a,7))
# print("barXY")
# pprint.pprint(makeBarDataset(allTokensEvents))
# print("noteXY")
# pprint.pprint(makeNoteDataset(novelo))
