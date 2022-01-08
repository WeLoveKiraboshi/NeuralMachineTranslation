from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd
import spacy
import torchtext
import math
import pickle
import os
import random

#from torchtext.legacy import data

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
# from nltk.translate.bleu_score import sentence_bleu
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
from datetime import datetime
# from janome.tokenizer import Tokenizer
from dataloader import MyIterator
from model import Seq2Seq, Encoder, Decoder, init_weights
from metrics import wer_score, GLEU_score, TER_score, BLEU_score
import time

#Hyper parameter
BATCH_SIZE = 64
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5





if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')

global max_src_in_batch, max_tgt_in_batch

##data loading

# # Build tokenizers for Japanese and English
JA = spacy.blank('ja')
# EN = spacy.load('en')
EN = spacy.load("en_core_web_sm")

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]

JA_TEXT = Field(tokenize=tokenize_ja)
EN_TEXT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

# # Create training and validation set
def create_train_val_set(kyoto_lexicon_df):
    # train, val, test = train_val_test_split(kyoto_lexicon_df, test_size=0.3)
    train, val, test = np.split(kyoto_lexicon_df.sample(frac=1),
                                [int(.6 * len(kyoto_lexicon_df)), int(.8 * len(kyoto_lexicon_df))])
    train.to_csv('train.csv', index=False)
    val.to_csv('val.csv', index=False)
    test.to_csv('test.csv', index=False)
    data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]

    train, val, test = TabularDataset.splits(path='./',
                                             train='train.csv',
                                             validation='val.csv',
                                             test='test.csv',
                                             format='csv',
                                             fields=data_fields)

    # train, val = train_test_split(kyoto_lexicon_df, test_size=0.3)
    # train.to_csv('train.csv', index=False)
    # val.to_csv('val.csv', index=False)
    # data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]
    #
    # train, val = TabularDataset.splits(
    #     path='./',
    #     train='train.csv',
    #     validation='val.csv',
    #     format='csv',
    #     fields=data_fields
    # )
    JA_TEXT.build_vocab(train, val)
    EN_TEXT.build_vocab(train, val)
    # train_iter = BucketIterator(
    #     train,
    #     batch_size=BATCH_SIZE,
    #     sort_key=lambda x: len(x.English),
    #     shuffle=True
    # )
    # batch = next(iter(train_iter))
    return JA_TEXT, EN_TEXT, train, val, test


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.Japanese))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.English) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)





def translate_sentence(sentence,src_field,trg_field,model,device, max_len=50):
    model.eval()
    if isinstance(sentence,str):
        nlp = spacy.load('ja_core_news_sm')
        tokens = [token.text for token in nlp(sentence)]
    else:
        tokens = [token for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        context = model.encoder(src_tensor)
        hidden = context

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    batch_size = src_tensor.shape[1]
    trg_len = src_tensor.shape[0]
    for i in range(1, max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            #print('trg_tensor={}  hidden={}  cell={}'.format(trg_tensor.shape, hidden.shape, cell.shape))
            #output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            output, hidden = model.decoder(trg_tensor, hidden, context)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:-1] # remove <sos> and <eos>



def translate_v2(model, src_tensor, trg_tensor, src_field, trg_field, device, max_len=50):#trg_field=EN_TEXT
    assert isinstance(src_tensor, torch.Tensor)
    model.eval()
    #src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        print('src input = {}'.format(src_tensor.shape))
        hidden, cell = model.encoder(src_tensor)
        #enc_src = model.encoder(src_tensor, src_mask)
    #print('src tensor len = ', len(src_tensor))
    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]
    # trg_indexes = []
    # for sentence in src_tensor.transpose(0, 1): #loop for bs(64)
    #     tmp = []
    #     # Start from the first token which skips the <start> token
    #     for i in sentence[1:]:
    #         # Targets are padded. So stop appending as soon as a padding or eos token is encountered
    #         if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
    #             break
    #         tmp.append(trg_field.vocab.itos[i])
    #     trg_indexes.append([tmp])

    input = trg_tensor[0, :]
    translations_done = [0] * len(src_tensor)
    for idx in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        with torch.no_grad():
            print('trg_tensor={}  hidden={}  cell={}'.format(trg_tensor.shape, hidden.shape, cell.shape))
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            #output = model.out(model.decoder(trg_tensor, enc_src, src_mask, trg_mask))
            pred_tokens = output.argmax(2)[:, -1]
            for i, pred_token_i in enumerate(pred_tokens):
                trg_indexes[i].append(pred_token_i)
                if pred_token_i == trg_field.vocab.stoi['<eos>']: #trg_field.vocab.stoi[trg_field.eos_token]:
                    translations_done[i] = 1
            if all(translations_done):
                break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field.vocab.stoi['<eos>']:
                break
            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])
        pred_sentences.append(pred_sentence)
    return pred_sentences


from torchtext.data.metrics import bleu_score


def test_epoch(data, src_field, trg_field, model, device, max_len=50):
    trgs = []
    trgs_v2 = []
    pred_trgs = []

    start = time.time()
    for idx, datum in tqdm(enumerate(data)):
        src = vars(datum)['Japanese']
        trg = vars(datum)['English']
        pred_trg = translate_sentence(src, src_field, trg_field, model, device, max_len)
        # cut off <eos> token
        # pred_trg = pred_trg[:-1]
        #print('cand = {}   ref = {}'.format(pred_trg, trg))
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        trgs_v2.append(trg)
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time * 1000) + "[msec]")
    print('total data length : {}'.format(len(data)))
    print(f'Avg processing time per sentence = {elapsed_time * 1000 / len(data):.5f}')

    print(f'BLEU score = {BLEU_score(pred_trgs, trgs) :.4f}')
    print(f'GLEU score = {GLEU_score(pred_trgs, trgs):.4f}')
    print(f'WER score = {wer_score(pred_trgs, trgs_v2):.4f}')
    print(f'TER score = {TER_score(pred_trgs, trgs_v2):.4f}')
    return 1



def test_epoch_v2(model, test_dataloader, src_field, trg_field, device, print_every=10, mode='test'):
    trgs = []
    pred_trgs = []
    total_dataset_length = len(test_dataloader)
    start = datetime.now()
    temp = start
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            src = batch.Japanese#.transpose(0, 1) #.to(device)
            trg = batch.English#.transpose(0, 1) #.to(device)
            # trg_input = trg[:, :-1]
            # # the words we are trying to predict
            # targets = trg[:, 1:].contiguous().view(-1)
            _trgs = []
            for sentence in trg.transpose(0, 1):
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg = translate_v2(model, src, trg, src_field, trg_field, device, max_len=50)#trg_field=EN_TEXT ,translate_sentence_vectorized(src, src_field, trg_field, model, device)
            pred_trgs += pred_trg
            rand_idx = random.randint(0, len(_trgs)-1)
            print('trg = {} pred = {}'.format(_trgs[rand_idx], pred_trg[rand_idx]))
            # print('夏は終わった。', translate(model, '夏は終わった。', custom_sentence=True))
            # exit(0)
            if (idx + 1) % print_every == 0:
                print("{}: [{}/{}] time = {}, {} per {} iters".format(
                    mode,
                    (idx + 1),
                    total_dataset_length,
                    (datetime.now() - start) // 60,
                    datetime.now() - temp,
                    print_every
                ))
        score = bleu_score(pred_trgs, trgs)
        print(f'BLEU score = {score :.2f}')
        return pred_trgs, trgs, score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    mode = 'test'
    dataset_id = 'anki'  # kyoto_lexicon anki merged
    print('_____________________________________________________')
    print(dataset_id)
    print('_____________________________________________________')
    JA_TEXT_PATH = '../Voc/' + dataset_id + '_JA_TEXT.pkl'  # kyoto_lexicon_JA_TEXT.pkl
    EN_TEXT_PATH = '../Voc/' + dataset_id + '_EN_TEXT.pkl'
    train_set_path = '../Voc/' + dataset_id + '_train.csv'
    val_set_path = '../Voc/' + dataset_id + '_val.csv'
    test_set_path = '../Voc/' + dataset_id + '_test.csv'
    ckpt_save_path = 'ckpt/model-best-'+dataset_id+'.pkl'
    JA_TEXT = pickle.load(open(JA_TEXT_PATH, "rb"))
    EN_TEXT = pickle.load(open(EN_TEXT_PATH, "rb"))
    data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]
    train, val, test = TabularDataset.splits(path='./',
                                             train=train_set_path,
                                             validation=val_set_path,
                                             test=test_set_path,
                                             format='csv',
                                             fields=data_fields)
    JA_TEXT.build_vocab(train, val,min_freq = 2)
    EN_TEXT.build_vocab(train, val,min_freq = 2)


    input_pad = JA_TEXT.vocab.stoi['<pad>']
    target_pad = EN_TEXT.vocab.stoi['<pad>']
    src_vocab = len(JA_TEXT.vocab)
    trg_vocab = len(EN_TEXT.vocab)
    print('Hyperparam: src_vocab={}  trg_vocab={}'.format(src_vocab, trg_vocab))
    enc = Encoder(src_vocab, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    dec = Decoder(trg_vocab, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    model.to(device)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    #load ckpt
    model_cp = pickle.load(open( ckpt_save_path, "rb"))
    model.load_state_dict(model_cp['model'], strict=False)

    #data loader for test
    print('train = {} val = {} test = {}'.format(len(train), len(val), len(test)))
    train_dataloader, val_dataloader, test_dataloader = BucketIterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.English),
        device=device)

    ##pred_trgs, trgs, bleu_score = test_epoch_v2(model, test_dataloader, JA_TEXT, EN_TEXT, device)
    bleu_score =  test_epoch(test, JA_TEXT, EN_TEXT, model, device, max_len=50)

    # print('僕は駅に行く途中なの', translate_sentence('僕は駅に行く途中なの', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('駅に行く途中なの。', translate_sentence('駅に行く途中なの', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('私は彼が医者であるかどうかわからない。', translate_sentence('私は彼が医者であるかどうかわからない。', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('お話し中、申し訳ありません。', translate_sentence('お話し中、申し訳ありません。', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('東京オリンピック組織委員会。', translate_sentence('東京オリンピック組織委員会。', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('日本語は難しいと思います', translate_sentence('日本語は難しいと思います', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('英語は簡単です', translate_sentence('英語は簡単です', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('京都は東京の街です。', translate_sentence('京都は東京の街です。', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('京都', translate_sentence('京都', JA_TEXT, EN_TEXT, model, device, max_len=50))
    # print('寺', translate_sentence('寺', JA_TEXT, EN_TEXT, model, device, max_len=50))



    #from torchsummary import summary
    # torch.Size([433, 1]) torch.Size([433, 2]) torch.Size([433, 1, 1]) torch.Size([433, 2, 2])
    #summary(model, (3, 240, 320))

















#

