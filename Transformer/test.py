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
from torchmetrics import WER
from tqdm import tqdm
from datetime import datetime
# from janome.tokenizer import Tokenizer
from dataloader import MyIterator
from model_v2 import Transformer
from metrics import wer_score, GLEU_score, TER_score, BLEU_score, BLEU_score
import time

#Hyper parameter
KYOTO_LEXICON_PATH = '/home/yukisaito/NLP_project/dataset/japaneseenglish-bilingual-corpus/kyoto_lexicon.csv'
BATCH_SIZE = 256
EPOCHS = 3
D_MODEL = 512
HEADS = 8
N=6




if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')

global max_src_in_batch, max_tgt_in_batch

##data loading
kyoto_lexicon_df = pd.read_csv(KYOTO_LEXICON_PATH, error_bad_lines=False)
kyoto_lexicon_df.columns
kyoto_lexicon_df.head(10)
len(kyoto_lexicon_df['日本語'])
len(kyoto_lexicon_df['英語'])

kyoto_lexicon_df = kyoto_lexicon_df[['日本語', '英語']]
kyoto_lexicon_df.columns = ['Japanese', 'English']
kyoto_lexicon_df.dropna(inplace=True)
kyoto_lexicon_df['Japanese'].iloc[123]

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




def create_masks_input(input_seq):
    input_pad = JA_TEXT.vocab.stoi['<pad>']
    # creates mask with 0s wherever there is padding in the input
    input_msk = (input_seq != input_pad).unsqueeze(1).to(device)
    return input_msk

def create_masks_target(target_seq):
    target_pad = EN_TEXT.vocab.stoi['<pad>']
    target_msk = (target_seq != target_pad).unsqueeze(1).to(device)
    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
    target_msk = target_msk & nopeak_mask
    return target_msk




def translate(model, src, max_len=80, custom_sentence=False, return_list=False):
    print(f'SRC={src}')
    model.eval()
    if custom_sentence == True:
        src = tokenize_ja(src)
        src = torch.autograd.Variable(torch.LongTensor([[JA_TEXT.vocab.stoi[tok] for tok in src]])).to(device)

    src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([EN_TEXT.vocab.stoi['<sos>']])

    for i in range(1, max_len):
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).to(device)

        out_src, attention = model.decoder(
                outputs[:i].unsqueeze(0),
                e_outputs,
                src_mask,
                trg_mask
            )
        out = model.out(out_src)
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == EN_TEXT.vocab.stoi['<eos>']:
            break
    if return_list:
        return [EN_TEXT.vocab.itos[ix] for ix in outputs[:i]], attention
    else:
        return ' '.join([EN_TEXT.vocab.itos[ix] for ix in outputs[:i]]), attention


def translate_v2(model, src_tensor, src_field, trg_field, device, max_len=50):#trg_field=EN_TEXT
    assert isinstance(src_tensor, torch.Tensor)
    model.eval()
    #src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        src_mask = create_masks_input(src_tensor) #(src_tensor != input_pad).unsqueeze(-2)
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]
    translations_done = [0] * len(src_tensor)
    for idx in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = create_masks_target(trg_tensor)
        with torch.no_grad():
            out_src, attention = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
            output = model.out(out_src)
            # print(output.shape)
            # print(output)
            # exit(0)
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


def test_epoch(model, test_set_path):
    test_df = pd.read_csv(test_set_path)
    print(test_df)
    import warnings
    warnings.filterwarnings('ignore')

    data_length = len(test_df["English"]);
    trgs = []
    pred_trgs = []

    start = time.time()
    for i in tqdm(range(data_length)):
        reference = tokenize_en(test_df["English"][i])
        #print(test_df["Japanese"][i])
        translation, attention = translate(model, test_df["Japanese"][i], custom_sentence=True)
        candidate = tokenize_en(translation)
        candidate = candidate[3:]
        #print('re={}  can={}'.format(reference, candidate))
        pred_trgs.append(candidate)
        trgs.append([reference])
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time * 1000) + "[msec]")
    print('total data length : {}'.format(data_length))
    print(f'Avg processing time per sentence = {elapsed_time * 1000 / data_length:.5f}')

    # print(f'BLEU score = {(total_score / data_length) * 100}')
    score = bleu_score(pred_trgs, trgs)
    print(f'BLEU score = {score * 100:.2f}')
    return score


def test_epoch_v2(model, test_dataloader, src_field, trg_field, device, print_every=10, mode='test'):
    trgs = []
    trgs_v2 = []
    pred_trgs = []
    total_dataset_length = len(test_dataloader)
    start = datetime.now()
    temp = start
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            src = batch.Japanese.transpose(0, 1) #.to(device)
            trg = batch.English.transpose(0, 1) #.to(device)
            # trg_input = trg[:, :-1]
            # # the words we are trying to predict
            # targets = trg[:, 1:].contiguous().view(-1)
            _trgs = []
            _trgs_v2 = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if i == trg_field.vocab.stoi[trg_field.eos_token] or i == trg_field.vocab.stoi[trg_field.pad_token]:
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
                _trgs_v2.append(tmp)
            trgs += _trgs
            trgs_v2 += _trgs_v2
            pred_trg =  translate_v2(model, src, src_field, trg_field, device, max_len=50)#trg_field=EN_TEXT ,translate_sentence_vectorized(src, src_field, trg_field, model, device)
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
        print(f'BLEU score (library) = {bleu_score(pred_trgs, trgs):.4f}')
        print(f'BLEU score (metrics.py)= {BLEU_score(pred_trgs, trgs) :.4f}')
        print(f'GLEU score = {GLEU_score(pred_trgs, trgs):.4f}')
        print(f'WER score = {wer_score(pred_trgs, trgs_v2):.4f}')
        print(f'TER score = {TER_score(pred_trgs, trgs_v2):.4f}')
        return pred_trgs, trgs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    #src_mask = model.make_src_mask(src_tensor)
    src_mask = (src_tensor != input_pad).unsqueeze(-2)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    #trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    outputs = torch.zeros(max_len).type_as(src_tensor.data)
    outputs[0] = torch.LongTensor([EN_TEXT.vocab.stoi['<sos>']])

    for i in range(max_len):
        #trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        #trg_mask = model.make_trg_mask(trg_tensor)
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).to(device)

        with torch.no_grad():
            output, attention = model.decoder(outputs[:i].unsqueeze(0), enc_src, src_mask, trg_mask)
            output = model.out(output)


        output = F.softmax(output, dim=-1)
        val, ix = output[:, -1].data.topk(1)
        outputs[i] = ix[0][0]
        if ix[0][0] == EN_TEXT.vocab.stoi['<eos>']:
            break
    return ' '.join([EN_TEXT.vocab.itos[ix] for ix in outputs[:i]]), attention



    #     pred_token = output.argmax(2)[:, -1].item()
    #
    #     trg_indexes.append(pred_token)
    #
    #     if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
    #         break
    #
    # trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    #
    # return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    import matplotlib.ticker as ticker

    assert n_rows * n_cols == n_heads
    fig = plt.figure(figsize=(15, 25))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        # ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
        #                        rotation=45)
        ax.set_xticklabels([''] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
        ax.set_yticklabels([''] + translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()
    


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    mode = 'test'
    dataset_id = 'anki'  # kyoto_lexicon anki merged
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
    JA_TEXT.build_vocab(train, val)
    EN_TEXT.build_vocab(train, val)


    input_pad = JA_TEXT.vocab.stoi['<pad>']
    target_pad = EN_TEXT.vocab.stoi['<pad>']
    src_vocab = len(JA_TEXT.vocab)
    trg_vocab = len(EN_TEXT.vocab)

    print('Hyperparam: src_vocab={}  trg_vocab={} D_MODEL={}  N={}  HEADS={}'.format(src_vocab, trg_vocab, D_MODEL, N, HEADS))
    model = Transformer(src_vocab, trg_vocab, D_MODEL, N, HEADS)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    #load ckpt
    model_cp = pickle.load(open( ckpt_save_path, "rb"))
    model.load_state_dict(model_cp['model'], strict=False)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    #data loader for test
    train_dataloader, val_dataloader, test_dataloader = BucketIterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.English),
        device=device)

    # for idx in range(10):
    #     example_idx = random.randint(0, len(train))
    #     src = vars(train.examples[example_idx])['Japanese']
    #     trg = vars(train.examples[example_idx])['English']
    #     print(f'src = {src}')
    #     print(f'trg = {trg}')
    #
    #     translation, attention = translate(model, ''.join(src), custom_sentence=True, return_list=True) #translate_sentence(src, JA_TEXT, EN_TEXT, model, device)
    #     translation = translation[1:]
    #     print(f'predicted trg = {translation}')
    #     print(f'attention = {attention.shape}')
    #     display_attention(src, translation, attention)

    #pred_trgs, trgs = test_epoch_v2(model, test_dataloader, JA_TEXT, EN_TEXT, device)
    #score = test_epoch(model, test_set_path)


    # print('僕は駅に行く途中なの', translate(model, '僕は駅に行く途中なの', custom_sentence=True))
    # print('駅に行く途中なの。', translate(model, '駅に行く途中なの。', custom_sentence=True) )
    # print('私は彼が医者であるかどうかわからない。', translate(model, '私は彼が医者であるかどうかわからない。', custom_sentence=True) )
    # print('お話し中、申し訳ありません。', translate(model, 'お話し中、申し訳ありません。', custom_sentence=True) )
    # print('東京オリンピック組織委員会。', translate(model, '東京オリンピック組織委員会。', custom_sentence=True))
    # print('日本語は難しいと思います', translate(model, '日本語は難しいと思います', custom_sentence=True) )
    # print('英語は簡単です', translate(model, '英語は簡単です' , custom_sentence=True))
    # print('京都は東京の街です。', translate(model, '京都は東京の街です。', custom_sentence=True))
    # print('京都', translate(model, '京都', custom_sentence=True))
    # print('寺', translate(model, '寺', custom_sentence=True))
    print(translate(model, '自分の歳より半分＋７歳以下の人とは絶対にお付き合いしない方がいいって、聞いたことがあります。トムは今、３０歳でメアリーは１７歳です。トムがメアリーと付き合えるようになるまで、トムはあと何年待たないといけないでしょう。', custom_sentence=True)[0])



    #from torchsummary import summary
    # torch.Size([433, 1]) torch.Size([433, 2]) torch.Size([433, 1, 1]) torch.Size([433, 2, 2])
    #summary(model, (3, 240, 320))




















#

