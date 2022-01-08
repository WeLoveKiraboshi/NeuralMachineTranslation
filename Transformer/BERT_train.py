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

#from torchtext.legacy import data

from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from datetime import datetime
# from janome.tokenizer import Tokenizer
from dataloader import MyIterator
from model import Transformer



#Hyper parameter
KYOTO_LEXICON_PATH = '~/NLP_project/dataset/japaneseenglish-bilingual-corpus/kyoto_lexicon.csv'
ANKI_LEXICON_PATH = '~/NLP_project/dataset/jpn-eng/eng-jpn.txt'
BATCH_SIZE = 64
EPOCHS = 100
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
anki_dataset_df = pd.read_csv(ANKI_LEXICON_PATH,sep='\t',names=['English','Japanese'])
anki_dataset_df = anki_dataset_df.reindex(columns=['Japanese', 'English'])
# print(anki_dataset_df.columns)
# print(anki_dataset_df.head(10))
# print(anki_dataset_df.tail(10))


kyoto_lexicon_df = pd.read_csv(KYOTO_LEXICON_PATH, error_bad_lines=False)
kyoto_lexicon_df = kyoto_lexicon_df[['日本語', '英語']]
kyoto_lexicon_df.columns = ['Japanese', 'English']
kyoto_lexicon_df.dropna(inplace=True)
# print(kyoto_lexicon_df.columns)
# print(kyoto_lexicon_df.head(10))
# print(kyoto_lexicon_df.tail(10))


# # Build tokenizers for Japanese and English
JA = spacy.blank('ja')
EN = spacy.load("en_core_web_sm")

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]

JA_TEXT = Field(tokenize=tokenize_ja)
EN_TEXT = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

frames = [kyoto_lexicon_df, anki_dataset_df]
merged_dataset_df = pd.concat(frames)
print(merged_dataset_df.columns)
print(merged_dataset_df.head(10))
print(merged_dataset_df.tail(10))

# # Create training and validation set



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




def create_masks(input_seq, target_seq):
    input_pad = JA_TEXT.vocab.stoi['<pad>']
    # creates mask with 0s wherever there is padding in the input
    input_msk = (input_seq != input_pad).unsqueeze(1).to(device)
    
    target_pad = EN_TEXT.vocab.stoi['<pad>']
    target_msk = (target_seq != target_pad).unsqueeze(1).to(device)
    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
    target_msk = target_msk & nopeak_mask
    
    return input_msk, target_msk






best_avg_error = None


def train_epoch(mode, dataset_id, train_dataloader, model, epoch, tb_logger, print_every=50):
    global best_avg_error
    model.train()
    start = datetime.now()
    temp = start
    total_loss = 0
    total_dataset_length = len(train_dataloader)
    for i, batch in enumerate(train_dataloader):
        src = batch.Japanese.transpose(0, 1).to(device)
        trg = batch.English.transpose(0, 1).to(device)
        # the French sentence we input has all words except
        # the last, as it is using each word to predict the next
        trg_input = trg[:, :-1]
        # the words we are trying to predict
        targets = trg[:, 1:].contiguous().view(-1)

        # create function to make masks using mask code above
        src_mask, trg_mask = create_masks(src, trg_input)
        # print("tensor device: src_mask={} trg_mask={} src={} trg_input={}".format(src_mask.device, trg_mask.device, src.device, trg_input.device))
        # exit(0)
        preds = model(src, trg_input, src_mask, trg_mask)

        optim.zero_grad()
        loss = F.cross_entropy(
            preds.view(-1, preds.size(-1)),
            targets,
            ignore_index=target_pad
        )
        loss.backward()
        optim.step()


        total_loss += loss.item()
        Loss_hist = []
        Perplexity_hist = []
        if (i + 1) % print_every == 0:
            loss_avg = total_loss / print_every
            perplexity = math.exp(loss_avg)

            print("{}: [{}/{}] time = {}, epoch {}, loss = {}, perplexity = {}, {} per {} iters".format(
                mode,
                (i + 1),
                total_dataset_length,
                (datetime.now() - start) // 60,
                epoch + 1,
                loss_avg,
                perplexity,
                datetime.now() - temp,
                print_every
            ))
            if i + 1 == 450:
                Loss_hist.append(loss_avg)
                Perplexity_hist.append(perplexity)
            total_loss = 0
            temp = datetime.now()

    loss_avg = total_loss / print_every
    perplexity = math.exp(loss_avg)
    if mode == 'val':  # check the best model for every val epochs
        if best_avg_error is None:
            best_avg_error = loss_avg
            print('Best Avg error in validation: %f, saving best checkpoint epoch %d, iter %d' % (
                best_avg_error, epoch, i + 1))
            model_cp = {'model': model.state_dict()}
            pickle.dump(model_cp, open('./ckpt/model-best-'+dataset_id+'.pkl', 'wb'))
        else:
            if loss_avg < best_avg_error:
                best_avg_error = loss_avg
                print('Best Avg error in validation: %f, saving the best checkpoint, epoch %d, iter %d' % (
                    best_avg_error, epoch, i + 1))
                model_cp = {'model': model.state_dict()}
                pickle.dump(model_cp, open('./ckpt/model-best-'+dataset_id+'.pkl', 'wb'))
    ##register history loss
    tb_logger.scalar_summary('CrossEntropy_' + mode, loss_avg, epoch)
    tb_logger.scalar_summary('CrossEntropyExp_' + mode, perplexity, epoch)
    print()



if __name__ == '__main__':
    mode = 'train'
    use_fixed_dataset = True
    dataset_id = 'anki'  # kyoto_lexicon anki merged
    JA_TEXT_PATH = '../Voc/' + dataset_id + '_JA_TEXT.pkl'  # kyoto_lexicon_JA_TEXT.pkl
    EN_TEXT_PATH = '../Voc/' + dataset_id + '_EN_TEXT.pkl'
    train_set_path = '../Voc/' + dataset_id + '_train.csv'
    val_set_path = '../Voc/' + dataset_id + '_val.csv'
    test_set_path = '../Voc/' + dataset_id + '_test.csv'

    if use_fixed_dataset:
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
        #JA_TEXT_vocab, EN_TEXT_vocab, train, val, test = create_train_val_set(kyoto_lexicon_df, True)
    else:
        train, val, test = np.split(anki_dataset_df.sample(frac=1),
                                    [int(.6 * len(anki_dataset_df)), int(.8 * len(anki_dataset_df))])
        train.to_csv(train_set_path, index=False)
        val.to_csv(val_set_path, index=False)
        test.to_csv(test_set_path, index=False)
        data_fields = [('Japanese', JA_TEXT), ('English', EN_TEXT)]
        train, val, test = TabularDataset.splits(path='./',
                                                 train=train_set_path,
                                                 validation=val_set_path,
                                                 test=test_set_path,
                                                 format='csv',
                                                 fields=data_fields)
        JA_TEXT.build_vocab(train, val)
        EN_TEXT.build_vocab(train, val)
        pickle.dump(JA_TEXT, open(JA_TEXT_PATH, 'wb'))
        pickle.dump(EN_TEXT, open(EN_TEXT_PATH, 'wb'))

    input_pad = JA_TEXT.vocab.stoi['<pad>']
    target_pad = EN_TEXT.vocab.stoi['<pad>']
    src_vocab = len(JA_TEXT.vocab)
    trg_vocab = len(EN_TEXT.vocab)
    print('Hyperparam: src_vocab={}  trg_vocab={} D_MODEL={}  N={}  HEADS={}'.format(src_vocab, trg_vocab, D_MODEL, N, HEADS))
    model = Transformer(src_vocab, trg_vocab, D_MODEL, N, HEADS)
    model.to(device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # train_dataloader = MyIterator(
    #     train,
    #     batch_size=BATCH_SIZE,
    #     device=0,
    #     repeat=False,
    #     sort_key=lambda x: (len(x.Japanese), len(x.English)),
    #     batch_size_fn=batch_size_fn,
    #     train=True,
    #     shuffle=True
    # )
    # val_dataloader = MyIterator(
    #     val,
    #     batch_size=BATCH_SIZE,
    #     device=0,
    #     repeat=False,
    #     sort_key=lambda x: (len(x.Japanese), len(x.English)),
    #     batch_size_fn=batch_size_fn,
    #     train=True,
    #     shuffle=True
    # )
    train_dataloader, val_dataloader, test_dataloader = BucketIterator.splits(
        (train, val, test),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.English),
        device=device)

    from tb_logger import Logger
    from utils import recreate_dirs

    recreate_dirs('./tb/')
    tb_logger = Logger(log_dir='./tb/', network='BERT', name=None)

    print('train = {} val = {} test = {}'.format(len(train), len(val), len(test)))
    for epoch in range(EPOCHS):
        train_epoch('train', dataset_id, train_dataloader, model, epoch, tb_logger)
        train_epoch('val', dataset_id, val_dataloader, model, epoch, tb_logger)













#

