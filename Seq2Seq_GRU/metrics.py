import numpy as np
import os
import shutil
from os import path
from os import listdir

import nltk
import nltk.translate.gleu_score as gleu

import numpy
import os

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')

from ter_src import ter
import nltk.translate.bleu_score as bleu
import math



def wer_score(hyps, refs, print_matrix=False):
    wer_list = []
    total_length = len(hyps)
    for i in range(total_length):
        hyp = hyps[i]
        ref = refs[i]
        #print('hyp={}  ref={}'.format(hyp, ref))
        N = len(hyp)
        M = len(ref)
        L = np.zeros((N, M))
        for i in range(0, N):
            for j in range(0, M):
                if min(i, j) == 0:
                    L[i, j] = max(i, j)
                else:
                    deletion = L[i - 1, j] + 1
                    insertion = L[i, j - 1] + 1
                    sub = 1 if hyp[i] != ref[j] else 0
                    substitution = L[i - 1, j - 1] + sub
                    L[i, j] = min(deletion, min(insertion, substitution))
                    # print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
        #print('hyp = {}  ref = {}'.format(hyp, ref))
        #print('L = {}  N = {}   M = {}'.format(L.shape, N, M))
        #print(L)
        if N == 0 or M == 0:
            pass
        else:
            wer_list.append(int(L[N-1, M-1]))
    if print_matrix:
        print("WER matrix ({}x{}): ".format(N, M))
        print(L)
    return np.array(wer_list).mean()



def GLEU_score(hyps, refs):
    total_length = len(hyps)
    gleu_score_list_4 = []
    gleu_score_list_2 = []
    for i in range(total_length):
        hyp = hyps[i]
        ref_b = refs[i]
        score_1to4grams = gleu.sentence_gleu(ref_b, hyp, min_len=1, max_len=4)
        score_1to2grams = gleu.sentence_gleu(ref_b, hyp, min_len=1, max_len=2)
        #print("1 to 4 grams: {}".format(score_1to4grams))
        #print("1 to 2 grams: {}".format(score_1to2grams))
        gleu_score_list_4.append(score_1to4grams)
        gleu_score_list_2.append(score_1to2grams)
    return np.array(gleu_score_list_4).mean()



def TER_score(hyps, refs):
    total_length = len(hyps)
    ter_list = []
    for i in range(total_length):
        hyp = hyps[i]
        ref = refs[i]
        N = len(hyp)
        M = len(ref)
        if N == 0 or M == 0:
            pass
        else:
            ter_list.append(ter(hyp, ref))
    return np.array(ter_list).mean()


def BLEU_score(hyps, refs):
    total_length = len(hyps)
    bleu_list = []
    for i in range(total_length):
        hyp = hyps[i]
        ref = refs[i]
        N = len(hyp)
        M = len(ref)
        if N == 0 or M == 0:
            pass
        else:
            bleu_list.append(bleu.sentence_bleu(ref, hyp))
    return np.array(bleu_list).mean()