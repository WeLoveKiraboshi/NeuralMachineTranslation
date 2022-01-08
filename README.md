# Neural Machine Translation System Demo for Japanese-English translation.

#Abstract
Machine Translation is an indispensable task in Natural Language Processing (NLP) and applied to various application around us such as Google Translation [1], DeepL[2], and Microsoft Translator[3]. In recent years, approaches using Deep Neural Networks (DNNs) has emerged and got massive attention from both academia and industry.
In this repository, I conducted surveys about the neural machine translation system and evaluated several approaches. Especially, I focused on the translation from Japanese to English sentences. For experiments, I implemented each method from scratch and conducted extensive evaluation using 2 public available datasets. 



#1. Dataset structure and how to use
In our scripts, there are 4 representative approaches. 1. LSTM-based Seq2Seq 2. GRU-based Seq2Seq 3.CNN-based Seq2Seq 4.Transformer.

We evaluate these models with 2 public dataset: [Kyoto Lexican Dataset](https://www.kaggle.com/team-ai/japaneseenglish-bilingual-corpus/) [4] and [Anki Dataset](https://www.manythings.org/anki/) [5]. 
To use our test/train split, please use csv file under Voc/ folder.


#2. Setup

Create Anaconda or minconda virtual environment. We tested our system with python=3.8 and Pytorch1.10.1 with CUDA 11.3 library, and other related libaries (torchtext spacy):
(For further information about PyTorch, see installation instructions on the PyTorch website.)
```
conda create -n nlp python=3.8
conda activate nlp
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -U pip setuptools wheel
pip install -U spacy[cuda113,transformers,lookups]
pip install torchtext
```

We'll also make use of spaCy to tokenize our data. To install spaCy, follow the instructions here making sure to install both the English and Japanese models with:
```
python -m spacy download en_core_web_sm
python -m spacy download ja_core_web_sm
```

#3. Training the model
To train the model, run the script file below.
```
python XXX/train.py 
```
To test the model, run the script file below.
```
python XXX/test.py 
```

# 4. Demo Results

#### Table1: Quantitative result of Kyoto Lexican Dataset.

|Method|BLEU↑|GLEU↑|WER↓|TER↓|
|:---|:---|:---|:---|:---|
|Seq2Seq (LSTM-based) [10]|0.001|0.030|6.010|2.580|
|Seq2Seq (GRU-based) [11]|0.005|0.051|5.766|2.542|
|Seq2Seq (CNN-based) [12]|0.003|0.024|19.81|8.349|
|Transformer (Attention-based) [13]|0.030|0.138|7.435|3.335|


#### Table2: Quantitative result of Anki Dataset.

|Method|BLEU↑|GLEU↑|WER↓|TER↓|
|:---|:---|:---|:---|:---|
|Seq2Seq (LSTM-based) [10]|0.099|0.159|7.280|1.060|
|Seq2Seq (GRU-based) [11]|0.103|0.233|6.074|0.878|
|Seq2Seq (CNN-based) [12]|0.124|0.213|12.35|1.874|
|Transformer (Attention-based) [13]|0.335|0.413|4.200|0.570|







#5. Reference
[1] [Google Translation (https://translate.google.co.jp/?hl=ja)](https://translate.google.co.jp/?hl=ja)

[2] [DeepL (https://www.deepl.com/translator)](https://www.deepl.com/translator)

[3] [Microsoft Translator (https://www.microsoft.com/ja-jp/translator/)](https://www.microsoft.com/ja-jp/translator/)

[4] [Kyoto Lexican Dataset from NICT (https://www.kaggle.com/team-ai/japaneseenglish-bilingual-corpus/)](https://www.kaggle.com/team-ai/japaneseenglish-bilingual-corpus/)

[5] [Anki Dataset from Tatoeba org(https://www.manythings.org/anki/)](https://www.manythings.org/anki/)

[6] [Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics (pp. 311-318).](https://aclanthology.org/P02-1040.pdf)

[7] [Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Dean, J. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.](https://arxiv.org/pdf/1609.08144.pdf)

[8] [Hunt, M. J. (1990). Figures of merit for assessing connected-word recognisers. Speech Communication, 9(4), 329-336.](https://www.sciencedirect.com/science/article/abs/pii/016763939090008W)

[9] [Snover, M., Dorr, B., Schwartz, R., Micciulla, L., & Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. In Proceedings of the 7th Conference of the Association for Machine Translation in the Americas: Technical Papers (pp. 223-231).](https://www.cs.umd.edu/~snover/pub/amta06/ter_amta.pdf)

[10] [Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).](https://arxiv.org/pdf/1409.3215.pdf)

[11] [Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.](https://arxiv.org/pdf/1406.1078.pdf)

[12] [Gehring, J., Auli, M., Grangier, D., Yarats, D., & Dauphin, Y. N. (2017, July). Convolutional sequence to sequence learning. In International Conference on Machine Learning (pp. 1243-1252). PMLR.](https://arxiv.org/pdf/1705.03122.pdf)

[13] [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).](https://arxiv.org/pdf/1706.03762.pdf)

[14] [Github Pytorch Seq2Seq Tutorial (https://github.com/bentrevett/pytorch-seq2seq)](https://github.com/bentrevett/pytorch-seq2seq)
