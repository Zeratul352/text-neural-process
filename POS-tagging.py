#load_ext autoreload
#autoreload 2

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report

import numpy as np

import pyconll
import csv
import torch
import copy
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import dlnlputils
from dlnlputils.data import tokenize_corpus, build_vocabulary, \
    character_tokenize, pos_corpus_to_tensor, POSTagger
from dlnlputils.pipeline import train_eval_loop, predict_with_model, init_random_seed

init_random_seed()


import wget

#wget.download('https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-train-a.conllu','datasets/ru_syntagrus-ud-train.conllu')

#wget.download('https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/master/ru_syntagrus-ud-dev.conllu','datasets/ru_syntagrus-ud-dev.conllu')

full_train = pyconll.load_from_file('./datasets/ru_syntagrus-ud-train.conllu')
full_test = pyconll.load_from_file('./datasets/ru_syntagrus-ud-dev.conllu')

for sent in full_train[:2]:
    for token in sent:
        print(token.form, token.upos)
    print()
MAX_SENT_LEN = max(len(sent) for sent in full_train)
MAX_ORIG_TOKEN_LEN = max(len(token.form) for sent in full_train for token in sent)
print('Наибольшая длина предложения', MAX_SENT_LEN)
print('Наибольшая длина токена', MAX_ORIG_TOKEN_LEN)
all_train_texts = [' '.join(token.form for token in sent) for sent in full_train]
print('\n'.join(all_train_texts[:10]))
train_char_tokenized = tokenize_corpus(all_train_texts, tokenizer=character_tokenize)
char_vocab, word_doc_freq = build_vocabulary(train_char_tokenized, max_doc_freq=1.0, min_count=5, pad_word='<PAD>')
print("Количество уникальных символов", len(char_vocab))
print(list(char_vocab.items())[:10])
UNIQUE_TAGS = ['<NOTAG>'] + sorted({token.upos for sent in full_train for token in sent if token.upos})
label2id = {label: i for i, label in enumerate(UNIQUE_TAGS)}
#label2id
train_inputs, train_labels = pos_corpus_to_tensor(full_train, char_vocab, label2id, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)
train_dataset = TensorDataset(train_inputs, train_labels)

test_inputs, test_labels = pos_corpus_to_tensor(full_test, char_vocab, label2id, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)
test_dataset = TensorDataset(test_inputs, test_labels)
#train_inputs[1][:5]
#train_labels[1]


class StackedConv1d(nn.Module):
    def __init__(self, features_num, layers_n=1, kernel_size=3, conv_layer=nn.Conv1d, dropout=0.0):
        super().__init__()
        layers = []
        for _ in range(layers_n):
            layers.append(nn.Sequential(
                conv_layer(features_num, features_num, kernel_size, padding=kernel_size // 2),
                nn.Dropout(dropout),
                nn.LeakyReLU()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """x - BatchSize x FeaturesNum x SequenceLen"""
        for layer in self.layers:
            x = x + layer(x)
        return x


class SingleTokenPOSTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size=32, **kwargs):
        super().__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.backbone = StackedConv1d(embedding_size, **kwargs)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Linear(embedding_size, labels_num)
        self.labels_num = labels_num

    def forward(self, tokens):
        """tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen

        features = self.backbone(char_embeddings)

        global_features = self.global_pooling(features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        logits_flat = self.out(global_features)  # BatchSize*MaxSentenceLen x LabelsNum
        logits = logits_flat.view(batch_size, max_sent_len, self.labels_num)  # BatchSize x MaxSentenceLen x LabelsNum
        logits = logits.permute(0, 2, 1)  # BatchSize x LabelsNum x MaxSentenceLen
        return logits




single_token_model = SingleTokenPOSTagger(len(char_vocab), len(label2id), embedding_size=64, layers_n=3, kernel_size=3, dropout=0.3)


train_parts = []
with open('./datasets/manifest.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            train_parts.append(row[1])
            line_count += 1



print('Количество параметров', sum(np.product(t.shape) for t in single_token_model.parameters()))

best_loss = float('inf')

best_model = copy.deepcopy(single_token_model)

for dataset_part in train_parts:
    part_train = pyconll.load_from_file(dataset_part)

    #MAX_PART_SENT_LEN = max(max(len(sent) for sent in part_train), max(len(sent_t) for sent_t in full_test))
    #MAX_PART_ORIG_TOKEN_LEN = max(max(len(token.form) for sent in part_train for token in sent),
    #                              max(len(token_t.form) for sent_t in full_test for token_t in sent_t))
    #MAX_PART_ORIG_TOKEN_LEN = max(len(token.form) for sent in part_train for token in sent)
    print('Наибольшая длина предложения в части', MAX_SENT_LEN)
    print('Наибольшая длина токена в части', MAX_ORIG_TOKEN_LEN)
    all_train_texts = [' '.join(token.form for token in sent) for sent in part_train]
    print('\n'.join(all_train_texts[:10]))
    train_char_tokenized = tokenize_corpus(all_train_texts, tokenizer=character_tokenize)
    char_vocab, word_doc_freq = build_vocabulary(train_char_tokenized, max_doc_freq=1.0, min_count=5, pad_word='<PAD>')
    print("Количество уникальных символов", len(char_vocab))
    print(list(char_vocab.items())[:10])
    UNIQUE_TAGS = ['<NOTAG>'] + sorted({token.upos for sent in part_train for token in sent if token.upos})
    label2id = {label: i for i, label in enumerate(UNIQUE_TAGS)}
    # label2id
    train_inputs, train_labels = pos_corpus_to_tensor(part_train, char_vocab, label2id, MAX_SENT_LEN,
                                                      MAX_ORIG_TOKEN_LEN)
    train_dataset = TensorDataset(train_inputs, train_labels)

    test_inputs, test_labels = pos_corpus_to_tensor(full_test, char_vocab, label2id, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)
    test_dataset = TensorDataset(test_inputs, test_labels)

    (current_val_loss,
     current_single_token_model) = train_eval_loop(best_model,
                                                train_dataset,
                                                test_dataset,
                                                F.cross_entropy,
                                                lr=5e-3,
                                                epoch_n=10,
                                                batch_size=64,
                                                device='cuda',
                                                early_stopping_patience=5,
                                                max_batches_per_epoch_train=500,
                                                max_batches_per_epoch_val=100,
                                                lr_scheduler_ctor=lambda
                                                    optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2,
                                                                                                      factor=0.5,
                                                                                                      verbose=True))
    if current_val_loss < best_loss:
        best_loss = current_val_loss
        best_model = copy.deepcopy(current_single_token_model)





torch.save(best_model.state_dict(), './models/single_token_pos.pth')


#sentence_level_model.load_state_dict(torch.load('./models/sentence_level_pos.pth'))

single_token_model.load_state_dict(torch.load('./models/single_token_pos.pth'))
train_pred = predict_with_model(single_token_model, train_dataset)
train_loss = F.cross_entropy(torch.tensor(train_pred),
                             torch.tensor(train_labels))
print('Среднее значение функции потерь на обучении', float(train_loss))
print(classification_report(train_labels.view(-1), train_pred.argmax(1).reshape(-1), target_names=UNIQUE_TAGS))
print()

test_pred = predict_with_model(single_token_model, test_dataset)
test_loss = F.cross_entropy(torch.tensor(test_pred),
                            torch.tensor(test_labels))
print('Среднее значение функции потерь на валидации', float(test_loss))
print(classification_report(test_labels.view(-1), test_pred.argmax(1).reshape(-1), target_names=UNIQUE_TAGS))

single_token_pos_tagger = POSTagger(single_token_model, char_vocab, UNIQUE_TAGS, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)

test_sentences = [
    'Мама мыла раму.',
    'Косил косой косой косой.',
    'Глокая куздра штеко будланула бокра и куздрячит бокрёнка.',
    'Сяпала Калуша с Калушатами по напушке.',
    'Пирожки поставлены в печь, мама любит печь.',
    'Ведро дало течь, вода стала течь.',
    'Три да три, будет дырка.',
    'Три да три, будет шесть.',
    'Сорок сорок'
]
test_sentences_tokenized = tokenize_corpus(test_sentences, min_token_size=1)

for sent_tokens, sent_tags in zip(test_sentences_tokenized, single_token_pos_tagger(test_sentences)):
    print(' '.join('{}-{}'.format(tok, tag) for tok, tag in zip(sent_tokens, sent_tags)))
    print()


class SentenceLevelPOSTagger(nn.Module):
    def __init__(self, vocab_size, labels_num, embedding_size=32, single_backbone_kwargs={},
                 context_backbone_kwargs={}):
        super().__init__()
        self.embedding_size = embedding_size
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
        self.global_pooling = nn.AdaptiveMaxPool1d(1)
        self.out = nn.Conv1d(embedding_size, labels_num, 1)
        self.labels_num = labels_num

    def forward(self, tokens):
        """tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        batch_size, max_sent_len, max_token_len = tokens.shape
        tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)

        char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        char_features = self.single_token_backbone(char_embeddings)

        token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        token_features = token_features_flat.view(batch_size, max_sent_len,
                                                  self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize
        token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxSentenceLen
        context_features = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen

        logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen
        return logits