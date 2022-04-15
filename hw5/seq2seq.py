"""Assignment 5: Machine translation and seq2seq modeling

Use the model to predict multiple sentences.
Extra (optional): Use BLEU to evaluate the predicted translations.
Additional resources for assignment
"""

import collections
import math
import torch
from torch import nn
from d2l import torch as d2l
from argparse import ArgumentParser
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as Parallel

from tqdm import tqdm
import functools
import os
import time


def get_args():

    ap = ArgumentParser(description='args for hw4')

    ap.add_argument('-s', '--save', action='store_true')
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('-t', '--train', action='store_true')
    ap.add_argument('-e', '--eval', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('-n', '--num_epochs', type=int)

    args = ap.parse_args()

    if not args.num_epochs:
        args.num_epochs = 5

    return args

class Environment():
    'environment variables to be passed around'

    def __init__(self):
        pass


class Seq2SeqEncoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X = self.embedding(X)
        # In RNN models, the first axis corresponds to time steps
        X = X.permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros
        output, state = self.rnn(X)
        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class Seq2SeqDecoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        # The output `X` shape: (`num_steps`, `batch_size`, `embed_size`)
        X = self.embedding(X).permute(1, 0, 2)
        # Broadcast `context` so it has the same `num_steps` as `X`
        context = state[-1].repeat(X.shape[0], 1, 1)

        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # `output` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class EncoderDecoder(nn.Module):

    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    mask = (
        torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :]
        < valid_len[:, None]
    )
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label
        )
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train(net, train_iter, tgt_vocab, optimizer, *, env):

    timer = time.perf_counter()

    num_epochs = env.args.num_epochs

    device = env.device
    net.to(device)
    net.train()

    loss = MaskedSoftmaxCELoss()
    losses = []
    for epoch in range(num_epochs):
        print(f'\nepoch: {epoch+1} of {num_epochs}')

        for batch in tqdm(train_iter):

            optimizer.zero_grad()

            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # Teacher forcing

            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()

            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            losses.append(l.sum())

            # save it ... after batch cuz it takes a while
            if env.args.save and l.mean() < min(losses):
                torch.save(net.state_dict(), env.file)

        print(f'loss: {l.sum()}')

    timer = int(time.perf_counter() - timer)
    print(f'Finished in {timer} seconds')
    print(f'loss: {losses[-1]}')

    return losses


#@save
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence."""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):  #@save
    """Compute the BLEU."""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def xavier_init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
    if type(layer) == nn.GRU:
        for param in layer._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(layer._parameters[param])


def main():

    args = get_args()

    env = Environment()
    env.file = __file__.split('/')[-1].split('.')[0] + '.pt'
    env.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env.args = args

    # hparam
    embed_size, num_hiddens, num_layers, dropout = 32, 512, 4, 0.1
    batch_size, num_steps = 64, 10


    # get training data
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    # create a model
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    if args.load:
        try:
            net.load_state_dict(torch.load(env.file))
        except:
            net.apply(xavier_init_weights)

    if args.verbose:
        print(net)

    # if torch.cuda.device_count() > 1:
    #     net = Parallel(net)

    # train
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    if args.train:
        hist = train(net, train_iter, tgt_vocab, optimizer, env=env)

    # do some predictions
    if args.eval:
        print('evaluation')

        engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .', 'my name is']
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .','je m\'appelle']
        for eng, fra in zip(engs, fras):
            translation, attention_weight_seq = predict_seq2seq(
                net, eng, src_vocab, tgt_vocab, num_steps, device)
            print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

        phrase = 'i have'
        translation, attention_weight_seq = predict_seq2seq(net, phrase, src_vocab, tgt_vocab, num_steps, device)
        print(f'{phrase} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')

if __name__ == "__main__":
    main()
