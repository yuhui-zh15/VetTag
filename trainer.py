import os
import sys
import json
import argparse
import numpy as np
import random
import torch
from data import get_data, pad_batch, batchify, Batch
from transformer import NoamOpt, make_transformer_model, make_lstm_model, make_caml_model
import logging
from sklearn import metrics


parser = argparse.ArgumentParser(description='Clinical Dataset')
# paths
parser.add_argument("--corpus", type=str, default='psvg', help="psvg|csu|pp")
parser.add_argument("--hypes", type=str, default='hypes/default.json', help="load in a hyperparameter file")
parser.add_argument("--outputdir", type=str, default='exp/', help="Output directory")
parser.add_argument("--inputdir", type=str, default='', help="Input model dir")
parser.add_argument("--cut_down_len", type=int, default="600", help="sentence will be cut down if tokens num greater than this")
# training
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--bptt_size", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dpout", type=float, default=0.1, help="residual, embedding, attention dropout") # 3 dropouts
parser.add_argument("--warmup_steps", type=int, default=8000, help="OpenNMT uses steps") # TransformerLM uses 0.2% of training data as warmup step, that's 5785 for DisSent5/8, and 8471 for DisSent-All
parser.add_argument("--factor", type=float, default=1.0, help="learning rate scaling factor")
parser.add_argument("--l2", type=float, default=0.01, help="on non-bias non-gain weights")
parser.add_argument("--max_norm", type=float, default=2., help="max norm (grad clipping). Original paper uses 1.")
parser.add_argument("--log_interval", type=int, default=100, help="how many batches to log once")
parser.add_argument('--lm_coef', type=float, default=0.5)
parser.add_argument("--train_emb", default=False, action='store_true', help="Allow to learn embedding, default to False")
parser.add_argument("--init_emb", default=False, action='store_true', help="Initialize embedding randomly, default to False")
parser.add_argument("--pick_hid", default=True, action='store_true', help="Pick correct hidden states")
parser.add_argument("--tied", default=True, action='store_true', help="Tie weights to embedding, should be always flagged True")
parser.add_argument("--model_type", type=str, default="transformer", help="transformer|lstm|caml")
parser.add_argument("--hierachical", default=False, action='store_true', help="hierachical training")
# model
parser.add_argument("--d_ff", type=int, default=2048, help="decoder nhid dimension")
parser.add_argument("--d_model", type=int, default=768, help="decoder nhid dimension")
parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--n_layers", type=int, default=6, help="decoder num layers")
parser.add_argument("--n_lstm_layers", type=int, default=1, help="decoder num lstm layers")
parser.add_argument("--n_kernels", type=int, default=50, help="caml kernel number")
parser.add_argument("--kernel_size", type=int, default=4, help="caml kernel size")
# gpu
parser.add_argument("--seed", type=int, default=1234, help="seed")

params, _ = parser.parse_known_args()

"""
SEED
"""
random.seed(params.seed)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
Logging
"""
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__name__)
if not os.path.exists(params.outputdir): os.makedirs(params.outputdir)
file_handler = logging.FileHandler("{0}/log.txt".format(params.outputdir))
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)
logger.info('\nTogrep : {0}\n'.format(sys.argv[1:]))
logger.info(params)

"""
Default json file loading
"""
json_config = json.load(open(params.hypes))
data_dir = json_config['data_dir']
prefix = json_config['prefix']
encoder_path = json_config['encoder_path']
label_size = json_config['label_size']
if params.init_emb: wordvec_path = json_config['wordvec_path']

"""
BPE encoder
"""
encoder = json.load(open(encoder_path))
encoder['_pad_'] = len(encoder)
encoder['_start_'] = len(encoder)
encoder['_end_'] = len(encoder)
encoder['_unk_'] = len(encoder)
n_special = 4

"""
DATA
"""
train, valid, test = get_data(encoder, data_dir, prefix, params.cut_down_len, label_size) 
max_len = 0.
if params.corpus == 'psvg':
    train['text'] = batchify(np.array(train['text'][0]), params.batch_size)
    valid['text'] = batchify(np.array(valid['text'][0]), params.batch_size)
    test['text'] = batchify(np.array(test['text'][0]), params.batch_size)

"""
Params
"""
if params.init_emb:
    word_embeddings = np.concatenate([np.load(wordvec_path).astype(np.float32),
                                      np.zeros((1, params.d_model), np.float32), # pad, zero-value!
                                      (np.random.randn(n_special - 1, params.d_model) * 0.02).astype(np.float32)], 0)
else:                                                          
    word_embeddings = (np.random.randn(len(encoder), params.d_model) * 0.02).astype(np.float32)


"""
MODEL
"""
# model config
config_model = {
    'n_words': len(encoder),
    'd_model': params.d_model, # same as word embedding size
    'd_ff': params.d_ff, # this is the bottleneck blowup dimension
    'n_layers': params.n_layers,
    'dpout': params.dpout,
    'bsize': params.batch_size,
    'n_classes': label_size,
    'n_heads': params.n_heads,
    'train_emb': params.train_emb,
    'init_emb': params.init_emb,
    'pick_hid': params.pick_hid,
    'tied': params.tied,
    'n_lstm_layers': params.n_lstm_layers,
    'n_kernels': params.n_kernels,
    'kernel_size': params.kernel_size
}

if params.model_type == "lstm":
    logger.info('model lstm')
    model = make_lstm_model(encoder, config_model, word_embeddings)
elif params.model_type == 'caml':
    logger.info('model caml')
    model = make_caml_model(encoder, config_model, word_embeddings)
else:
    logger.info('model transformer')
    model = make_transformer_model(encoder, config_model, word_embeddings)
logger.info(model)
need_grad = lambda x: x.requires_grad
model_opt = NoamOpt(params.d_model, params.factor, params.warmup_steps, torch.optim.Adam(filter(need_grad, model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))
model.cuda()


"""
TRAIN
"""
def train_epoch_csu(epoch):
    # initialize
    logger.info('\nTRAINING : Epoch {}'.format(epoch))
    model.train()
    all_costs, all_em, all_p, all_r, all_f1 = [], [], [], [], []

    # shuffle the data
    permutation = np.random.permutation(len(train['text']))
    text = train['text'][permutation]
    label = train['label'][permutation]

    for stidx in range(0, len(text), params.batch_size):
        # prepare batch
        text_batch = pad_batch(text[stidx: stidx + params.batch_size].tolist(), encoder, pad_start_end=True)
        label_batch = label[stidx: stidx + params.batch_size]
        
        b = Batch(text_batch, label_batch, encoder['_pad_'])

        # model forward
        if params.lm_coef == 0.: clf_output = model(b, clf=True, lm=False)
        else: clf_output, text_y_hat = model(b, clf=True, lm=True)

        # evaluation
        pred = (torch.sigmoid(clf_output) > 0.5).data.cpu().numpy().astype(float)
        em = metrics.accuracy_score(label_batch, pred)
        p, r, f1, s = metrics.precision_recall_fscore_support(label_batch, pred, average='weighted')

        all_em.append(em)
        all_p.append(p)
        all_r.append(r)
        all_f1.append(f1)
 
        if params.hierachical: loss = model.compute_hierachical_loss(clf_output, b.label)
        else: loss = model.compute_clf_loss(clf_output, b.label)
        if params.lm_coef != 0.0:
            lm_loss = model.compute_lm_loss(text_y_hat, b.text_y, b.text_loss_mask)
            loss += params.lm_coef * lm_loss

        all_costs.append(loss.data.item())
        
        # backward
        model_opt.optimizer.zero_grad()
        loss.backward()

        # optimizer step
        model_opt.step()

        # log and reset 
        if len(all_costs) == params.log_interval:
            logger.info('{}; loss {}; em {}; p {}; r {}; f1 {}; lr {}; embed_norm {}'.format(
                stidx, 
                round(np.mean(all_costs), 2),
                round(np.mean(all_em), 3),
                round(np.mean(all_p), 3),
                round(np.mean(all_r), 3),
                round(np.mean(all_f1), 3),
                model_opt.rate(),
                model.tgt_embed[0].lut.weight.data.norm()
            ))
            all_costs, all_em, all_p, all_r, all_f1 = [], [], [], [], []

    # save
    torch.save(model, os.path.join(params.outputdir, "model-{}.pickle".format(epoch)))


def evaluate_epoch_csu(epoch, eval_type='valid'):
    # initialize
    logger.info('\n{} : Epoch {}'.format(eval_type.upper(), epoch))
    model.eval()
    
    # data without shuffle
    if eval_type == 'train': text, label = train['text'], train['label']
    elif eval_type == 'valid': text, label = valid['text'], valid['label']
    else: text, label = test['text'], test['label']

    valid_scores, valid_preds, valid_labels = [], [], []

    for stidx in range(0, len(text), params.batch_size):
        # prepare batch
        text_batch = pad_batch(text[stidx: stidx + params.batch_size].tolist(), encoder, pad_start_end=True)
        label_batch = label[stidx: stidx + params.batch_size]
        
        b = Batch(text_batch, label_batch, encoder['_pad_'])

        # model forward
        clf_output = model(b, clf=True, lm=False)

        # evaluation
        score = torch.sigmoid(clf_output).data.cpu().numpy()
        pred = (score > 0.5).astype(float)
        valid_scores.extend(score.tolist())
        valid_preds.extend(pred.tolist())
        valid_labels.extend(label_batch.tolist())
        
    valid_scores, valid_preds, valid_labels = np.array(valid_scores), np.array(valid_preds), np.array(valid_labels)
    np.save('{}/scores-{}.npy'.format(params.outputdir, epoch), valid_scores)
    
    if params.hierachical:
        parents = json.load(open('data/parents.json'))
        id2label = json.load(open('data/labels.json'))
        label2id = dict([(j, i) for i, j in enumerate(id2label)])
        for i in range(valid_preds.shape[0]): 
            last_pred_i = valid_preds[i].copy()
            while True:
                for j in range(valid_preds.shape[1]):
                    did = id2label[j]
                    flag = True
                    now = did
                    while now in parents:
                        now = parents[now]
                        if now not in label2id: break
                        if valid_preds[i, label2id[now]] == 0:
                            flag = False
                            break
                    if not flag: 
                        valid_preds[i, j] = 0.
                if (valid_preds[i] == last_pred_i).all(): break
                last_pred_i = valid_preds[i].copy()

    em = metrics.accuracy_score(valid_labels, valid_preds)
    p, r, f1, s = metrics.precision_recall_fscore_support(valid_labels, valid_preds, average='weighted')

    logger.info('{}; em {}; p {}; r {}; f1 {}'.format(
        epoch, 
        round(em, 3),
        round(p, 3),
        round(r, 3),
        round(f1, 3)
    ))


def train_epoch_psvg(epoch):
    # initialize
    logger.info('\nTRAINING : Epoch {}'.format(epoch))
    model.train()
    all_costs = []
    text = train['text']

    for stidx in range(0, len(text[0]), params.bptt_size):
        # prepare batch      
        text_batch = text[:, stidx: stidx + params.bptt_size + 1]
        b = Batch(text_batch, [], encoder['_pad_'])

        # model forward
        text_y_hat = model(b, clf=False, lm=True)

        # loss
        loss = model.compute_lm_loss(text_y_hat, b.text_y, b.text_loss_mask)
        all_costs.append(loss.data.item())

        # backward
        model_opt.optimizer.zero_grad()
        loss.backward()

        # optimizer step
        model_opt.step()

        # log and reset
        if len(all_costs) == params.log_interval:
            logger.info('{}; loss {}; perplexity: {}; lr {}; embed_norm: {}'.format(
                stidx, 
                round(np.mean(all_costs), 2),
                round(np.exp(np.mean(all_costs)), 2),
                model_opt.rate(),
                model.tgt_embed[0].lut.weight.data.norm()
            ))
            all_costs = []

    # save
    torch.save(model, os.path.join(params.outputdir, "model-{}.pickle".format(epoch)))


def evaluate_epoch_psvg(epoch, eval_type='valid'):
    # initialize
    logger.info('\n{} : Epoch {}'.format(eval_type.upper(), epoch))
    model.eval()
    text = valid['text'] if eval_type == 'valid' else test['text']
    all_costs = []

    for stidx in range(0, len(text[0]), params.bptt_size):
        # prepare batch
        text_batch = text[stidx: stidx + params.batch_size + 1]
        b = Batch(text_batch, [], encoder['_pad_'])

        # model forward
        text_y_hat = model(b, clf=False, lm=True)

        # loss
        loss = model.compute_lm_loss(text_y_hat, b.text_y, b.text_loss_mask)
        all_costs.append(loss.data.item())

    logger.info('loss {}; perplexity: {}'.format(
        round(np.mean(all_costs), 2),
        round(np.exp(np.mean(all_costs)), 2),
    ))


if __name__ == '__main__':
    epoch = 1
    if params.corpus == 'pp':
        del model
        model = torch.load(params.inputdir)
        model.config = config_model
        evaluate_epoch_csu(epoch, eval_type='test')
    elif params.corpus == 'csu':
        # del model
        # model = torch.load(params.inputdir)
        # evaluate_epoch_csu(epoch, eval_type='test')
        if len(params.inputdir) != 0:
            logger.info('Load Model from %s' % (params.inputdir))
            model.load_state_dict(torch.load(params.inputdir), strict=False)
        while epoch <= params.n_epochs:
            train_epoch_csu(epoch)
            evaluate_epoch_csu(epoch, eval_type='valid')
            evaluate_epoch_csu(epoch, eval_type='test')
            epoch += 1
    elif params.corpus == 'psvg':
        while epoch <= params.n_epochs:
            train_epoch_psvg(epoch)
            evaluate_epoch_psvg(epoch)
            evaluate_epoch_psvg(epoch, eval_type='test')
            epoch += 1

