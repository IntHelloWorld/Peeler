# -*- coding: utf8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

import sys

import argparse
import numpy as np
import torch
from os import path
from matplotlib.pyplot import table
from prettytable import PrettyTable

from distutils.util import strtobool

from torch.utils.data import DataLoader

from model.model import *
from dataset_builder import *
from dataset_reader import *

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

sys.path.append(".")

parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=234, help="random_seed")

# paths & dirs
parser.add_argument("--corpus_path", type=str, default="./dataset_user_study/samples.txt", help="corpus_path")
parser.add_argument(
    "--token_idx_path",
    type=str,
    default="./dataset_user_study/tokens.txt",
    help="token_idx_path",
)
parser.add_argument(
    "--token_emb_path",
    type=str,
    default="./embeddings_user_study/tokens_embeddings.txt",
    help="token_emb_path",
)
parser.add_argument(
    "--method_emb_path",
    type=str,
    default="./embeddings_user_study/methods_embeddings.txt",
    help="method_emb_path",
)
parser.add_argument("--saved_model_name", type=str, default="b64_encode384_pl20_pc100_hd128_tv256_lr001_10FoldCrossValidation", help="saved_model_name")
parser.add_argument("--output_dir", type=str, default="outputs/b64_encode384_pl20_pc100_hd128_tv256_lr001_10FoldCrossValidation", help="output_dir")

# predict mode
parser.add_argument(
    "--predict_model",
    type=str,
    default="./outputs/b64_encode384_pl20_pc100_hd128_tv256_lr001_10FoldCrossValidation/b64_encode384_pl20_pc100_hd128_tv256_lr001_10FoldCrossValidationcross_5_best.model",
    help="predict_model",
)
parser.add_argument("--predict_mode", type=bool, default=True, help="predict_mode")

# cross validation mode
parser.add_argument("--cross_validation_mode", type=bool, default=False, help="cross_validation_mode")
parser.add_argument("--k_cross", type=int, default=10, help="k_cross")
parser.add_argument("--test_result_path", type=str, default="test_result.csv", help="test_result_path")

# ablation study
parser.add_argument("--ablation_mode", type=str, default="none", help="ablation_mode:none/no_inter_file/no_inner_file")

# RQ3 study
parser.add_argument("--rq3_mode", type=bool, default=False, help="rq3_mode")
parser.add_argument("--no_project", type=str, default="none", help="exclude project:none/projectName")
parser.add_argument("--only_project", type=str, default="none", help="only consider project:none/projectName")

# model hyper parameters
parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
parser.add_argument("--token_embed_size", type=int, default=128, help="token_embed_size")
parser.add_argument("--method_embed_size", type=int, default=384, help="method_embed_size")
parser.add_argument("--encode_size", type=int, default=384, help="embedding_encode_size")
parser.add_argument("--test_vector_size", type=int, default=256, help="test_vector_size")
parser.add_argument("--max_path_length", type=int, default=20, help="max_path_length")
parser.add_argument("--max_path_count", type=int, default=100, help="max_path_count")
parser.add_argument("--lr", type=float, default=0.001, help="lr")
parser.add_argument("--max_epoch", type=int, default=15, help="max_epoch")
parser.add_argument("--bad_epoch_count", type=int, default=10, help="bad_epoch_count")
parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout_prob")
parser.add_argument("--add_bias", type=bool, default=False, help="add_bias")

# LSTM
parser.add_argument(
    "--hidden_dim",
    type=int,
    default=128,
    help="hidden dimension (output dimension) of LSTM model",
)
parser.add_argument("--n_layers", type=int, default=2, help="mount of LSTM model layers")

parser.add_argument("--beta_min", type=float, default=0.9, help="beta_min")
parser.add_argument("--beta_max", type=float, default=0.999, help="beta_max")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument("--label_count", type=int, default=2, help="label_count")
parser.add_argument("--test_batch_cycle", type=int, default=20, help="test_batch_cycle")


parser.add_argument("--no_cuda", action="store_true", default=False, help="no_cuda")
parser.add_argument("--gpu", type=str, default="cuda:0", help="gpu")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

parser.add_argument("--env", type=str, default=None, help="env")
parser.add_argument("--print_sample_cycle", type=int, default=1, help="print_sample_cycle")
parser.add_argument("--eval_method", type=str, default="subtoken", help="eval_method")

parser.add_argument(
    "--find_hyperparams",
    action="store_true",
    default=False,
    help="find optimal hyperparameters",
)
parser.add_argument("--num_trials", type=int, default=100, help="num_trials")

parser.add_argument(
    "--angular_margin_loss",
    action="store_true",
    default=False,
    help="use angular margin loss",
)
parser.add_argument("--angular_margin", type=float, default=0.5, help="angular margin")
parser.add_argument("--inverse_temp", type=float, default=30.0, help="inverse temperature")

parser.add_argument(
    "--infer_method_name",
    type=lambda b: bool(strtobool(b)),
    default=True,
    help="infer method name like code2vec task",
)

args = parser.parse_args()

# set logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s: %(message)s", "%m/%d/%Y %I:%M:%S %p")
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

if not os.path.exists("logs"):
    os.mkdir("logs")
fout = logging.FileHandler(os.path.join("logs", args.saved_model_name + ".log"))
fout.setFormatter(fmt)
logger.addHandler(fout)

device = torch.device(args.gpu if not args.no_cuda and torch.cuda.is_available() else "cpu")
logger.info("device: {0}".format(device))


if args.env == "tensorboard":
    from tensorboardX import SummaryWriter

if args.find_hyperparams:
    import optuna


class Option(object):
    """configurations of the model"""

    def __init__(self, reader):
        self.predict_model = args.predict_model
        self.cross_validation_mode = args.cross_validation_mode
        self.k_cross = args.k_cross
        self.test_result_path = args.test_result_path
        self.ablation_mode = args.ablation_mode

        self.max_path_length = args.max_path_length
        self.max_path_count = args.max_path_count
        self.max_tuple_length = int((args.max_path_length + 1) / 2)

        self.terminal_count = reader.token_vocab.len()
        self.label_count = args.label_count

        self.token_embed_size = args.token_embed_size
        self.method_embed_size = args.method_embed_size
        self.encode_size = args.encode_size
        self.test_vector_size = args.test_vector_size
        self.test_batch_cycle = args.test_batch_cycle

        self.dropout_prob = args.dropout_prob
        self.batch_size = args.batch_size
        self.eval_method = args.eval_method

        self.angular_margin_loss = args.angular_margin_loss
        self.angular_margin = args.angular_margin
        self.inverse_temp = args.inverse_temp

        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers

        self.device = device


def load_data():
    reader = DatasetReader(
        args.corpus_path, args.token_idx_path, args.token_emb_path, args.method_emb_path, args.label_count, args.infer_method_name, args.rq3_mode, args.no_project, args.only_project
    )
    option = Option(reader)
    builder = DatasetBuilder(option, reader)
    return option, reader, builder


def train(option, reader, builder, index=0):
    """train the model"""
    torch.manual_seed(args.random_seed)

    # Add bias
    if args.add_bias:
        label_freq = torch.tensor(reader.label_freq_list, dtype=torch.float32).to(device)
        criterion = nn.NLLLoss(weight=1 / label_freq).to(device)
    else:
        criterion = nn.NLLLoss().to(device)
    model = FlakyDetect(option).to(device)
    # print(model)
    # for param in model.parameters():
    #     print(type(param.data), param.size())

    learning_rate = args.lr
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(args.beta_min, args.beta_max),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.max_epoch / 5), gamma=0.75)

    _train(model, optimizer, scheduler, criterion, option, reader, builder, index, None)


def _train(model, optimizer, scheduler, criterion, option, reader, builder, index, trial):
    """train the model"""

    f1 = 0.0
    best_f1 = None
    best_table = None
    best_result = None
    last_loss = None
    last_accuracy = None
    bad_count = 0

    if args.env == "tensorboard":
        summary_writer = SummaryWriter()
    else:
        summary_writer = None

    try:
        for epoch in range(args.max_epoch):
            if option.cross_validation_mode:
                logger.info("current cross index: {}".format(index))
            logger.info("current learning rate: {}".format(optimizer.param_groups[0]["lr"]))
            train_loss = 0.0

            if option.cross_validation_mode:
                builder.refresh_cross_validation_dataset("positive", index)
            else:
                builder.refresh_positive_train_dataset()
            positive_train_data_loader = iter(
                DataLoader(
                    builder.positive_train_dataset,
                    batch_size=int(option.batch_size / 2),
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=args.num_workers,
                )
            )
            if option.cross_validation_mode:
                builder.refresh_cross_validation_dataset("negetive", index)
            else:
                builder.refresh_negetive_train_dataset()
            negetive_train_data_loader = DataLoader(
                builder.negetive_train_dataset,
                batch_size=int(option.batch_size / 2),
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
            )

            model.train()
            for i_batch, negetive_batched in enumerate(negetive_train_data_loader):
                n_batch = i_batch % len(positive_train_data_loader)
                if n_batch == 0 and i_batch != 0:
                    if option.cross_validation_mode:
                        builder.refresh_cross_validation_dataset("positive", index)
                    else:
                        builder.refresh_positive_train_dataset()
                    positive_train_data_loader = iter(
                        DataLoader(
                            builder.positive_train_dataset,
                            batch_size=int(option.batch_size / 2),
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers,
                        )
                    )
                positive_batched = next(positive_train_data_loader)
                sample_batched = negetive_batched + positive_batched
                random.shuffle(sample_batched)
                _, _, paths, labels, path_lengths, path_nums = load_batch_embeddings(sample_batched, reader, option)
                paths = paths.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                preds, _, _ = model.forward(paths, path_lengths, path_nums)
                loss = calculate_loss(preds, labels, criterion)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Validation each <test_batch_cycle> batches
                if i_batch % option.test_batch_cycle == 0:
                    with torch.no_grad():
                        expected_labels = []
                        actual_labels = []
                        expected_labels.extend(labels)
                        _, preds_label = torch.max(preds, dim=1)
                        actual_labels.extend(preds_label)
                        expected_labels = np.array(expected_labels, dtype=np.uint64)
                        actual_labels = np.array(actual_labels, dtype=np.uint64)
                        accuracy_b, precision_b, recall_b, f1_b = None, None, None, None
                        precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(expected_labels, actual_labels, pos_label=1, average="binary")
                        accuracy_b = accuracy_score(expected_labels, actual_labels)
                        logger.info("[batch {0}]    train_loss:{1}, accuracy:{2}, precision:{3}, recall:{4}, f1:{5}".format(i_batch, loss.item(), accuracy_b, precision_b, recall_b, f1_b))
                else:
                    logger.info("[batch {0}]".format(i_batch))

            # Test
            if option.cross_validation_mode:
                builder.refresh_cross_validation_dataset("test", index)
            else:
                builder.refresh_test_dataset()
            test_data_loader = DataLoader(
                builder.test_dataset,
                batch_size=option.batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
            )
            test_loss, accuracy, precision, recall, f1, proj_dict, test_result = test(model, test_data_loader, criterion, option, reader)

            logger.info("epoch {0}".format(epoch))
            logger.info('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
            logger.info('{{"metric": "test_loss", "value": {0}}}'.format(test_loss))
            logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
            logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
            logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
            logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))

            if args.env == "tensorboard":
                summary_writer.add_scalar("metric/train_loss", train_loss, epoch)
                summary_writer.add_scalar("metric/test_loss", test_loss, epoch)
                summary_writer.add_scalar("metric/accuracy", accuracy, epoch)
                summary_writer.add_scalar("metric/precision", precision, epoch)
                summary_writer.add_scalar("metric/recall", recall, epoch)
                summary_writer.add_scalar("metric/f1", f1, epoch)

            if trial is not None:
                intermediate_value = 1.0 - f1
                trial.report(intermediate_value, epoch)
                if trial.should_prune(epoch):
                    raise optuna.structs.TrialPruned()

            if best_f1 is None or best_f1 < f1:
                logger.info('{{"metric": "best_f1", "value": {0}}}'.format(f1))
                if args.env == "tensorboard":
                    summary_writer.add_scalar("metric/best_f1", f1, epoch)

                best_f1 = f1
                best_result = test_result
                best_table = PrettyTable(["Project", "TP", "FP", "FN", "TN"])
                best_table.align["Project"] = "l"
                best_table.reversesort = False
                for proj in proj_dict:
                    best_table.add_row([proj, proj_dict[proj]["TP"], proj_dict[proj]["FP"], proj_dict[proj]["FN"], proj_dict[proj]["TN"]])

                output_dir = args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if trial is None:
                    if option.cross_validation_mode:
                        torch.save(model.state_dict(), path.join(output_dir, args.saved_model_name + "cross_{}.model".format(index)))
                    else:
                        torch.save(model.state_dict(), path.join(output_dir, args.saved_model_name + ".model"))

            if last_loss is None or train_loss < last_loss or last_accuracy is None or last_accuracy < accuracy:
                last_loss = train_loss
                last_accuracy = accuracy
                bad_count = 0
            else:
                bad_count += 1
            if bad_count > args.bad_epoch_count:
                print("early stop loss:{0}, bad:{1}".format(train_loss, bad_count))
                break
            scheduler.step()  # update the lr

    finally:
        if args.env == "tensorboard":
            summary_writer.close()
        logger.info(best_table)
        with open(os.path.join(output_dir, option.test_result_path), "a") as f:
            for name, res in best_result:
                f.write(",".join(name.split(":")) + ",{}\n".format(res))

    return 1.0 - f1


def test(model, data_loader, criterion, option, reader):
    """test the model"""
    model.eval()
    with torch.no_grad():
        TP = FP = FN = TN = 0
        test_loss = 0.0
        expected_labels = []
        actual_labels = []
        test_names = []
        res = []
        proj_dict = {}

        for i_batch, sample_batched in enumerate(data_loader):
            logger.info("[batch {0}]".format(i_batch))
            _, inputs_names, paths, labels, path_lengths, path_nums = load_batch_embeddings(sample_batched, reader, option)
            paths = paths.to(device)
            labels = labels.to(device)
            test_names.extend(inputs_names)
            expected_labels.extend(labels)
            preds, _, _ = model.forward(paths, path_lengths, path_nums)
            loss = calculate_loss(preds, labels, criterion)
            test_loss += loss.item()
            _, preds_label = torch.max(preds, dim=1)
            actual_labels.extend(preds_label)
            for i in range(len(labels)):
                projName = inputs_names[i].split(":")[0]
                if projName not in proj_dict:
                    proj_dict[projName] = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                if labels[i] == 1:
                    if preds_label[i] == labels[i]:
                        TP += 1
                        proj_dict[projName]["TP"] += 1
                        res.append("TP")
                    else:
                        FN += 1
                        proj_dict[projName]["FN"] += 1
                        res.append("FN")
                elif labels[i] == 0:
                    if preds_label[i] == labels[i]:
                        TN += 1
                        proj_dict[projName]["TN"] += 1
                        res.append("TN")
                    else:
                        FP += 1
                        proj_dict[projName]["FP"] += 1
                        res.append("FP")

        expected_labels = np.array(expected_labels, dtype=np.uint64)
        actual_labels = np.array(actual_labels, dtype=np.uint64)
        accuracy, precision, recall, f1 = None, None, None, None
        precision, recall, f1, _ = precision_recall_fscore_support(expected_labels, actual_labels, pos_label=1, average="binary")
        accuracy = accuracy_score(expected_labels, actual_labels)
        logger.info("TP:{}, FP:{}, FN:{}, TN:{}".format(TP, FP, FN, TN))
        test_result = zip(test_names, res)

        return test_loss, accuracy, precision, recall, f1, proj_dict, test_result


def predict(option, reader, builder):

    model = FlakyDetect(option).to(device)
    logger.info("Load pre-trained model...")
    model.load_state_dict(torch.load(option.predict_model))

    builder.refresh_all_dataset()
    predict_data_loader = DataLoader(
        builder.all_dataset,
        batch_size=option.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Add bias
    if args.add_bias:
        label_freq = torch.tensor(reader.label_freq_list, dtype=torch.float32).to(device)
        criterion = nn.NLLLoss(weight=1 / label_freq).to(device)
    else:
        criterion = nn.NLLLoss().to(device)

    _, accuracy, precision, recall, f1, proj_dict, test_result = test(model, predict_data_loader, criterion, option, reader)
    logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
    logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
    logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
    logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, option.test_result_path), "a") as f:
        for name, res in test_result:
            f.write(",".join(name.split(":")) + ",{}\n".format(res))

    table = PrettyTable(["Project", "TP", "FP", "FN", "TN"])
    table.align["Project"] = "l"
    table.sortby = "TP"
    table.reversesort = False
    TP = FP = FN = TN = 0
    for proj in proj_dict:
        TP += proj_dict[proj]["TP"]
        FP += proj_dict[proj]["FP"]
        FN += proj_dict[proj]["FN"]
        TN += proj_dict[proj]["TN"]
        table.add_row([proj, proj_dict[proj]["TP"], proj_dict[proj]["FP"], proj_dict[proj]["FN"], proj_dict[proj]["TN"]])
    table.add_row(["Total", TP, FP, FN, TN])
    logger.info(table)


def load_batch_embeddings(batch, reader, option):
    """Loading embeddings for a batch of data"""
    inputs_id = []
    inputs_names = []
    inputs_paths = []
    inputs_label = []
    path_lengths = []
    path_nums = []
    max_tuple_length = option.max_tuple_length
    tuple_embed_size = option.method_embed_size + option.token_embed_size
    for item in batch:
        inputs_id.append(item["id"])
        inputs_names.append(item["name"])
        inputs_label.append(int(item["label"]))
        paths = []
        #  Shuffling path contexts and picking up items (items <= max_path_length)
        random.shuffle(item["paths"])
        for path_context in item["paths"][: option.max_path_count]:  # TODO: if need a algorithm to select paths here
            embed_tuple = []
            tuple_state = 0
            path = []
            for element in path_context[: option.max_path_length]:
                if element[0][0].startswith("FUNC:"):
                    if option.ablation_mode == "no_inter_file":
                        embed_tuple.append(np.array([0] * option.method_embed_size, np.float32))
                    else:
                        if option.ablation_mode == "no_inner_file":
                            tuple_state = 2
                        method_idx = int(element[0][0][5:])
                        embed_tuple.append(reader.method_emb_vocab[method_idx])
                else:
                    tokens = []
                    for token in element:
                        sub_tokens = []
                        for sub_token in token:
                            sub_tokens.append(reader.token_emb_vocab[int(sub_token)])
                        sub_tokens = np.array(sub_tokens, np.float32)
                        tokens.append(sub_tokens.mean(axis=0))
                    tokens = np.array(tokens, np.float32)
                    embed_tuple.append(tokens.mean(axis=0))
                if len(embed_tuple) == 2:
                    if option.ablation_mode == "no_inner_file":
                        if tuple_state == 2:
                            tuple_state = 1
                        elif tuple_state == 1:
                            combined = embed_tuple[0]
                            combined = np.pad(combined, (0, tuple_embed_size - combined.size), "constant")
                            path.append(combined)
                            embed_tuple = []
                            tuple_state = 0
                            continue
                        elif tuple_state == 0:
                            combined = np.array([0] * tuple_embed_size, np.float32)
                            path.append(combined)
                            embed_tuple = []
                            continue
                    combined = np.concatenate((embed_tuple[0], embed_tuple[1]), axis=0)
                    combined = np.pad(combined, (0, tuple_embed_size - combined.size), "constant")
                    path.append(combined)
                    embed_tuple = []

            if len(embed_tuple) == 1:
                if option.ablation_mode == "no_inner_file":
                    if tuple_state == 1:
                        combined = embed_tuple[0]
                        combined = np.pad(combined, (0, tuple_embed_size - combined.size), "constant")
                        path.append(combined)
                else:
                    combined = np.pad(embed_tuple[0], (0, tuple_embed_size - embed_tuple[0].size), "constant")
                    path.append(combined)
            path_lengths.append(len(path))
            #  Pad single path
            if len(path) < max_tuple_length:
                path = pad_inputs(path, max_tuple_length, tuple_embed_size)
            path = np.array(path, dtype=np.float32)
            paths.append(path)

        path_nums.append(len(paths))
        paths = np.array(paths, dtype=np.float32)
        inputs_paths.extend(paths)

    inputs_paths = torch.tensor(inputs_paths, dtype=torch.float32)
    inputs_label = torch.tensor(inputs_label, dtype=torch.long)
    path_lengths = torch.tensor(path_lengths, dtype=torch.int32)
    return inputs_id, inputs_names, inputs_paths, inputs_label, path_lengths, path_nums


def calculate_loss(predictions, label, criterion):
    preds = F.log_softmax(predictions, dim=1)
    loss = criterion(preds, label)
    return loss


def collate_fn(data):
    return data


def pad_inputs(data, length, embed_size, pad_value=0):
    """pad values"""
    count = length - len(data)
    data.extend([np.array([pad_value] * embed_size, dtype=np.float32)] * count)
    return data


#
# for optuna
#
def find_optimal_hyperparams():
    """find optimal hyperparameters"""
    torch.manual_seed(args.random_seed)

    reader = DatasetReader(
        args.corpus_path,
        args.token_idx_path,
        args.token_emb_path,
        args.method_emb_path,
        infer_method=args.infer_method_name,
    )
    option = Option(reader)

    builder = DatasetBuilder(option, reader)

    label_freq = torch.tensor(reader.label_vocab.get_freq_list(), dtype=torch.float32).to(device)
    criterion = nn.NLLLoss(weight=1 / label_freq).to(device)

    def objective(trial):
        option.encode_size = int(trial.suggest_loguniform("encode_size", 100, 300))
        option.dropout_prob = trial.suggest_loguniform("dropout_prob", 0.5, 0.9)
        option.batch_size = int(trial.suggest_loguniform("batch_size", 256, 2048))

        model = FlakyDetect(option).to(device)
        # print(model)
        # for param in model.parameters():
        #     print(type(param.data), param.size())

        weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
        lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        return _train(model, optimizer, criterion, option, reader, builder, trial)

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.num_trials)

    best_params = study.best_params
    best_value = study.best_value
    if args.env == "floyd":
        print("best hyperparams: {0}".format(best_params))
        print("best value: {0}".format(best_value))
    else:
        logger.info("optimal hyperparams: {0}".format(best_params))
        logger.info("best value: {0}".format(best_value))


def main():
    option, reader, builder = load_data()
    if args.predict_mode:
        predict(option, reader, builder)
    elif args.cross_validation_mode:
        for i in range(0, args.k_cross):
            train(option, reader, builder, i)
    else:
        if args.find_hyperparams:
            find_optimal_hyperparams()
        else:
            train(option, reader, builder)


if __name__ == "__main__":
    main()
