# -*- coding: utf8 -*-

from collections import defaultdict

from sklearn.utils import shuffle
from dataset import *

from tqdm import tqdm
import logging
import numpy as np

logger = logging.getLogger()

QUESTION_TOKEN_INDEX = 1
QUESTION_TOKEN_NAME = "@question"


class VocabReader(object):
    """read vocabulary file"""

    def __init__(self, filename, extra_tokens=[]):
        self.filename = filename
        self.extra_tokens = extra_tokens

    def read(self):
        vocab = Vocab()
        extra_size = len(self.extra_tokens)
        index = 1
        for name in self.extra_tokens:
            vocab.append(name, index)

        with open(self.filename, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                data = line.strip(" \r\n\t").split("\t")
                index = int(data[0])
                if index > 0:
                    index += extra_size
                if len(data) > 1:
                    name = data[1]
                else:
                    name = ""
                vocab.append(name, index)
                line = f.readline()
        return vocab


class EmbeddingReader(object):
    """read embedding file"""

    def __init__(self, filename):
        self.filename = filename

    def read(self, begin=1):
        # <begin> is the index of the begining of an embedding
        emb_vocab = {}
        index = 1

        with open(self.filename, mode="r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                data = line.strip(" \r\n\t").split(" ")
                index = int(data[0])
                emb = np.array(line.split(" ")[begin:], np.float32)
                if index not in emb_vocab:
                    emb_vocab[index] = emb
                line = f.readline()
        return emb_vocab


class DatasetReader(object):
    """read dataset file"""

    def __init__(self, corpus_path, token_index_path, token_emb_path, method_emb_path, label_count, infer_method, rq3_mode, no_project, only_project):
        self.token_vocab = VocabReader(token_index_path).read()
        logger.info("token vocab size: {0}".format(self.token_vocab.len()))

        self.token_emb_vocab = EmbeddingReader(token_emb_path).read(2)
        logger.info("token embedding vocab size: {0}".format(len(self.token_emb_vocab)))

        self.method_emb_vocab = EmbeddingReader(method_emb_path).read(2)
        logger.info("method embedding vocab size: {0}".format(len(self.method_emb_vocab)))

        self.infer_method = infer_method

        self.label_vocab = Vocab()
        self.label_freq_list = [0] * label_count
        self.positive_items = []
        self.negetive_items = []
        self.load(corpus_path)

        """For single project"""
        # projects = ["hbase", "okhttp", "spring-boot", "ambari"]
        # projects = ["spring-boot"]
        # self.positive_items = [item for item in self.positive_items if item.projectName in projects]
        # self.negetive_items = [item for item in self.negetive_items if item.projectName in projects]

        if rq3_mode:
            if no_project != "none":
                self.positive_items = [item for item in self.positive_items if item.projectName != no_project]
                self.negetive_items = [item for item in self.negetive_items if item.projectName != no_project]
            elif only_project != "none":
                self.positive_items = [item for item in self.positive_items if item.projectName == only_project]
                self.negetive_items = [item for item in self.negetive_items if item.projectName == only_project]

        logger.info("label vocab size: {0}".format(self.label_vocab.len()))
        logger.info("corpus: {0}".format(len(self.positive_items) + len(self.negetive_items)))

    def load(self, corpus_path):
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            code_data = None
            path_contexts_append = None
            parse_mode = 0
            label_vocab = self.label_vocab
            label_vocab_append = label_vocab.append
            label_freq = {}
            line = f.readline()
            n_samples = int(line.strip(" \r\n\t")[10:])
            logger.info("sample size: {0}".format(n_samples))

            with tqdm(total=n_samples) as pbar:
                pbar.set_description("Loading samples:")
                line = f.readline()
                while line:
                    line = line.strip(" \r\n\t")

                    if code_data is None:
                        code_data = CodeData()
                        path_contexts_append = code_data.path_contexts.append

                    if line.startswith("id:"):
                        code_data.id = int(line[3:])
                        pbar.update(1)
                    elif line.startswith("projectName:"):
                        code_data.projectName = line[12:]
                    elif line.startswith("label:"):
                        code_data.label = line[6:]
                        if label_freq.get(code_data.label) is None:
                            label_freq[code_data.label] = 0
                        label_freq[code_data.label] += 1
                    elif line.startswith("method:"):
                        method = line[7:]
                        code_data.method = method
                        normalized_label = Vocab.normalize_method_name(method)
                        subtokens = Vocab.get_method_subtokens(normalized_label)
                        normalized_lower_label = normalized_label.lower()
                        code_data.normalized_label = normalized_lower_label
                        if self.infer_method:
                            label_vocab_append(normalized_lower_label, code_data.label, subtokens=subtokens)
                    elif line.startswith("class:"):
                        code_data.source = line[6:]
                    elif line.startswith("paths:"):
                        parse_mode = 1
                    elif line == "":
                        parse_mode = 0
                        if code_data is not None and len(code_data.path_contexts) > 0:
                            if code_data.label == "1":
                                self.positive_items.append(code_data)
                            elif code_data.label == "0":
                                self.negetive_items.append(code_data)
                            code_data = None
                    elif parse_mode == 1:
                        path_context = [e for e in re.split(r"\(|\)", line) if e != ""]
                        path_context = [re.split(r"\t", e) for e in path_context]
                        path_context = [[tuple(re.split(r"\|", w)) for w in e] for e in path_context]
                        path_contexts_append((path_context))
                    line = f.readline()
            if code_data is not None and len(code_data.path_contexts) > 0:
                if code_data.label == "1":
                    self.positive_items.append(code_data)
                elif code_data.label == "0":
                    self.negetive_items.append(code_data)

            # self.label_freq_list = list(label_freq.values())
            for i in range(len(label_freq)):
                self.label_freq_list[i] = list(label_freq.values())[i]
