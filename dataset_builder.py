# -*- coding: utf8 -*-

from turtle import shape
from click import option
from matplotlib.pyplot import axis
import torch

from dataset_reader import *

import random
import logging

logger = logging.getLogger()


class DatasetBuilder(object):
    """transform dataset for training and test"""

    def __init__(self, option, reader):
        split_ratio = 1 / option.k_cross if option.cross_validation_mode else 0.2
        self.reader = reader
        pl = len(reader.positive_items)
        nl = len(reader.negetive_items)
        p_step = int(pl * split_ratio) + 1
        n_step = int(nl * split_ratio) + 1
        random.shuffle(reader.positive_items)
        random.shuffle(reader.negetive_items)

        if option.cross_validation_mode:
            p_cross_dataset = []
            n_cross_dataset = []
            t_cross_dataset = []
            for i in range(0, option.k_cross):
                t_cross_dataset.append([])

            for i in range(0, pl, p_step):
                p_cross_dataset.append(reader.positive_items[0:i] + reader.positive_items[i + p_step : pl])
                t_cross_dataset[int(i / p_step)] += reader.positive_items[i : i + p_step]
            for j in range(0, nl, n_step):
                n_cross_dataset.append(reader.negetive_items[0:j] + reader.negetive_items[j + n_step : nl])
                t_cross_dataset[int(j / n_step)] += reader.negetive_items[j : j + n_step]

            logger.info("train dataset size: {0}".format(len(p_cross_dataset[0]) + len(n_cross_dataset[0])))
            logger.info("test dataset size: {0}".format(len(t_cross_dataset[0])))
            self.p_cross_dataset = p_cross_dataset
            self.n_cross_dataset = n_cross_dataset
            self.t_cross_dataset = t_cross_dataset

        else:
            positive_train_items = reader.positive_items[p_step:]
            negetive_train_items = reader.negetive_items[n_step:]
            test_items = reader.positive_items[0:p_step] + reader.negetive_items[0:n_step]

            """For single project"""
            # projects = {"hbase": {"p": [], "n": []}, "okhttp": {"p": [], "n": []}, "spring-boot": {"p": [], "n": []}, "ambari": {"p": [], "n": []}}
            # projects = {"spring-boot": {"p": [], "n": []}}
            # for item in reader.positive_items:
            #     if item.projectName in projects:
            #         projects[item.projectName]["p"].append(item)
            # for item in reader.negetive_items:
            #     if item.projectName in projects:
            #         projects[item.projectName]["n"].append(item)
            # ratio = 0.2
            # positive_train_items = []
            # negetive_train_items = []
            # test_items = []
            # for proj in projects:
            #     p = projects[proj]["p"]
            #     n = projects[proj]["n"]
            #     random.shuffle(p)
            #     random.shuffle(n)
            #     p_step = int(len(p) * ratio)
            #     n_step = int(len(n) * ratio)
            #     positive_train_items.extend(p[0:p_step])
            #     negetive_train_items.extend(n[0:n_step])
            #     test_items.extend(p[p_step:])
            #     test_items.extend(n[n_step:])

            logger.info("train dataset size: {0}".format(len(positive_train_items) + len(negetive_train_items)))
            logger.info("test dataset size: {0}".format(len(test_items)))

            self.positive_train_items = positive_train_items
            self.negetive_train_items = negetive_train_items
            self.test_items = test_items
            self.positive_train_dataset = None
            self.negetive_train_dataset = None
            self.test_dataset = None
            self.all_dataset = None

    def refresh_cross_validation_dataset(self, type, index):
        if type == "positive":
            (
                inputs_id,
                inputs_names,
                inputs_paths,
                inputs_label,
            ) = self.build_data(self.p_cross_dataset[index], switch="positive train")
            self.positive_train_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)
        elif type == "negetive":
            (
                inputs_id,
                inputs_names,
                inputs_paths,
                inputs_label,
            ) = self.build_data(self.n_cross_dataset[index], switch="negetive train")
            self.negetive_train_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)
        elif type == "test":
            (
                inputs_id,
                inputs_names,
                inputs_paths,
                inputs_label,
            ) = self.build_data(self.t_cross_dataset[index], switch="test")
            self.test_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)

    def refresh_all_dataset(self):
        """refresh the whole dataset"""
        all_items = self.reader.positive_items + self.reader.negetive_items
        (
            inputs_id,
            inputs_names,
            inputs_paths,
            inputs_label,
        ) = self.build_data(all_items, switch="validation")
        self.all_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)

    def refresh_positive_train_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        (
            inputs_id,
            inputs_names,
            inputs_paths,
            inputs_label,
        ) = self.build_data(self.positive_train_items, switch="positive train")
        self.positive_train_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)

    def refresh_negetive_train_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        (
            inputs_id,
            inputs_names,
            inputs_paths,
            inputs_label,
        ) = self.build_data(self.negetive_train_items, switch="negetive train")
        self.negetive_train_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)

    def refresh_test_dataset(self):
        """refresh test dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        (
            inputs_id,
            inputs_names,
            inputs_paths,
            inputs_label,
        ) = self.build_data(self.test_items, switch="test")
        self.test_dataset = CodeDataset(inputs_id, inputs_names, inputs_paths, inputs_label)

    def build_data(self, items, switch):
        inputs_id = []
        inputs_names = []
        inputs_paths = []
        inputs_label = []
        for item in tqdm(items, desc="load {} embeddings".format(switch)):
            inputs_id.append(item.id)
            inputs_names.append(item.projectName + ":" + item.source + ":" + item.method)
            #  Shuffling path contexts and picking up items (items <= max_path_length)
            random.shuffle(item.path_contexts)
            inputs_paths.append(item.path_contexts)
            inputs_label.append(int(item.label))
        return inputs_id, inputs_names, inputs_paths, inputs_label
