#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from main.data.global_info import info


class MultiLabelEntropyFeatureSelector:
    """
    Multi-label entropy feature selector.
    All feature will be considered to have 0-1 value during feature selection.
    The binarization rule is: val --> 1 if val > 0 else 0
    """
    def __init__(self):
        self.old_idx_to_new_idx = {}
        self.is_csr = False
        self.is_sparse_feat_vec = False
        self.k_best = None
        self.final_feat_num = None
        self.new_idx_to_entropy = {}

    def fit(self, data, labels, k_best=10):
        """
        Construct the feature selector given data and label
        :param data: sparse or dense feature vector list of the dataset
        :param labels: 0-1 label vector list of the dataset
        :param k_best: select the k_best features
        :return:
        """
        self.k_best = k_best

        label_num = len(labels[0])
        feat_to_label_cnt = {}
        feat_cnt = defaultdict(int)
        labels = np.array(labels, dtype=float)
        label_sum_vec = np.zeros(label_num, dtype=float)
        self.is_sparse_feat_vec = type(data[0]) is dict
        self.is_csr = type(data) is csr_matrix
        data_num = data.shape[0] if self.is_csr else len(data)
        for idx, datum in enumerate(data):
            label_sum_vec += labels[idx]
            if self.is_csr:
                feat_iterator = zip(datum.indices, datum.data)
            elif self.is_sparse_feat_vec:
                feat_iterator = datum.iteritems()
            else:
                feat_iterator = enumerate(datum)
            for feat_idx, feat_val in feat_iterator:
                feat_cnt[feat_idx] += 1
                if feat_val > 0:    # the feature is present
                    if feat_idx not in feat_to_label_cnt:
                        feat_to_label_cnt[feat_idx] = np.zeros(label_num, dtype=float)
                    feat_to_label_cnt[feat_idx] += labels[idx]

        feat_to_entropy = {}
        for feat_idx, label_vec in feat_to_label_cnt.iteritems():
            if feat_cnt[feat_idx] <= 15:
                feat_to_entropy[feat_idx] = 999999
                continue
            p = label_vec / feat_cnt[feat_idx]
            pe = -p * np.log(p)
            pe[np.isnan(pe)] = 0
            q = 1 - p
            qe = -q * np.log(q)
            qe[np.isnan(qe)] = 0
            entropy = np.sum(pe) + np.sum(qe)        # some elements in p may be 0
            if math.isnan(entropy):
                print(label_vec)
                print(feat_cnt[feat_idx])
                print(info.id_to_article[feat_idx])
                print(p)
                print(q)
                print(pe)
                print(qe)
                print("********************************")
            feat_to_entropy[feat_idx] = entropy

            # neg entropy
            # p = (label_sum_vec - label_vec) / (data_num - feat_cnt[feat_idx])
            # pe = -p * np.log(p)
            # pe[np.isnan(pe)] = 0
            # q = 1 - p
            # qe = -q * np.log(q)
            # qe[np.isnan(qe)] = 0
            # entropy = np.sum(pe) + np.sum(qe)        # some elements in p may be 0
            # feat_to_entropy[feat_idx] += entropy

        final_feat_to_entropy = sorted(feat_to_entropy.iteritems(), key=lambda t: t[1], reverse=False)[:k_best]

        for new_idx, feat_pair in enumerate(final_feat_to_entropy):
            feat_idx, entropy = feat_pair
            print(info.id_to_word[feat_idx], entropy, feat_cnt[feat_idx])
            self.old_idx_to_new_idx[feat_idx] = new_idx
            self.new_idx_to_entropy[new_idx] = entropy

        self.final_feat_num = len(feat_to_label_cnt)

    def transform(self, data):
        """
        Transform the input data to selected k-best features
        :param data: input data
        :return: new data
        """
        if self.is_csr:
            res_row = []
            res_col = []
            res_data = []
            for row_idx, row in enumerate(data):
                for col_idx, val in zip(row.indices, row.data):
                    if col_idx in self.old_idx_to_new_idx:
                        # print(col_idx, val)
                        res_row.append(row_idx)
                        res_col.append(self.old_idx_to_new_idx[col_idx])
                        res_data.append(val)
            return csr_matrix((res_data, (res_row, res_col)), shape=[data.shape[0], self.final_feat_num])
        else:
            res_data = []
            for datum in data:
                if self.is_sparse_feat_vec:
                    new_feat_dic = {}
                    for feat_idx, feat_val in datum.iteritems():
                        if feat_idx in self.old_idx_to_new_idx:
                            new_idx = self.old_idx_to_new_idx[feat_idx]
                            new_feat_dic[new_idx] = feat_val
                    res_data.append(new_feat_dic)
                else:
                    new_feat_vec = [0 for i in xrange(self.k_best)]
                    for old_idx, new_idx in self.old_idx_to_new_idx.iteritems():
                        new_feat_vec[new_idx] = datum[old_idx]
                    res_data.append(new_feat_vec)
            return res_data

    def fit_transform(self, data, labels, k_best=10):
        self.fit(data, labels, k_best)
        return self.transform(data)


class NearestWord:
    def __init__(self, tgt_word_path=None, word_embed=None, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.word_embed = word_embed
        self.tgt_word_list = None
        self.nbrs = None
        self.idx_to_tgt_word = None
        self.tgt_word_to_idx = None
        self.sim_word_dic = defaultdict(list)
        if tgt_word_path is not None:
            self.load_word_list(tgt_word_path)
            if word_embed is not None:
                self.build_nbrs()

    def load_word_list(self, inpath):
        with open(inpath, "r") as infile:
            word_list = []
            for line in infile:
                word_list.append(line.strip().split("\t")[0])
        self.tgt_word_list = word_list

    def build_nbrs(self):
        cnt = 0
        word_embed_matrix = []
        target_word_set = set(self.tgt_word_list)
        idx_to_tgt_word = {}
        tgt_word_to_idx = {}
        for idx, embed in enumerate(self.word_embed.embedding_matrix):
            word = self.word_embed.idx2word[idx]
            if word in target_word_set:
                word_embed_matrix.append(embed)
                idx_to_tgt_word[cnt] = word
                tgt_word_to_idx[word] = cnt
                cnt += 1
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='kd_tree', metric="euclidean").fit(word_embed_matrix)
        self.nbrs = nbrs
        self.idx_to_tgt_word = idx_to_tgt_word
        self.tgt_word_to_idx = tgt_word_to_idx

    def load_sim_dic(self, inpath):
        infile = open(inpath, "r")
        for line in infile:
            line = line.strip()
            if line == "":
                break
            line_split = line.split()
            word = line_split[0]
            for sim_word_str in line_split[1:]:
                sim_word, distance = sim_word_str.split(",")
                distance = float(distance)
                self.sim_word_dic[word].append((sim_word, distance))
        infile.close()

    def find_neighbor_word(self, word, n_neighbors=None):
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if len(self.sim_word_dic) > 0:
            words = map(lambda t: t[0], self.sim_word_dic[word][:n_neighbors])
            distances = map(lambda t: t[1], self.sim_word_dic[word][:n_neighbors])
            return words, distances
        embed = self.word_embed.embedding_matrix[self.word_embed.word2idx[word]]
        distances, indices = self.nbrs.kneighbors([embed], n_neighbors=n_neighbors)
        return map(lambda idx: self.idx_to_tgt_word[idx], indices[0]), distances[0]


