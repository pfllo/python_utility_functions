#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import struct
import math
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from scipy.spatial import distance
except ImportError as e:
    print("Warning: Failed to import scipy.spatial.distance", file=sys.stderr)

from functools import reduce
from nltk.tokenize import TweetTokenizer

from .math_util import safe_divide, cal_F1
from .collection_util import pad_list

tokenizer = TweetTokenizer()


class WordEmbedding:
    # special tokens that may be usefule
    unk_token = "*UNKNOWN*"
    pad_token = "*PADDING*"
    subj_token = "*SUBJECT*"
    obj_token = "*OBJECT*"
    name_token = "*NAME*"
    time_token = "*TIME*"
    num_token = "*NUM*"
    accusal_token = "*ACCUSAL*"

    def __init__(self, inpath=None, init_scale=0.01, normalize=False, has_header=False):
        """
        Constructor of word embedding object
        :param inpath: input path of the word vector file
        :param init_scale: uniform initialization scale
        :param normalize: normalize the word vectors to length 1 while initialization or not
        :return:
        """
        self.word2idx = {}
        self.idx2word = {}
        self.embedding_matrix = []
        self.init_scale = init_scale
        self.vec_dim = None
        self.word_num = 0
        if inpath:
            if inpath.endswith(".txt"):
                self.load_from_txt(inpath, has_header=has_header, normalize=normalize)
            elif inpath.endswith(".bin"):
                self.load_from_bin(inpath, normalize=normalize)
            else:
                raise(SystemError('Unable to handle the given word embedding file with extension "{0}"'.format(
                    "." + inpath.split(".")[-1]
                )))

    def load_from_txt(self, inpath, has_header=False, normalize=False):
        """
        Load word vector, generate related data structures. We require following format:
        first line: number_of_words, dimension_embedding (separated by tab, optional)
        other lines: one word per line, separated by tab, first element is word, the rest are numbers
        :param inpath:  input file path
        :param has_header: the word vector file has header line or not
        :param normalize: normalize the word vectors to length 1 while initialization or not
        :return:
        """
        infile = open(inpath, "r")
        if has_header:
            header_split = infile.readline().strip().split()
            # self.word_num = int(header_split[0])
            self.vec_dim = int(header_split[1])
        for line in infile:
            line_split = line.strip().split()
            if self.vec_dim is None:
                self.vec_dim = len(line_split) - 1
            if len(line_split) - 1 == self.vec_dim-1:     # the word is a blank symbol
                continue
            word = line_split[0]
            self.word2idx[word] = self.word_num
            self.idx2word[self.word_num] = word
            self.word_num += 1
            embedding = map(lambda x: float(x), line_split[1:])
            if normalize:
                sqrt_sum_square = math.sqrt(reduce(lambda acc, cur: acc + cur**2, embedding, 0))
                embedding = [ele / sqrt_sum_square for ele in embedding]
            self.embedding_matrix.append(embedding)
        # if not has_header:
        #     self.vec_dim = len(self.embedding_matrix[0])
        infile.close()

    def load_from_bin(self, inpath, normalize=False):
        """
        Load word vector from .bin file with the following format:
        first line contains: number_of_words, dimension_embedding (string)
        other lines: one word per line, separated by tab, first element is word, the rest are numbers (float32)
        :param inpath: input file path
        :param normalize: normalize the word vectors to length 1 while initialization or not
        :return:
        """
        infile = open(inpath, "rb")
        header_split = infile.readline().strip().split()
        self.word_num = int(header_split[0])
        self.vec_dim = int(header_split[1])
        for word_idx in range(self.word_num):
            word = ""
            while True:
                tmp_char = infile.read(1)
                if tmp_char == "" or tmp_char == " ":
                    break
                if tmp_char != "\n":    # not sure why
                    word += tmp_char
            word_vec = [struct.unpack("f", infile.read(4))[0] for iii in range(self.vec_dim)]
            if normalize:
                sqrt_sum_square = math.sqrt(reduce(lambda acc, cur: acc + cur**2, word_vec, 0))
                word_vec = [ele / sqrt_sum_square for ele in word_vec]
            self.word2idx[word] = word_idx
            self.idx2word[word_idx] = word
            self.embedding_matrix.append(word_vec)
        infile.close()

    def add_words(self, word_array):
        """
        Add a list of words, the vector of this words will be randomly initialized.
        Note: words that already exist will be ignored
        :param word_array:  the array of words to be added
        :return:
        """
        for word in word_array:
            if word in self.word2idx:  # already exists
                continue
            self.word2idx[word] = self.word_num
            self.idx2word[self.word_num] = word
            self.word_num += 1
            embedding = np.random.uniform(-self.init_scale, self.init_scale, self.vec_dim).tolist()
            self.embedding_matrix.append(embedding)

    def normalize_vec(self):
        for i in range(len(self.word2idx)):
            square_sum = reduce(lambda acc, ele: acc + ele**2, self.embedding_matrix[i])
            square_sum_sqrt = math.sqrt(square_sum)
            for j in range(len(self.embedding_matrix[i])):
                self.embedding_matrix[i][j] /= square_sum_sqrt

    def persist(self):
        """
        Tell this object that the words will not change afterwards.
        After this operation, this object is ready to use.
        :return:
        """
        self.embedding_matrix = np.array(self.embedding_matrix)

    def dump_word_index(self, outpath):
        with open(outpath, "w") as outfile:
            for key, value in self.word2idx.iteritems():
                outfile.writelines("{0}\t{1}\n".format(key, value))

    def find_neighbor_word(self, in_word, out_len=20):
        if in_word not in self.word2idx:
            return []
        in_index = self.word2idx[in_word]
        src_vec = self.embedding_matrix[in_index]
        sim_array = []
        for i in range(len(self.embedding_matrix)):
            if i == in_index:
                continue
            cosine_sim = 1 - distance.cosine(src_vec, self.embedding_matrix[i])
            sim_array.append(cosine_sim)

        sorted_sim_index = sorted(range(len(sim_array)), key=lambda k: sim_array[k], reverse=True)
        res_array = []
        skipped = False
        for i in range(out_len):
            temp_index = sorted_sim_index[i]
            if temp_index == in_index:
                skipped = True
                continue
            res_array.append((self.idx2word[temp_index], sim_array[temp_index]))

        if skipped:
            temp_index = sorted_sim_index[out_len]
            res_array.append((self.idx2word[temp_index], sim_array[temp_index]))

        return res_array

    def prompt_neighbor(self):
        while True:
            str_split = input("> ").strip().split()
            out_len = 20
            if len(str_split) == 2:
                out_len = int(str_split[1])
            word = str_split[0]
            res_array = self.find_neighbor_word(word, out_len)
            for ele in res_array:
                print(ele)
            if len(res_array) == 0:
                print("Not in vocabulary")
            print("")

    @staticmethod
    def load_word_index(inpath):
        word_to_index = {}
        index_to_word = {}
        with open(inpath, "r") as infile:
            for line in infile:
                line = line.strip()
                if line == "":
                    break
                line_split = line.split("\t")
                word = line_split[0]
                index = int(line_split[1])
                word_to_index[word] = index
                index_to_word[index] = word
        return word_to_index, index_to_word


def load_stop_words(inpath):
    """
    Load stop word dictionary.
    :param inpath:  input file path, one word per line
    :return:    {stop_word: True}
    """
    res_dic = {}
    with open(inpath, "r") as infile:
        for line in infile:
            if line.strip() == "":  # EOF
                break
            res_dic[line.strip()] = True
    return res_dic


def sent_to_wid(sent, length, word_to_index, padding_symbol, unknown_symbol):
    """
    Convert a sent to word index array
    :param sent: input sentence (space separated words string)
    :param length: word index length
    :param word_to_index:   word to index mapping
    :param padding_symbol:  padding symbol (string, not index)
    :param unknown_symbol:  unknown symbol (string, not index)
    :return:
    """
    tokens = tokenizer.tokenize(sent)
    wid_array = []
    for token in tokens:
        token = token.lower()
        if token in word_to_index:
            wid_array.append(word_to_index[token])
        else:
            wid_array.append(word_to_index[unknown_symbol])
    wid_array = pad_list(wid_array, length, word_to_index[padding_symbol])
    return wid_array


def summarize_multi_class_prediction(prediction, target, neg_id=-1, in_detail=False):
    """
    Given prediction of a model, and the corresponding target, calculate summary statistics
    :param prediction:  predicted label index array
    :param target:  target label index array
    :param neg_id:   negative target id (used to calculate overall precision and recall), default is -1, then precision
    and recall is the same as accuracy
    :param in_detail:   detail version or not
    :return:    summary dictionary (keys: precision, right_cnt, total_cnt),
                detail version has these statistics and recall, F1 for each label
    """
    assert len(prediction) == len(target)
    summary = dict()
    summary["right_cnt"] = 0
    summary["total_cnt"] = len(prediction)
    summary["posi_right_cnt"] = 0
    summary["posi_pred_cnt"] = 0
    summary["posi_total_cnt"] = 0
    if in_detail:
        summary["label_stats"] = {}

    def ensure_label_existence(label):
        if label not in summary["label_stats"]:
            summary["label_stats"][label] = {}
            summary["label_stats"][label]["right_cnt"] = 0
            summary["label_stats"][label]["pred_cnt"] = 0
            summary["label_stats"][label]["total_cnt"] = 0

    for i in range(len(prediction)):
        if prediction[i] == target[i]:
            summary["right_cnt"] += 1
            if prediction[i] != neg_id:
                summary["posi_right_cnt"] += 1
        if prediction[i] != neg_id:
            summary["posi_pred_cnt"] += 1
        if target[i] != neg_id:
            summary["posi_total_cnt"] += 1
        if in_detail:
            ensure_label_existence(prediction[i])
            ensure_label_existence(target[i])
            summary["label_stats"][target[i]]["total_cnt"] += 1
            summary["label_stats"][prediction[i]]["pred_cnt"] += 1
            if prediction[i] == target[i]:
                summary["label_stats"][prediction[i]]["right_cnt"] += 1

    summary["accuracy"] = summary["right_cnt"] / summary["total_cnt"]
    summary["precision"] = safe_divide(summary["posi_right_cnt"], summary["posi_pred_cnt"])
    summary["recall"] = safe_divide(summary["posi_right_cnt"], summary["posi_total_cnt"])
    if in_detail:
        for key, value in summary["label_stats"].iteritems():
            value["precision"] = value["right_cnt"] / value["pred_cnt"] if value["pred_cnt"] else 0
            value["recall"] = value["right_cnt"] / value["total_cnt"] if value["total_cnt"] else 0
    return summary


def summarize_multi_label_prediction(prediction, target, in_detail=False, threshold=0.5, neg_idx=-1):
    """
    Given prediction of a model, and the corresponding target, calculate summary statistics
    :param prediction:  (batch_num, label_num), each element represent the probability of each label
    :param target:  (batch_num, label_num) the element of gold label is one
    :param in_detail:   detail version or not
    :param threshold:   the threshold to determine a positive detection (can also be a list)
    :param neg_idx: index of negative label
    :return:    summary dictionary (keys: precision, right_cnt, total_cnt),
                detail version has these statistics and recall, F1 for each label
    """
    assert len(prediction) == len(target)
    assert len(prediction[0]) == len(target[0])
    summary = dict()
    summary["ins_right_cnt"] = 0
    summary["ins_total_cnt"] = 0
    summary["label_right_cnt"] = 0
    summary["label_total_cnt"] = 0
    summary["posi_label_right_cnt"] = 0
    summary["posi_label_total_cnt"] = 0
    summary["posi_label_pred_cnt"] = 0
    if in_detail:
        summary["label_stats"] = {}

    def ensure_label_existence(label):
        if label not in summary["label_stats"]:
            summary["label_stats"][label] = {}
            summary["label_stats"][label]["right_cnt"] = 0
            summary["label_stats"][label]["pred_cnt"] = 0
            summary["label_stats"][label]["total_cnt"] = 0

    for i in range(len(prediction)):
        all_right = True
        summary["ins_total_cnt"] += 1
        for j in range(len(prediction[i])):
            summary["label_total_cnt"] += 1
            if type(threshold) is list:
                tmp_pred = 1 if prediction[i][j] >= threshold[j] else 0
            else:
                tmp_pred = 1 if prediction[i][j] >= threshold else 0
            if tmp_pred == target[i][j]:
                summary["label_right_cnt"] += 1
            else:
                all_right = False
            if j != neg_idx:    # positive label
                if target[i][j] == 1:
                    summary["posi_label_total_cnt"] += 1
                if tmp_pred == 1:
                    summary["posi_label_pred_cnt"] += 1
                if target[i][j] == 1 and tmp_pred == 1:
                    summary["posi_label_right_cnt"] += 1
            if in_detail:
                ensure_label_existence(j)
                if target[i][j] == 1:
                    summary["label_stats"][j]["total_cnt"] += 1
                if tmp_pred == 1:
                    summary["label_stats"][j]["pred_cnt"] += 1
                if target[i][j] == 1 and tmp_pred == 1:
                    summary["label_stats"][j]["right_cnt"] += 1

        if all_right:
            summary["ins_right_cnt"] += 1

    summary["ins_accuracy"] = summary["ins_right_cnt"] / summary["ins_total_cnt"] if summary["ins_total_cnt"] else 0
    summary["label_accuracy"] = summary["label_right_cnt"] / summary["label_total_cnt"] if summary["label_total_cnt"] else 0
    summary["posi_precision"] = summary["posi_label_right_cnt"] / summary["posi_label_pred_cnt"] if summary["posi_label_pred_cnt"] else 0
    summary["posi_recall"] = summary["posi_label_right_cnt"] / summary["posi_label_total_cnt"] if summary["posi_label_total_cnt"] else 0
    if in_detail:
        for key, value in summary["label_stats"].iteritems():
            value["precision"] = value["right_cnt"] / value["pred_cnt"] if value["pred_cnt"] else 0
            value["recall"] = value["right_cnt"] / value["total_cnt"] if value["total_cnt"] else 0
    return summary


def multi_label_summary_to_string(summary, index_to_class, full_output=False):
    """
    Convert multi-label summary to read-to-print string
    :param summary: multi-label summary
    :param index_to_class: index to class name dictionary
    :param full_output: output all the labels or not
    :return: ready-to-print string
    """
    out_str = ""

    ins_accuracy = summary["ins_right_cnt"] / summary["ins_total_cnt"] if summary["ins_total_cnt"] else 0
    label_accuracy = summary["label_right_cnt"] / summary["label_total_cnt"] if summary["label_total_cnt"] else 0
    out_str += "Instance Accuracy {0:.4f}({1}/{2}), Label Accuracy {3:.4f}({4}/{5})\n".format(
        ins_accuracy, summary["ins_right_cnt"], summary["ins_total_cnt"],
        label_accuracy, summary["label_right_cnt"], summary["label_total_cnt"]
    )

    posi_precision = summary["posi_label_right_cnt"] / summary["posi_label_pred_cnt"] if summary["posi_label_pred_cnt"] else 0
    posi_recall = summary["posi_label_right_cnt"] / summary["posi_label_total_cnt"] if summary["posi_label_total_cnt"] else 0
    out_str += "Positive Precision {0:.4f}({1}/{2}), Recall {3:.4f}({4}/{5}), F1 {6:.4f}\n".format(
        posi_precision, summary["posi_label_right_cnt"], summary["posi_label_pred_cnt"],
        posi_recall, summary["posi_label_right_cnt"], summary["posi_label_total_cnt"],
        cal_F1(posi_precision, posi_recall)
    )

    macro_posi_precision = 0
    macro_posi_recall = 0
    cnt = 0
    for class_index in summary["label_stats"]:
        if summary["label_stats"][class_index]["total_cnt"] > 0:
            cnt += 1
        if not full_output and summary["label_stats"][class_index]["pred_cnt"] == 0 \
                and summary["label_stats"][class_index]["total_cnt"] == 0:
            continue
        macro_posi_precision += safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"])
        macro_posi_recall += safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"])
    if cnt == 0:
        cnt = 1
    macro_posi_precision /= cnt
    macro_posi_recall /= cnt
    macro_F1 = cal_F1(macro_posi_precision, macro_posi_recall)
    out_str += "Macro Positive Precision {0:.4f}, Macro Recall {1:.4f}, Macro F1 {2:.4f}, Label Count {3}\n".format(
        macro_posi_precision, macro_posi_recall, macro_F1, cnt
    )

    for class_index in summary["label_stats"]:
        if not full_output and summary["label_stats"][class_index]["pred_cnt"] == 0 \
                and summary["label_stats"][class_index]["total_cnt"] == 0:
            continue
        temp_precision = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"])
        temp_recall = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"])
        out_str += "\t{1} {0}: Precision {2:.4f} ({5}/{6}), Recall {3:.4f} ({7}/{8}), F1 {4:.4f}\n".format(
            index_to_class[class_index], class_index, temp_precision, temp_recall, cal_F1(temp_precision, temp_recall),
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"],
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"],
        )
    return out_str.strip()


def summary_to_string(summary, index_to_class):
    """
    Convert multi-class summary to ready-to-print string
    :param summary:     summary dictionary
    :param index_to_class:  index to class name dictionary
    """
    out_str = ""

    accuracy = safe_divide(summary["right_cnt"], summary["total_cnt"])
    precision = safe_divide(summary["posi_right_cnt"], summary["posi_pred_cnt"])
    recall = safe_divide(summary["posi_right_cnt"], summary["posi_total_cnt"])
    out_str += "Accuracy {0:.4f}({4}/{5}), Precision {1:.4f}({6}/{7}), Recall {2:.4f}({8}/{9}), F1 {3:.4f}\n".format(
        accuracy, precision, recall, cal_F1(precision, recall),
        summary["right_cnt"], summary["total_cnt"],
        summary["posi_right_cnt"], summary["posi_pred_cnt"],
        summary["posi_right_cnt"], summary["posi_total_cnt"],
    )

    for class_index in summary["label_stats"]:
        temp_precision = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"])
        temp_recall = safe_divide(summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"])
        out_str += "\t{1} {0}: Precision {2:.4f} ({5}/{6}), Recall {3:.4f} ({7}/{8}), F1 {4:.4f}\n".format(
            index_to_class[class_index], class_index, temp_precision, temp_recall, cal_F1(temp_precision, temp_recall),
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["pred_cnt"],
            summary["label_stats"][class_index]["right_cnt"], summary["label_stats"][class_index]["total_cnt"],
        )
    return out_str.strip()


def dump_loss_history(loss_history, dump_path):
    """
    Dump loss history dictionary to file
    :param loss_history: loss history dictionary
    :param dump_path: output file path
    :return:
    """
    with open(dump_path, "w") as outfile:
        sorted_loss = sorted(loss_history.iteritems(), key=lambda tuple: tuple[0], reverse=False)
        for key, value in sorted_loss:
            outfile.writelines("{0}\t{1}\n".format(key, value))


def load_pr_data(inpath, precision_col, recall_col, splitter=None):
    """
    Load precision and recall data from file
    :param inpath: input file path
    :param precision_col: precision column
    :param recall_col: recall column
    :param splitter: splitter character (default None, use default splitter)
    :return:
    """
    precision_list = []
    recall_list = []
    with open(inpath, "r") as infile:
        for line in infile:
            if splitter is None:
                line_split = line.strip().split()
            else:
                line_split = line.strip().split(splitter)
            if len(line_split) == 0:
                continue
            precision_list.append(float(line_split[precision_col]))
            recall_list.append(float(line_split[recall_col]))
    return precision_list, recall_list
