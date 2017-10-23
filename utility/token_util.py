#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
from collections import Counter
from functools import reduce, partial
from multiprocessing import Pool, cpu_count


def gen_token_cnt_from_file(file_list, separator=None, workers=1, parallel_mode="size"):
    """
    Generate token count from a list of files
    :param file_list: a list of file paths
    :param separator: the separator used to get each token
    :param workers: number of workers
    :param parallel_mode: "size" means parallel by size, "file" means parallel by file
    :return: a Counter object {word: cnt}
    """
    assert workers > 0
    if workers == 1:
        c = Counter()
        for inpath in file_list:
            with open(inpath, "rb") as infile:
                for line in infile:
                    tokens = line.decode("utf8").strip().split(separator)
                    c.update(tokens)
        return c
    elif parallel_mode == "size":
        res_list = []
        for inpath in file_list:
            pool = Pool(workers)
            filesize = os.path.getsize(inpath)
            for i in range(workers):
                p1 = filesize * i // workers
                p2 = filesize * (i+1) // workers
                args = [inpath, p1, p2, separator]
                res = pool.apply_async(func=_count_file_segment, args=args)
                res_list.append(res)
            pool.close()
            pool.join()

        res_c = Counter()
        res_c.update(reduce(operator.add, [r.get() for r in res_list]))
        return res_c
    elif parallel_mode == "file":
        pool = Pool(workers)
        res_list = pool.map(partial(_count_file, separator=separator), file_list)
        res_c = Counter()
        res_c.update(reduce(operator.add, res_list))
        return res_c
    else:
        raise ValueError("Unsupported parallel mode: {0}".format(parallel_mode))


def gen_token_id_from_file(in_data, min_cnt=-1, max_size=-1, separator=None):
    """
    Generate token2id mapping from a list of files
    :param in_data: a list of file paths OR a Counter of words
    :param min_cnt: if specified, only tokens with more appearances than min_cnt will be included (exclusive)
    :param max_size: if specified, only max_size number of tokens will be included
    :param separator: the separator used to get each token
    :return: id2token list (id from 0)
    """
    if type(in_data) is Counter:
        pass
    elif type(in_data) is list:
        assert type(in_data[0]) is str
        in_data = gen_token_cnt_from_file(in_data, separator=separator, workers=2, parallel_mode="size")
    else:
        raise TypeError("Invalid in_data type: {0}".format(type(in_data)))
    sorted_token_cnt = sorted(in_data.items(), key=lambda t: t[1], reverse=True)
    if max_size >= 0:
        sorted_token_cnt = sorted_token_cnt[:max_size]
    id2token = []
    for t in sorted_token_cnt:
        if t[1] > min_cnt:
            id2token.append(t[0])
    return id2token


def load_token_id(inpath):
    """
    Load token2id from file
    :param inpath:
    :return: token2id, id2token
    """
    with open(inpath, "rb") as infile:
        token2id_tuple_list = [line.decode("utf8").strip().split() for line in infile.readlines()]
        token2id = {t[0]: int(t[1]) for t in token2id_tuple_list}
        id2token = [t[0] for t in token2id_tuple_list]
        return token2id, id2token


def _count_file_segment(fn, p1, p2, separator):
    """
    Count tokens in file fn, from position p1 to p2
    :param fn: input file path
    :param p1: start byte position (excluded)
    :param p2: end byte position (excluded)
    :param separator:
    :return: Counter object
    """
    c = Counter()
    with open(fn, "rb") as f:
        if p1:
            f.seek(p1-1)
            while b'\n' not in f.read(1):   # find next line
                pass
        while 1:
            line = f.readline().decode("utf8").strip()
            c.update(line.split(separator))
            pos = f.tell()
            if pos >= p2:
                return c


def _count_file(fn, separator):
    """
    Count tokens in a single file
    :param fn: input file path
    :param separator:
    :return: Count object
    """
    c = Counter()
    with open(fn, "rb") as f:
        for line in f:
            c.update(line.decode("utf8").strip().split(separator))
    return c

