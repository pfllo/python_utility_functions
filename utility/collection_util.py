#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import random
import six
import numpy as np
from collections import defaultdict


def list_to_dict(in_list, default_value=True):
    """
    Generate a dictionary using values in the list as keys
    :param in_list: input list
    :param default_value: default value for the generated dictionary
    :return: dictionary using values in the list as keys
    """
    res_dict = {}
    for val in in_list:
        res_dict[val] = default_value
    return res_dict


def safe_initialize_key(dic, key, value):
    """
    If dic doesn't contain key, then dic[key]=value
    :param dic:     input dictionary
    :param key:     key to initialize
    :param value:   initial value
    :return:
    """
    if key not in dic:
        dic[key] = value


def safe_add_one(dic, key):
    """
    If dic contains key, dic[key]=0, else dic[key]+=1
    :param dic: input dicitonary
    :param key: key to add one
    :return:
    """
    safe_initialize_key(dic, key, 0)
    dic[key] += 1


def first_element_of_dic(input_dic):
    """
    Return the first element of a dictionary
    :param input_dic:   input dictionary
    :return:    first element, (key, value)
    """
    return six.next(six.iteritems(input_dic))


def randomize_arrays(*array_list):
    """
    Randomize multiple arrays simultaneously with the same index array
    :param array_list:  a list of arrays to randomize
    :return:    randomized array list, index array
    """
    length = len(array_list[0])
    index_array = range(length)
    random.shuffle(index_array)

    res_array_list = []
    for array in array_list:
        new_array = [array[index_array[i]] for i in range(length)]
        res_array_list.append(new_array)

    return res_array_list, index_array


def split_array_with_ratio(array, ratio_array, allow_empty_bin=True):
    """
    Split an array with given ratio.
    :param array:   input array
    :param ratio_array:   array of ratio (e.g [3, 2, 1])
    :param allow_empty_bin: if False, will evenly split the array if empty bin exists
    :return:    list of split array
    """
    length = len(array)
    ratio_sum = np.sum(ratio_array)
    split_point = []
    cumulative_ratio = 0
    for ratio in ratio_array:
        cumulative_ratio += ratio
        split_point.append(int(math.ceil(length * cumulative_ratio / ratio_sum)))

    res_array = []
    start_point = 0
    empty_bin = False
    for end_point in split_point:
        res_array.append(array[start_point: end_point])
        if start_point == end_point:
            empty_bin = True
        start_point = end_point

    if not allow_empty_bin and empty_bin:
        bin_size = int(math.floor(len(array) / len(ratio_array)))
        if bin_size == 0:       # array is too small
            return res_array
        start_point = 0
        res_array = []
        for i in range(len(ratio_array)):
            end_point = len(array) if i == len(ratio_array)-1 else start_point + bin_size
            res_array.append(array[start_point: end_point])
            start_point = end_point

    return res_array


def pad_list(_list, length, padding):
    """
    Pad or truncate a list to required length (operate upon the original list)
    :param _list: given list
    :param length: target length
    :param padding: padding element
    :return:
    """
    if len(_list) > length:
        _list = _list[:length]
    elif len(_list) < length:
        _list += [padding for i in range(length - len(_list))]
    return _list


def eliminate_padding(array, padding_index_array):
    """
    Eleminate padding elements in an array given padding data indices.
    :param array:   can be python array or numpy ndarray
    :param padding_index_array: array of padding data indices
    :return:    cleaned array
    """
    if type(array) == list:     # python array
        res_array = []
        padding_index_dic = {}
        for padding_index in padding_index_array:
            padding_index_dic[padding_index] = True
        for i in range(len(array)):
            if i not in padding_index_dic:
                res_array.append(array[i])
        return res_array
    elif type(array == np.ndarray):     # numpy ndarray
        valid_index_array = []
        start_index = 0
        for i in range(len(padding_index_array)):
            valid_index_array += range(start_index, padding_index_array[i])
            start_index = padding_index_array[i] + 1
        valid_index_array += range(start_index, len(array))
        return array[valid_index_array]
    else:
        raise TypeError("Given input array of type {0}, list or numpy.ndarray is required.".format(type(array)))


def load_dic_from_txt(inpath, splitter="\t", key_func=None, value_func=None):
    """
    Assume each line in input file is separated by "splitter", and has either 1 or 2 columns,
    load the first column as key and second column as value (if only 1 column, value is "True")
    :param inpath:  input file path
    :param splitter: splitter used to split each line
    :param key_func: a function that takes key as input, and its output is considered as the new key
    :param value_func: a function that takes value as input, and its output is considered as the new value
    :return     dictionary (key is the first column, value is the second column (or "True" if only 1 column))
    """
    res_dic = {}
    with open(inpath, "r") as infile:
        while True:
            line = infile.readline().strip()
            if line == "":
                break
            line_split = line.split(splitter)
            key = line_split[0] if key_func is None else key_func(line_split[0])
            value = line_split[1] if value_func is None else value_func(line_split[1])
            res_dic[key] = value
    return res_dic


def load_seq_dic_from_list_file(inpath, func=None, ignore_list=[]):
    """
    Assume each line contains a single key that we want to use to build a sequential dictionary.
    e.g. Given a file containing "a\nb\nc", we will return {"a": 0, "b": 1, "c", 2} and its reverse dictionary
    :param inpath: input file path
    :param func: a function that will be applied to each line to produce the final key
    :return: two dictionaries: key_to_id and id_to_key
    """
    key_to_id = {}
    ignore_set = set(ignore_list)
    with open(inpath, "r") as infile:
        idx = 0
        for line in infile:
            line = line.strip()
            if line == "":
                continue
            key = line if func is None else func(line)
            if key in ignore_set:
                continue
            key_to_id[key] = idx
            idx += 1
    id_to_key = dic_from_key_value_reverse(key_to_id)
    return key_to_id, id_to_key


def dic_from_key_value_reverse(raw_dic, group_keys=False, allow_conflict=0):
    """
    Build a new dictionary from raw_dic, using value as key and key as value
    :param raw_dic: raw dictionary
    :param group_keys: group keys into list with identical value (if false, will raise exception when multiple
    keys share the same value)
    :param allow_conflict: 0 (don't allow conflict), 1 (use the first value when conflict occurs),
    2 (use the last value when conflict occurs)
    :return:
    """
    if group_keys:
        new_dic = defaultdict(list)
    else:
        new_dic = {}
    for key, val in raw_dic.iteritems():
        if group_keys:
            new_dic[val].append(key)
        else:
            if val in new_dic:
                if allow_conflict == 0:
                    raise KeyError("Two keys ({0}, {1}) share the same value {2}".format(new_dic[val], key, val))
                elif allow_conflict == 1:
                    continue
                elif allow_conflict == 2:
                    new_dic[val] = key
                else:
                    raise ValueError("Wrong value for parameter allow_conflict: {0}".format(allow_conflict))
            else:
                new_dic[val] = key
    return new_dic


def get_range_index(lower_bound, upper_bound, additional_symbols=[]):
    """
    Chagne a range of integers to index
    :param lower_bound: lower bound of the range
    :param upper_bound: upper bound of the range
    :param additional_symbols:  additional symbols to add
    :return: integer_to_index, index_to_integer
    """
    integer_array = range(lower_bound, upper_bound)
    integer_to_index = {}
    index_to_integer = {}
    cnt = 0
    for integer in integer_array:
        integer_to_index[integer] = cnt
        index_to_integer[cnt] = integer
        cnt += 1
    for integer in additional_symbols:
        integer_to_index[integer] = cnt
        index_to_integer[cnt] = integer
        cnt += 1
    return integer_to_index, index_to_integer


def get_index_map(in_array):
    """
    Get element_to_index and index_to_element mapping dictionary given an array of elements to map.
    :param in_array:    input array of elements to map
    :return:    element_to_index, index_to_element
    """
    ele_to_index = {}
    index_to_ele = {}
    cnt = 0
    for ele in in_array:
        ele_to_index[ele] = cnt
        index_to_ele[cnt] = ele
        cnt += 1
    return ele_to_index, index_to_ele


def change_encoding(in_obj, in_encoding="unicode", out_encoding="utf8"):
    """
    Convert an string or array of string of any encoding to another encoding
    :param in_obj:   input string or string array
    :param in_encoding:     encoding of the input
    :param out_encoding:    encoding of the output
    :return     utf-8 string array
    """
    if in_encoding == out_encoding:
        return in_obj
    if isinstance(in_obj, str):
        if in_encoding == "unicode":
            return in_obj.encode(out_encoding)
        elif out_encoding == "unicode":
            return in_obj.decode(in_encoding)
        else:
            return in_obj.decode(in_encoding).encode(out_encoding)
    elif isinstance(in_obj, list):
        if in_encoding == "unicode":
            for i in range(len(in_obj)):
                in_obj[i] = in_obj[i].encode(out_encoding)
        elif out_encoding == "unicode":
            for i in range(len(in_obj)):
                in_obj[i] = in_obj[i].decode(in_encoding)
        else:
            for i in range(len(in_obj)):
                in_obj[i] = in_obj[i].decode(in_encoding).encode(out_encoding)
        return in_obj
    else:
        raise TypeError("Expected {0} or {1} of input object, got {2} instead.".format(str, list, type(in_obj)))


def set_to_zero_one_vec(index_set, depth):
    """
    Change a index set to a 0-1 vector
    :param index_set: input index set
    :param depth: lenght of the vector
    :return: corresponding 0-1 vector
    """
    res_vec = [0 for i in range(depth)]
    for idx in index_set:
        res_vec[idx] = 1
    return res_vec

