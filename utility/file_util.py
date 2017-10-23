#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil


def get_filename_list_from_dir(in_dir, suffix="", ignore_suffix_list=[], return_dict=False):
    """
    Get filename list from a given directory
    :param in_dir: input directory
    :param suffix: file name suffixes that you would like to remove
    :param ignore_suffix_list: ignore files with these suffixes
    :param return_dict: return dictionary (if true, return a dictionary using file name as key and True as value)
    :return: list of file names
    """
    res_list = []
    for filename in os.listdir(in_dir):
        for suffix_ignored in ignore_suffix_list:
            if filename.endswith(suffix_ignored):
                continue
            filename = filename[:-len(suffix)] if suffix and filename.endswith(suffix) else filename
            res_list.append(filename)
    if return_dict:
        res_dict = {}
        for filename in res_list:
            res_dict[filename] = True
        return res_dict
    else:
        return res_list


def move_file_with_name_list(src_dir, dst_dir, filename_list):
    """
    Move a collection of files to another directory with given filename list
    :param src_dir: from directory
    :param dst_dir: to direction
    :param filename_list:
    :return:
    """
    for filename in filename_list:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.move(src_path, dst_path)


def load_str_until_empty_line(infile, sp_line_dic={}, default_mark="no_mark"):
    """
    Load the string until encounter an empty line (empty line is read off the input stream)
    If special lines in sp_line_dic values appear, output string will be marked as the corresponding key in sp_line_dic,
    otherwise it will be marked as default_mark
    :param infile:  input stream
    :param sp_line_dic: dictionary of special lines and corresponding keys
    :return:    (string, mark)
    """
    res_str = ""
    res_mark = default_mark
    while True:
        line = infile.readline().strip()
        if line == "":
            break
        res_str += line + "\n"

        if res_mark == default_mark:
            for k, v in sp_line_dic.iteritems():
                if line == v:
                    res_mark = k

    return res_str, res_mark


def add_suffix_to_file_name(file_name, suffix):
    """
    Add a suffix to a file name (can be a file path), and the extension is retained.
    :param file_name:   raw file name
    :param suffix:  suffix to add
    :return:    new file name
    """
    last_dot_position = file_name.rfind(".")
    if last_dot_position == -1:
        last_dot_position = len(file_name)
    return file_name[:last_dot_position] + suffix + file_name[last_dot_position:]


def ensure_dir_exists(dir_path, recursive=False):
    """
    If the given directory does not exist, create the directory
    :param dir_path: input directory path
    :param recursive: create directory recursively, e.g we want a/b/c/d/, and we only have a/b/, then the function will
    create a/b/c/ and a/b/c/d/
    :return:
    """
    if not os.path.isdir(dir_path):
        if recursive:
            os.makedirs(dir_path)
        else:
            os.mkdir(dir_path)
