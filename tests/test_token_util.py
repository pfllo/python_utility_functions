#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import unittest
import logging
from collections import Counter

from utility import token_util


class TestFileReadingFunctions(unittest.TestCase):
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        self.one_word_per_line_path = os.path.join(self.data_dir, "one_word_per_line.txt")
        self.one_sent_per_line_path = os.path.join(self.data_dir, "one_sent_per_line.txt")
        self.token2id_path = os.path.join(self.data_dir, "token2id.txt")
        self.word_cnt_path_list = [self.one_sent_per_line_path, self.one_word_per_line_path]

        self.logger = logging.getLogger("ReadingFunctions Test Logger")

    def test_token_cnt(self):
        one_word_per_line_counter = Counter({"a_1": 1, "b_2": 2, "c_3": 3, "d_4": 4})
        one_sent_per_line_counter = Counter({"a_1": 1, "b_2": 2, "c_3": 3, "d_4": 4, "e_5": 5, "f_6": 6})

        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path], separator=None, workers=1, parallel_mode="size")
        self.assertEqual(c, one_word_per_line_counter)
        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path], separator=None, workers=3, parallel_mode="size")
        self.assertEqual(c, one_word_per_line_counter)

        c = token_util.gen_token_cnt_from_file([self.one_sent_per_line_path], separator=None, workers=1, parallel_mode="size")
        self.assertEqual(c, one_sent_per_line_counter)
        c = token_util.gen_token_cnt_from_file([self.one_sent_per_line_path], separator=None, workers=3, parallel_mode="size")
        self.assertEqual(c, one_sent_per_line_counter)

        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path, self.one_sent_per_line_path], separator=None, workers=1, parallel_mode="size")
        self.assertEqual(c, one_word_per_line_counter + one_sent_per_line_counter)
        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path, self.one_sent_per_line_path], separator=None, workers=3, parallel_mode="size")
        self.assertEqual(c, one_word_per_line_counter + one_sent_per_line_counter)

        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path, self.one_sent_per_line_path], separator=None, workers=1, parallel_mode="file")
        self.assertEqual(c, one_word_per_line_counter + one_sent_per_line_counter)
        c = token_util.gen_token_cnt_from_file([self.one_word_per_line_path, self.one_sent_per_line_path], separator=None, workers=3, parallel_mode="file")
        self.assertEqual(c, one_word_per_line_counter + one_sent_per_line_counter)

    def test_gen_token_id_from_file(self):
        one_word_per_line_counter = Counter({"a_1": 1, "b_2": 2, "c_3": 3, "d_4": 4})
        one_sent_per_line_counter = Counter({"a_1": 1, "b_2": 2, "c_3": 3, "d_4": 4, "e_5": 5, "f_6": 6})

        res_list = token_util.gen_token_id_from_file(one_sent_per_line_counter, min_cnt=-1, max_size=-1, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5", "d_4", "c_3", "b_2", "a_1"])
        res_list = token_util.gen_token_id_from_file(one_sent_per_line_counter, min_cnt=2, max_size=-1, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5", "d_4", "c_3"])
        res_list = token_util.gen_token_id_from_file(one_sent_per_line_counter, min_cnt=-1, max_size=2, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5"])

        res_list = token_util.gen_token_id_from_file([self.one_sent_per_line_path], min_cnt=-1, max_size=-1, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5", "d_4", "c_3", "b_2", "a_1"])
        res_list = token_util.gen_token_id_from_file([self.one_sent_per_line_path], min_cnt=2, max_size=-1, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5", "d_4", "c_3"])
        res_list = token_util.gen_token_id_from_file([self.one_sent_per_line_path], min_cnt=-1, max_size=2, separator=None)
        self.assertEqual(res_list, ["f_6", "e_5"])

        res_list = token_util.gen_token_id_from_file([self.one_sent_per_line_path, self.one_word_per_line_path], min_cnt=2, max_size=-1, separator=None)
        self.assertAlmostEqual(res_list, ["d_4", "f_6", "c_3", "e_5", "b_2"], delta=2)
        res_list = token_util.gen_token_id_from_file([self.one_sent_per_line_path, self.one_word_per_line_path], min_cnt=-1, max_size=3, separator=None)
        self.assertAlmostEqual(res_list, ["d_4", "f_6", "c_3"], delta=2)

    def test_load_token_id(self):
        token2id, id2token = token_util.load_token_id(self.token2id_path)
        self.assertEqual(token2id, {"a_0": 0, "b_1": 1, "c_2": 2, "d_3": 3, "UNK": 4})
        self.assertEqual(id2token, ["a_0", "b_1", "c_2", "d_3", "UNK"])


if __name__ == "__main__":
    unittest.main()

