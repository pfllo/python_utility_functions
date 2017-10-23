#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def range_overlap(range_1, range_2):
    """
    Judge if two range overlaps
    :param range_1: first range (format: [start, end])
    :param range_2: second range (format: [start, end])
    :return:    True if overlaps, False otherwise
    """
    if range_1[0] <= range_2[0] <= range_1[1]:
        return True
    if range_2[0] <= range_1[0] <= range_2[1]:
        return True
    return False


def safe_divide(dividend, divisor):
    """
    Behave like normal division when divisor is non-zero.
    Return 0 when divisor is zero
    :param dividend:    the number to be divided
    :param divisor:     the number by which the dividend is divided
    :return:
    """
    return dividend / divisor if divisor else 0


def cal_F1(precision, recall):
    """
    Calculate F1 from precision and recall
    :param precision:   precision
    :param recall:  recall
    :return:
    """
    return safe_divide(2 * precision * recall, (precision + recall))

