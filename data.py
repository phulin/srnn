#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 19:42:16 2017

@author: ldy
"""

from os.path import join

from dataset import DatasetFromFolder


def get_training_set():

    return DatasetFromFolder(join("train", "orig"))