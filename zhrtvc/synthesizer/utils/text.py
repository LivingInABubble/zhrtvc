#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/18
"""
"""
from phkit.chinese import text_to_sequence, sequence_to_text

if __name__ == "__main__":
    print(__file__)
    text = "ka3 er3 pu3 pei2 wai4 sun1 wan2 hua2 ti1 . "
    out = text_to_sequence(text, cleaner_names='pinyin')
    print(out)
    out = sequence_to_text(out)
    print(out)
