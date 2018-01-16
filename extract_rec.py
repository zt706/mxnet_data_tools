# -*- coding: utf-8 -*-

import os
import sys
import mxnet as mx
import argparse

def extract_rec(idx, rec, max_label, out_idx, out_rec, test_rate, out_test_idx, out_test_rec):
    all_old_idx = []
    with open(idx, 'r') as f:
        lines = f.readlines()
        for l in lines:
            i = l.split('\t')[0]
            #print i
            all_old_idx.append(i)
    record = mx.recordio.MXIndexedRecordIO(idx, rec, 'r')
    record_out = mx.recordio.MXIndexedRecordIO(out_idx, out_rec, 'w')
            
    out_j = 1
    for i,j in enumerate(all_old_idx):
        info = record.read_idx(int(j))
        header, s = mx.recordio.unpack(info)
        if int(header.label) > max_label:
            continue
        if test_rate > 0. and i % (1/test_rate) == 0:
            continue
        print 'selected number: ', i, 'label: ', header.label
        record_out.write_idx(out_j, info)
        out_j += 1
    record_out.close()

    if test_rate > 0.:
        record.reset()
        record_out_test = mx.recordio.MXIndexedRecordIO(out_test_idx, out_test_rec, 'w')
        out_j = 1
        for i, j in enumerate(all_old_idx):
            info = record.read_idx(int(j))
            header, s = mx.recordio.unpack(info)
            if int(header.label) > max_label:
                continue
            if i % (1/test_rate) != 0:
                continue
            print 'selected number: ', i, 'test label: ', header.label
            record_out_test.write_idx(out_j, info)
            out_j += 1
        record_out_test.close()

if __name__ == '__main__':
    #idx = 'mxnet_test.idx'
    #rec = 'mxnet_test.rec'
    #max_label = 100
    #out_idx = 'mxnet_test_out.idx'
    #out_rec = 'mxnet_test_out.rec'
    parser = argparse.ArgumentParser(description="extrace mxnet rec")
    parser.add_argument('--idx', help='input idx', default="mxnet_train_all.idx", type=str)
    parser.add_argument('--rec', help='input rec', default="mxnet_train_all.rec", type=str)
    parser.add_argument('--max_label', help='max label', default=100, type=int)
    parser.add_argument('--test_rate', help='test sample number', default=0., type=float)
    parser.add_argument('--out_idx', help='output idx', default="mxnet_train_out.idx", type=str)
    parser.add_argument('--out_rec', help='output rec', default="mxnet_train_out.rec", type=str)
    parser.add_argument('--out_test_idx', help='output test idx', default="mxnet_test_out.idx", type=str)
    parser.add_argument('--out_test_rec', help='output test rec', default="mxnet_test_out.rec", type=str)
    args = parser.parse_args()

    extract_rec(args.idx, args.rec, args.max_label, args.out_idx, args.out_rec, args.test_rate, args.out_test_idx, args.out_test_rec)
