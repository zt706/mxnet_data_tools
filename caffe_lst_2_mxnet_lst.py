# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

res = re.compile('([0-9]+)$')

def read_caffe_lst(path, out, max_label):
    f_out = open(out, 'w')
    with open(path, 'r') as f:
        lines = f.readlines()
        idx = 1
        for l in lines:
            m = res.search(l)
            #print m.groups()
            label = m.groups()[0]
            if int(label) > max_label:
                continue
            #name = l[: 0-len(label)-2].decode('utf-8')
            name = l[: 0-len(label)-2]
            '''
            print name
            print open(name)
            print label, len(label)
            break
            '''
            line_info = '%d\t' % idx
            line_info += '%f\t' % float(label)
            line_info += '%s\n' % name
            print(line_info)
            f_out.write(line_info)
            idx += 1



if __name__ == '__main__':
    #max_label = 100000000
    #caff_lst_path = './test.lst'
    #mxnet_lst_path = './mxnet_test.lst'
    parser = argparse.ArgumentParser(description="convert cafft lst to mxnet lst")
    parser.add_argument('--caffe_lst', help='caffe lst path', default="test.lst", type=str)
    parser.add_argument('--mxnet_lst', help='mxnet lst path', default="mxnet_test.lst", type=str)
    parser.add_argument('--max_label', help='max label', default=100000000, type=int)
    args = parser.parse_args()

    read_caffe_lst(args.caffe_lst, args.mxnet_lst, args.max_label)

    #python /home/zhouji/external_libs/mxnet_for_ssd/tools/im2rec.py --resize 480 --quality 90 --num-thread 4 mxnet_test /
