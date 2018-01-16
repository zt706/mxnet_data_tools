# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import random

def tf_2_mxnet_list(tf_lst, out_lst, is_blance, blance_thread):
    cls_2_path = {}
    cls_2_num = {}
    with open(tf_lst) as f:
        print('begin readlines')
        lines = f.readlines()
        print('end readlines')
        for idx, l in enumerate(lines):
            if idx % 10000 == 0:
                print('split %d' % idx)
            info = l.split('\t')
            path = info[2]
            cls = info[1]
            cls_2_num[cls] = 0
            cls_2_path[cls] = []

        for idx, l in enumerate(lines):
            if idx % 10000 == 0:
                print('split 2 %d' %idx)
            info = l.split('\t')
            path = info[2]
            cls = info[1]

            cls_2_num[cls] +=1
            cls_2_path[cls].append(path)

    print('finish read data')

    if is_blance:
        idx = 0
        # blance_thread = 40
        for k, n in cls_2_num.items():
            if n > blance_thread:
                print("[statistics cls_num > %d] cls_id:%s, num = %d" % (blance_thread, k, n))
                p_set = set(cls_2_path[k])
                cls_2_path[k] = [p for p in p_set]
            elif n <= blance_thread:
                print("[statistics cls_num < %d] cls_id:%s, num = %d " % (blance_thread, k, n))
                for i in range(blance_thread - n):
                    pick = random.randint(0, n-1)
                    cls_2_path[k].append(cls_2_path[k][pick])
            idx += 1
            if idx % 10000 == 0:
                print('%d imgs append end' % idx)

    # just do statistics
    total_img = 0 
    for cls_id, img_path_list in cls_2_path.items():
        l = len(img_path_list)
        print("[statistics blance] cls_id:%s, img_num: %d" % (cls_id, l))
        total_img += l
    print("[statistics blance] avg_num_per_cls: %d img per cls" % (total_img/len(cls_2_path.keys())))
    k_l = cls_2_path.keys()
    print("[statistics blance] max_cls_id: %d, min_cls_id: %d, total_cls_id: %d" % \
            (max([int(float(k))for k in k_l]), min([int(float(k))for k in k_l]), len(k_l)))

    print('random select')
    all_lines = []
    for k, paths in cls_2_path.items():
        for p in paths:
            all_lines.append((k, p))

    random.seed(100)
    random.shuffle(all_lines)

    print('suffle')
    # continue label
    cls_2_continue_cls = {}
    all_cls = [float(i) for i in cls_2_path.keys()]
    all_cls.sort()
    for idx, cls in enumerate(all_cls):
        cls_2_continue_cls[cls] = idx
        print('[continue cls] old_cls: %f, new_cls: %d' % (cls, idx))

    idx = 0
    with open(out_lst, 'w') as f:
        for l in all_lines:
            label = float(l[0])
            label_continue = cls_2_continue_cls[label] 
            idx += 1
            f.write(str(idx) + '\t' + "{:.6f}".format(label_continue) + '\t' + l[1])
            if idx % 10000 == 0:
                print('%d imgs write end' % idx)
    print('end write')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert cafft lst to mxnet lst")
    parser.add_argument('--tf_lst', help='tf lst path', default="train.lst", type=str)
    parser.add_argument('--mxnet_out_lst', help='mxnet output lst path', default="mxnet_all.lst", type=str)
    parser.add_argument('--max_label', help='max label', default=100000000, type=int)
    parser.add_argument('--is_blance', help='is blance', default=1, type=int)
    parser.add_argument('--blance_thread', help='thread of blance', default=40, type=int)
    args = parser.parse_args()

    assert not args.tf_lst == args.mxnet_out_lst
    tf_2_mxnet_list(args.tf_lst, args.mxnet_out_lst, args.is_blance, args.blance_thread)
    print('pure mxnet lst path: %s' % args.mxnet_out_lst)

    # step 1
    # python -u ./tf_2_mxnet_lst.py --tf_lst train.lst --mxnet_out_lst mxnet_all.lst --is_blance 0 --blance_thread 40 > statistics.txt
    # step 2
    #python /home/zhouji/external_libs/mxnet_for_ssd/tools/im2rec.py --resize 480 --quality 90 --num-thread 4 mxnet_all /
