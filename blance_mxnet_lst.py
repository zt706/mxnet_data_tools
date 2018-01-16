# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import random
def blance_mxnet_list(mxnet_lst, out_lst, is_blance):
    cls_2_path = {}
    cls_2_num = {}
    with open(mxnet_lst) as f:
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
        for k, n in cls_2_num.items():
            if n > 40:
                print("[statistics cls_num > 40] cls_id:%s, num = %d" % (k, n))
                p_set = set(cls_2_path[k])
                cls_2_path[k] = [p for p in p_set]
            elif n <= 40:
                print("[statistics cls_num < 40] cls_id:%s, num = %d " % (k, n))
                for i in range(40-n):
                    pick = random.randint(0,n-1)
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
    parser.add_argument('--mxnet_lst', help='mxnet lst path', default="train.lst", type=str)
    parser.add_argument('--mxnet_out_lst', help='mxnet output lst path', default="mxnet_all.lst", type=str)
    parser.add_argument('--max_label', help='max label', default=100000000, type=int)
    parser.add_argument('--is_blance', help='is blance', default=1, type=int)
    args = parser.parse_args()
    assert not args.mxnet_lst == args.mxnet_out_lst
    blance_mxnet_list(args.mxnet_lst, args.mxnet_out_lst, args.is_blance)
    print('pure mxnet lst path: %s' % args.mxnet_out_lst)
    # step 1
    # python -u /home/zhouji/tp_server/tpdetect/huge_cls/tools/blance_mxnet_lst.py --mxnet_lst train.lst --mxnet_out_lst mxnet_all.lst --is_blance 1 > blance_statistics.txt
    # step 2
    #python /home/zhouji/external_libs/mxnet_for_ssd/tools/im2rec.py --resize 480 --quality 90 --num-thread 4 mxnet_all /
