# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import random

def get_train_eval_list(train_eval_lst, train_lst, eval_lst, eval_percent, is_blance, blance_thread):
    cls_2_path = {}
    cls_2_num = {}
    with open(train_eval_lst) as f:
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

    print('begin pick data')
    cls_2_path_pick_train = {}
    cls_2_num_pick_train = {}
    cls_2_path_pick_eval = {}
    cls_2_num_pick_eval = {}
    total_img = 0 
    for cls_id, img_path_list in cls_2_path.items():
        l = len(img_path_list)
        total_img += l
        cls_2_path_pick_train[cls_id] = []
        cls_2_num_pick_train[cls_id] = 0
        cls_2_path_pick_eval[cls_id] = []
        cls_2_num_pick_eval[cls_id] = 0
    print ('total img: %d' % total_img)
    pick_percent = float(eval_percent)

    idx_train = 0
    idx_eval = 0
    for cls_id, img_path_list in cls_2_path.items():
        l = len(img_path_list)
        pick_num_eval = int(pick_percent * l)
        #pick_idx = random.randint(0, l-1)
        pick_idx_eval = random.sample(range(0, l), pick_num_eval)
        pick_idx_train = list(set(range(0, l)) - set(pick_idx_eval))

        for i in pick_idx_train:
            cls_2_path_pick_train[cls_id].append(cls_2_path[cls_id][i])
            cls_2_num_pick_train[cls_id] += 1
            idx_train += 1
            if idx_train % 10000 == 0:
                print('%d pick train imgs append end' % idx_train)
        for i in pick_idx_eval:
            cls_2_path_pick_eval[cls_id].append(cls_2_path[cls_id][i])
            cls_2_num_pick_eval[cls_id] += 1
            idx_eval += 1
            if idx_eval % 10000 == 0:
                print('%d pick eval imgs append end' % idx_eval)

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
    
    print('random select')
    all_lines_train = []
    for k, paths in cls_2_path_pick_train.items():
        for p in paths:
            all_lines_train.append((k, p))

    random.seed(100)
    random.shuffle(all_lines_train)

    all_lines_eval = []
    for k, paths in cls_2_path_pick_eval.items():
        for p in paths:
            all_lines_eval.append((k, p))

    random.seed(100)
    random.shuffle(all_lines_eval)

    print('suffle')
    """
    # continue label
    cls_2_continue_cls = {}
    all_cls = [float(i) for i in cls_2_path_pick.keys()]
    all_cls.sort()
    for idx, cls in enumerate(all_cls):
        cls_2_continue_cls[cls] = idx
        print('[continue cls] old_cls: %f, new_cls: %d' % (cls, idx))
    """

    idx = 0
    with open(train_lst, 'w') as f:
        for l in all_lines_train:
            label = float(l[0])
            #label_continue = cls_2_continue_cls[label] 
            idx += 1
            f.write(str(idx) + '\t' + "{:.6f}".format(label) + '\t' + l[1])
            if idx % 10000 == 0:
                print('%d train imgs write end' % idx)
    print('train lst end write')

    idx = 0
    with open(eval_lst, 'w') as f:
        for l in all_lines_eval:
            label = float(l[0])
            #label_continue = cls_2_continue_cls[label] 
            cls_2_path_pick_eval
            idx += 1
            f.write(str(idx) + '\t' + "{:.6f}".format(label) + '\t' + l[1])
            if idx % 10000 == 0:
                print('%d eval imgs write end' % idx)
    print('eval lst end write')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert cafft lst to mxnet lst")
    parser.add_argument('--eval_percent', help='total need pick num', default=0.1, type=float)
    parser.add_argument('--train_eval_lst', help='train and eval lst path', default="train_eval_mx.lst", type=str)
    parser.add_argument('--train_lst', help='mxnet picked train output lst path', default="train_mx.lst", type=str)
    parser.add_argument('--eval_lst', help='mxnet picked eval output lst path', default="eval_mx.lst", type=str)
    parser.add_argument('--is_blance', help='is blance', default=0, type=int)
    parser.add_argument('--blance_thread', help='thread of blance', default=40, type=int)
    args = parser.parse_args()

    assert not args.train_eval_lst == args.train_lst 
    assert not args.train_eval_lst == args.eval_lst 
    get_train_eval_list(args.train_eval_lst, args.train_lst, args.eval_lst, args.eval_percent, args.is_blance, args.blance_thread)
    print('picked %f eval img, pick lst path: %s' % (args.eval_percent, args.train_eval_lst))

    # step 1
    # python -u ./get_min_data.py --eval_percent 0.1 --train_eval_lst train_eval_mx.lst --train_lst train_mx.lst --eval_lst eval_mx.lst --is_blance 0 --blance_thread 40 > statistics.txt
    # step 2
    #python /home/zhouji/external_libs/mxnet_for_ssd/tools/im2rec.py --resize 480 --quality 90 --num-thread 4 train_mx /
