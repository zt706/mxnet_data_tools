# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse
import random

def blance_mxnet_list(in_lst, out_lst, pick_cls_num):
    cls_2_path = {}
    cls_2_num = {}
    with open(in_lst) as f:
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

    # just do statistics
    total_img_ae_10 = 0
    total_img_be_10 = 0
    cls_2_path_ae_10 = {}           # 保存图片数目 >= 10 的
    cls_2_path_be_10 = {}           # 保存图片数目 <= 10 的
    for cls_id, img_path_list in cls_2_path.items():
        l = len(img_path_list)
        if l >= 10:
            cls_2_path_ae_10[cls_id] = img_path_list
            total_img_ae_10 += l
            print("[statistics need pick cls] cls_id:%s, img_num: %d" % (cls_id, l))
        else:
            cls_2_path_be_10[cls_id] = img_path_list
            total_img_be_10 += l
            print("[statistics not need cls] cls_id:%s, img_num: %d" % (cls_id, l))

    print("[statistics extract cls >= 10 iid] avg_num_per_cls: %d img per cls" % (total_img_ae_10/len(cls_2_path_ae_10.keys())))
    print("[statistics extract cls <= 10 iid] avg_num_per_cls: %d img per cls" % (total_img_be_10/len(cls_2_path_be_10.keys())))

    #k_l = cls_2_path.keys()
    k_l = cls_2_path_ae_10.keys()
    max_cls = max([int(float(k))for k in k_l])
    min_cls = min([int(float(k))for k in k_l])
    total_cls = len(k_l)
    print("[statistics extract cls iid] max_cls_id: %d, min_cls_id: %d, total_cls_id: %d" % \
            #(max([int(float(k))for k in k_l]), min([int(float(k))for k in k_l]), len(k_l)))
            (max_cls, min_cls, total_cls))

    print('random select cls ')
    all_lines = []
    for k, paths in cls_2_path_ae_10.items():
        for p in paths:
            all_lines.append((k, p))

    random.seed(100)
    random.shuffle(all_lines)

    print('suffle end')

    random.seed(101)
    assert len(k_l) > pick_cls_num 
    cls_2_path_pick = {}
    pick_cls_list = random.sample(k_l, pick_cls_num)
    for cls in pick_cls_list:
        cls_2_path_pick[cls] = []
    for cls in pick_cls_list:
        cls_2_path_pick[cls] = cls_2_path_ae_10[str("{:.6f}".format(float(cls)))]

    # continue picked label
    cls_2_continue_cls = {}
    all_cls = [float(i) for i in cls_2_path_pick.keys()]
    all_cls.sort()
    for idx, cls in enumerate(all_cls):
        cls_2_continue_cls[cls] = idx
        print('[continue cls] old_cls: %f, new_cls: %d' % (cls, idx))

    idx = 0
    with open(out_lst, 'w') as f:
        for l in all_lines:
            label = float(l[0])
            label_continue = cls_2_continue_cls.get(label) 
            if label_continue is None:
                continue
            idx += 1
            f.write(str(idx) + '\t' + "{:.6f}".format(label_continue) + '\t' + l[1])
            if idx % 10000 == 0:
                print('%d imgs write end' % idx)
    print('end write')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert cafft lst to mxnet lst")
    parser.add_argument('--in_lst', help='mxnet lst path', default="mxnet_all.lst", type=str)
    parser.add_argument('--out_lst', help='picked mxnet output lst path', default="mxnet_pick.lst", type=str)
    parser.add_argument('--pick_cls_num', help='cls number of pick', default=16000, type=int)
    args = parser.parse_args()

    assert not args.in_lst == args.out_lst
    blance_mxnet_list(args.in_lst, args.out_lst, args.pick_cls_num)
    print('picked mxnet lst path: %s' % args.out_lst)

    # step 1
    # python -u ./extract_mxnet_lst.py --in_lst train.lst --out_lst mxnet_pick.lst --pick_cls_num 16000 > statistics_pick.txt
    # step 2
    # python -u /home/zhouji/tp_server/tpdetect/huge_cls/tools/blance_mxnet_lst.py --mxnet_lst mxnet_pick.lst --mxnet_out_lst mxnet_all.lst --is_blance 1 --blance_thread 40 > statistics_blance.txt
    # step 3
    # python -u ./split_train_eval.py --eval_percent 0.1 --train_eval_lst mxnet_all.lst --train_lst train_mx.lst --eval_lst eval_mx.lst --is_blance 0 --blance_thread 40 > statistics_split_train_val.txt
    # step 4
    #python /home/zhouji/external_libs/mxnet_for_ssd/tools/im2rec.py --resize 480 --quality 90 --num-thread 4 mxnet_all /
