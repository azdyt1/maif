import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from model_sasrec import SASREC
from model_fdsa import FDSA
from model_nova import NOVA
from model_dif import DIF
from model_maif import MAIF
import numpy as np
import time
from tqdm import tqdm
import random
from sklearn.metrics import roc_auc_score
import sys
import math
from utils import *

def compute_emb_similarity(embedding1, embedding2):
    embedding1 = np.reshape(embedding1, [-1])
    embedding2 = np.reshape(embedding2, [-1])
    num = np.dot(embedding1, embedding2.T)
    denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    cos = num / denom
    # cos = cos * 0.5 + 0.5
    return cos

def compute_emb_distance(embedding1, embedding2):
    embedding1 = np.reshape(embedding1, [-1])
    embedding2 = np.reshape(embedding2, [-1])
    return np.linalg.norm(embedding1 - embedding2)

def dcg_score(y_true, y_score, k=20):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=20):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def ild_compute(emb_list):
    val = 0.0
    for i in range(len(emb_list)):
        for j in range(len(emb_list)):
            if i == j:
                continue
            val += np.sqrt(np.sum(np.square(emb_list[i]-emb_list[j]), -1))
    return val / (len(emb_list)*len(emb_list)-1)

def diversity_compute(cate_list):
    val = 0.0
    for i in range(len(cate_list)):
        for j in range(i+1, len(cate_list)):
            val += cate_list[i] != cate_list[j]
    return val / (len(cate_list)*(len(cate_list)-1)/2)


input_format = {
    'num_epoch': 100,
    'batch_size': 256,
    'test_batch_size': 256,
    'user_num': 20000,
    'item_num': 30000,
    'review_count_num': 20000,
    'year_num': 20,
    'stars_num': 100,
    'city_num': 700,
    'cate_num': 2000,
    'nums_expert':1,
    'expert_units': 64,
    'final_layer_hidden_units': [[64],[32]],
    'final_layer_activation': tf.nn.relu,
    'nums_label':1,
    'dims': 16,
    'seq_dims': 16,
    'eval_topk': 20,
    'rerank_topk': 100,
    'matching_seq_max_len': 200,
    'his_long_seq_max_len': 100,
    'his_short_seq_max_len': 25,
    'lambdaConstantMMR': 0.8,
    'lambdaConstantDPP': 0.9,
    'lambdaConstantCATE': 0.01,
    'lambdaConstantPMFA': 0.5,
    'lambdaConstantPMFB': 0.5,
    'loss': 'logits_loss',
    'max_train_steps': None,
    'optimizer': tf.train.AdamOptimizer(learning_rate=0.001),
    'blocks': 1,
    'block_shape': [16],
    'heads': 1,
    'is_drop': False,
    'dropout_keep_prob': 1,
    'label_name': ['label'],
    'dense_dropout_keep_prob': 1.0,
    'patience': 5
}


user_train_matching_seq_map = {}
user_valid_matching_seq_map = {}
user_test_matching_seq_map = {}

user_train_long_seq_map = {}
user_valid_long_seq_map = {}
user_test_long_seq_map = {}

item_list = []
cate_list = []
his_item_seq_list = []
his_cate_seq_list = []
match_item_seq_list = []
match_cate_seq_list = []
label_list = []
weight_list = []
user_list = []
query_list = []

item_valid_list = []
cate_valid_list = []
his_item_seq_valid_list = []
his_cate_seq_valid_list = []
match_item_seq_valid_list = []
match_cate_seq_valid_list = []
label_valid_list = []
weight_valid_list = []
user_valid_list = []
query_valid_list = []

item_test_list = []
cate_test_list = []
his_item_seq_test_list = []
his_cate_seq_test_list = []
match_item_seq_test_list = []
match_cate_seq_test_list = []
label_test_list = []
weight_test_list = []
user_test_list = []
query_test_list = []

def load_sku_info_map(sku_info_file):
    data = np.load(sku_info_file, allow_pickle=True)
    sku_info_map = data['sku_info'].tolist()
    return sku_info_map

def load_user_info_map(user_info_file):
    data = np.load(user_info_file, allow_pickle=True)
    user_info_map = data['user_info'].tolist()
    return user_info_map

def read_train_file(file_path_train):
    print('read train file:')
    cnt = 0
    with open(file_path_train, 'r') as f:
        first = True
        for line in tqdm(f, total=3000000):
            if first == True:
                first = False
                continue
            # cnt += 1
            # if cnt >= 10000:
            #     break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            uid = int(line_split[1])
            item_id = int(line_split[2])
            his_item_seq = line_split[3]
            label = float(line_split[4])

            if his_item_seq == '' or his_item_seq == 'NULL':
                his_item_seq_padding = [0] * 25
            else:
                his_item_seq_padding = [int(x) for x in his_item_seq.split('-')] + [0] * 25
                his_item_seq_padding = his_item_seq_padding[:25]


            item_list.append(item_id)
            his_item_seq_list.append(his_item_seq_padding)
            label_list.append(label)
            user_list.append(uid)
            query_list.append(queryid)

def read_valid_file(file_path_valid):
    print('read valid file:')
    cnt = 0
    with open(file_path_valid, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            # cnt += 1
            # if cnt >= 10000:
            #     break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            uid = int(line_split[1])
            item_id = int(line_split[2])
            his_item_seq = line_split[3]
            label = float(line_split[4])

            if his_item_seq == '' or his_item_seq == 'NULL':
                his_item_seq_padding = [0] * 25
            else:
                his_item_seq_padding = [int(x) for x in his_item_seq.split('-')] + [0] * 25
                his_item_seq_padding = his_item_seq_padding[:25]

            item_valid_list.append(item_id)
            his_item_seq_valid_list.append(his_item_seq_padding)
            label_valid_list.append(label)
            user_valid_list.append(uid)
            query_valid_list.append(queryid)

def read_test_file(file_path_test):
    print('read test file:')
    cnt = 0
    with open(file_path_test, 'r') as f:
        first = True
        for line in tqdm(f, total=100000):
            if first == True:
                first = False
                continue
            # cnt += 1
            # if cnt >= 10000:
            #     break
            line_split = line.strip('\n').split('\t')
            queryid = line_split[0]
            uid = int(line_split[1])
            item_id = int(line_split[2])
            his_item_seq = line_split[3]
            label = float(line_split[4])

            if his_item_seq == '' or his_item_seq == 'NULL':
                his_item_seq_padding = [0] * 25
            else:
                his_item_seq_padding = [int(x) for x in his_item_seq.split('-')] + [0] * 25
                his_item_seq_padding = his_item_seq_padding[:25]

            item_test_list.append(item_id)
            his_item_seq_test_list.append(his_item_seq_padding)
            label_test_list.append(label)
            user_test_list.append(uid)
            query_test_list.append(queryid)


def generate_train_batch(batch_data_index, sku_info_map, user_info_map):

    item = np.reshape([item_list[i] for i in batch_data_index], [-1,1])
    cate1 = np.reshape([sku_info_map[item_list[i]]['cate1'] for i in batch_data_index], [-1,1])
    cate2 = np.reshape([sku_info_map[item_list[i]]['cate2'] for i in batch_data_index], [-1, 1])
    city = np.reshape([sku_info_map[item_list[i]]['city'] for i in batch_data_index], [-1, 1])
    review_count = np.reshape([sku_info_map[item_list[i]]['review_count'] for i in batch_data_index], [-1, 1])
    stars = np.reshape([sku_info_map[item_list[i]]['stars'] for i in batch_data_index], [-1, 1])

    his_item_seq = [his_item_seq_list[i] for i in batch_data_index]
    his_cate1_seq = []
    his_cate2_seq = []
    his_city_seq = []
    his_review_count_seq = []
    his_stars_seq = []
    for l in his_item_seq:
        his_cate1_seq.append([sku_info_map[i]['cate1'] for i in l])
        his_cate2_seq.append([sku_info_map[i]['cate2'] for i in l])
        his_city_seq.append([sku_info_map[i]['city'] for i in l])
        his_review_count_seq.append([sku_info_map[i]['review_count'] for i in l])
        his_stars_seq.append([sku_info_map[i]['stars'] for i in l])

    label = np.reshape([label_list[i] for i in batch_data_index], [-1,1])
    user = np.reshape([user_list[i] for i in batch_data_index], [-1,1])
    review_count_u = np.reshape([user_info_map[user_list[i]]['review_count'] for i in batch_data_index], [-1, 1])
    year = np.reshape([user_info_map[user_list[i]]['yelping_since'] for i in batch_data_index], [-1, 1])
    stars_u = np.reshape([user_info_map[user_list[i]]['average_stars'] for i in batch_data_index], [-1, 1])

    return [item, cate1, cate2, city, review_count, stars, his_item_seq, his_cate1_seq, his_cate2_seq, his_city_seq, his_review_count_seq, his_stars_seq, label, user, review_count_u, year, stars_u]


def generate_valid_batch(batch_size, global_index, sku_info_map, user_info_map):
    item = np.reshape(item_valid_list[global_index-batch_size:global_index], [-1, 1])
    cate1 = np.reshape([sku_info_map[item_valid_list[i]]['cate1'] for i in range(global_index-batch_size, global_index)], [-1, 1])
    cate2 = np.reshape([sku_info_map[item_valid_list[i]]['cate2'] for i in range(global_index-batch_size, global_index)], [-1, 1])
    city = np.reshape([sku_info_map[item_valid_list[i]]['city'] for i in range(global_index-batch_size, global_index)], [-1, 1])
    review_count = np.reshape([sku_info_map[item_valid_list[i]]['review_count'] for i in range(global_index-batch_size, global_index)], [-1, 1])
    stars = np.reshape([sku_info_map[item_valid_list[i]]['stars'] for i in range(global_index-batch_size, global_index)], [-1, 1])

    his_item_seq = his_item_seq_valid_list[global_index-batch_size:global_index]
    his_cate1_seq = []
    his_cate2_seq = []
    his_city_seq = []
    his_review_count_seq = []
    his_stars_seq = []
    for l in his_item_seq:
        his_cate1_seq.append([sku_info_map[i]['cate1'] for i in l])
        his_cate2_seq.append([sku_info_map[i]['cate2'] for i in l])
        his_city_seq.append([sku_info_map[i]['city'] for i in l])
        his_review_count_seq.append([sku_info_map[i]['review_count'] for i in l])
        his_stars_seq.append([sku_info_map[i]['stars'] for i in l])

    label = np.reshape(label_valid_list[global_index-batch_size:global_index], [-1, 1])
    user = np.reshape(user_valid_list[global_index-batch_size:global_index], [-1, 1])
    review_count_u = np.reshape([user_info_map[user_valid_list[i]]['review_count'] for i in range(global_index - batch_size, global_index)], [-1, 1])
    year = np.reshape(
        [user_info_map[user_valid_list[i]]['yelping_since'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])
    stars_u = np.reshape(
        [user_info_map[user_valid_list[i]]['average_stars'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])

    return [item, cate1, cate2, city, review_count, stars, his_item_seq, his_cate1_seq, his_cate2_seq, his_city_seq, his_review_count_seq, his_stars_seq, label, user, review_count_u, year, stars_u]

def generate_test_batch(batch_size, global_index, sku_info_map, user_info_map):
    item = np.reshape(item_test_list[global_index - batch_size:global_index], [-1, 1])
    cate1 = np.reshape(
        [sku_info_map[item_test_list[i]]['cate1'] for i in range(global_index - batch_size, global_index)], [-1, 1])
    cate2 = np.reshape(
        [sku_info_map[item_test_list[i]]['cate2'] for i in range(global_index - batch_size, global_index)], [-1, 1])
    city = np.reshape(
        [sku_info_map[item_test_list[i]]['city'] for i in range(global_index - batch_size, global_index)], [-1, 1])
    review_count = np.reshape(
        [sku_info_map[item_test_list[i]]['review_count'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])
    stars = np.reshape(
        [sku_info_map[item_test_list[i]]['stars'] for i in range(global_index - batch_size, global_index)], [-1, 1])

    his_item_seq = his_item_seq_test_list[global_index - batch_size:global_index]
    his_cate1_seq = []
    his_cate2_seq = []
    his_city_seq = []
    his_review_count_seq = []
    his_stars_seq = []
    for l in his_item_seq:
        his_cate1_seq.append([sku_info_map[i]['cate1'] for i in l])
        his_cate2_seq.append([sku_info_map[i]['cate2'] for i in l])
        his_city_seq.append([sku_info_map[i]['city'] for i in l])
        his_review_count_seq.append([sku_info_map[i]['review_count'] for i in l])
        his_stars_seq.append([sku_info_map[i]['stars'] for i in l])

    label = np.reshape(label_test_list[global_index - batch_size:global_index], [-1, 1])
    user = np.reshape(user_test_list[global_index - batch_size:global_index], [-1, 1])
    review_count_u = np.reshape(
        [user_info_map[user_test_list[i]]['review_count'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])
    year = np.reshape(
        [user_info_map[user_test_list[i]]['yelping_since'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])
    stars_u = np.reshape(
        [user_info_map[user_test_list[i]]['average_stars'] for i in range(global_index - batch_size, global_index)],
        [-1, 1])

    return [item, cate1, cate2, city, review_count, stars, his_item_seq, his_cate1_seq, his_cate2_seq, his_city_seq,
            his_review_count_seq, his_stars_seq, label, user, review_count_u, year, stars_u]


def eval_full_data(trace_list, sku_info_map, emb_sku, flag='rank'):

    aucs_ctr = []
    ndcgs_ctr_10 = []
    ndcgs_ctr_20 = []
    cnt_trace_20 = 0

    print('evaluate all trace:')
    for trace_id, sku_list in tqdm(trace_list.items()):
        cnt_trace_20 += 1
        if flag == 'rank':
            sku_sorted_list = sorted(sku_list, key=lambda x: x['prediction'], reverse=True)
        else:
            sku_sorted_list = sku_list

        label_ctr = [x['ctr_label'] for x in sku_sorted_list]
        new_score = [-i for i in range(len(sku_sorted_list))]

        aucs_ctr.append(roc_auc_score(label_ctr, new_score))

        label_ctr_10 = label_ctr[:10]
        new_score_10 = new_score[:10]

        if np.sum(label_ctr_10) == 0.0:
            ndcgs_ctr_10.append(0.)
        else:
            ndcgs_ctr_10.append(ndcg_score(label_ctr_10, new_score_10, len(label_ctr_10)))

        label_ctr_20 = label_ctr[:20]
        new_score_20 = new_score[:20]

        if np.sum(label_ctr_20) == 0.0:
            ndcgs_ctr_20.append(0.)
        else:
            ndcgs_ctr_20.append(ndcg_score(label_ctr_20, new_score_20, len(label_ctr_20)))


    return np.mean(aucs_ctr), np.mean(ndcgs_ctr_10), np.mean(ndcgs_ctr_20)

def train(sku_info_map, user_info_map, exp_name, model_name):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        input_format['data_type'] = data_type
        if model_name == 'SASREC':
            model = SASREC(input_format)
        elif model_name == 'FDSA':
            model = FDSA(input_format)
        elif model_name == 'NOVA':
            model = NOVA(input_format)
        elif model_name == 'DIF':
            model = DIF(input_format)
        elif model_name == 'MAIF':
            model = MAIF(input_format)
        else:
            model = SASREC(input_format)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        global_auc = 0.
        global_patience = 0

        file_log = open(exp_name+'_log.txt', 'w')

        best_model_path = './'+exp_name+'/best_model/'

        print('start training:')
        start_time = time.time()
        # train
        for i in range(input_format['num_epoch']):
            new_index = np.arange(len(user_list)).tolist()
            np.random.shuffle(new_index)

            loss_sum = 0
            print('train epoch:', i)
            file_log.write('train epoch: ' + str(i) + '\n')
            pbar = tqdm(total=len(user_list)//input_format['batch_size'])
            global_index = input_format['batch_size']
            while global_index <= len(user_list):
                batch_data = generate_train_batch(new_index[global_index-input_format['batch_size']:global_index], sku_info_map, user_info_map)
                loss, op_ = model.train(batch_data, sess, data_type)
                loss_sum += loss
                global_index += input_format['batch_size']
                pbar.update(1)
            pbar.close()
            print('training loss:{}, global_step:{}'.format(loss_sum / (len(user_list)//input_format['batch_size']), i))
            file_log.write('training loss: '+str(loss_sum / (len(user_list)//input_format['batch_size']))+', global_step: '+ str(i)+'\n')

            time_cur = time.time()
            time_cost = (time_cur-start_time)/60.0

            print('time for training {} epoch is {} min'.format(i, time_cost))
            file_log.write('time for training: '+str(i)+' epoch is '+str(time_cost)+' min'+'\n')

            # valid
            if i % 1 == 0:
                print('begin valid after epoch:', i)
                file_log.write('begin valid after epoch: ' + str(i) + '\n')
                trace = {}
                sku_emb = model.get_sku_emb(sess)
                global_index = input_format['batch_size']
                while global_index <= len(user_valid_list):
                    batch_data = generate_valid_batch(input_format['batch_size'], global_index, sku_info_map, user_info_map)
                    predict = model.test(batch_data, sess, data_type)
                    predict_l = np.reshape(predict, [-1]).tolist()
                    skuid_l = np.reshape(batch_data[0], [-1]).tolist()
                    uid_l = np.reshape(batch_data[13], [-1]).tolist()
                    label_l = np.reshape(batch_data[12], [-1]).tolist()
                    for k in range(len(uid_l)):
                        uid = uid_l[k]
                        skuid = skuid_l[k]
                        predict_k = predict_l[k]
                        label_k = label_l[k]
                        if uid not in trace.keys():
                            trace[uid] = []
                        trace[uid].append({'prediction':predict_k, 'ctr_label':label_k, 'skuid':skuid})
                    global_index += input_format['batch_size']

                auc, ndcg10, ndcg20 = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
                if auc > global_auc:
                    global_auc = auc
                    global_patience = 0
                    model.save(sess, best_model_path)
                else:
                    global_patience += 1
                    if global_patience >= input_format['patience']:
                        break
                print('auc:{}, ndcg10:{}, ndcg20:{} global_step:{}'.format(auc,ndcg10,ndcg20, i))
                file_log.write('auc: '+str(auc)+', ndcg10: '+str(ndcg10)+', ndcg20: '+str(ndcg20)+', global_step: '+str(i)+'\n\n')

        model.restore(sess, best_model_path)
        file_log.write('begin test after epoch: ' + str(i) + '\n')
        print('begin test after epoch: ', i)
        trace = {}
        sku_emb = model.get_sku_emb(sess)
        global_index = input_format['batch_size']
        while global_index <= len(user_test_list):
            batch_data = generate_test_batch(input_format['batch_size'], global_index, sku_info_map, user_info_map)
            predict = model.test(batch_data, sess, data_type)
            predict_l = np.reshape(predict, [-1]).tolist()
            skuid_l = np.reshape(batch_data[0], [-1]).tolist()
            uid_l = np.reshape(batch_data[13], [-1]).tolist()
            label_l = np.reshape(batch_data[12], [-1]).tolist()
            for k in range(len(uid_l)):
                uid = uid_l[k]
                skuid = skuid_l[k]
                predict_k = predict_l[k]
                label_k = label_l[k]
                if uid not in trace.keys():
                    trace[uid] = []
                trace[uid].append({'prediction': predict_k, 'ctr_label': label_k, 'skuid': skuid})
            global_index += input_format['batch_size']

        file_log.write(model_name +' model results:\n')
        print(model_name + ' model results:')

        auc, ndcg10, ndcg20 = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
        file_log.write('auc: ' + str(auc) + ', ndcg10: ' + str(ndcg10) + ', ndcg20: ' + str(ndcg20) + '\n')
        print('auc:{}, ndcg10:{}, ndcg20:{}'.format(auc, ndcg10, ndcg20))


def test(best_model_path, sku_info_map, user_info_map, model_name):
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        input_format['data_type'] = data_type
        if model_name == 'SASREC':
            model = SASREC(input_format)
        elif model_name == 'FDSA':
            model = FDSA(input_format)
        elif model_name == 'NOVA':
            model = NOVA(input_format)
        elif model_name == 'DIF':
            model = DIF(input_format)
        elif model_name == 'MAIF':
            model = MAIF(input_format)
        else:
            model = SASREC(input_format)
        model.restore(sess, best_model_path)

        file_log = open(model_name+'_test_result.txt', 'w')

        print('begin test:')
        trace = {}
        sku_emb = model.get_sku_emb(sess)
        global_index = input_format['batch_size']
        while global_index <= len(user_test_list):
            batch_data = generate_test_batch(input_format['batch_size'], global_index, sku_info_map, user_info_map)
            predict = model.test(batch_data, sess, data_type)
            predict_l = np.reshape(predict, [-1]).tolist()
            skuid_l = np.reshape(batch_data[0], [-1]).tolist()
            uid_l = np.reshape(batch_data[13], [-1]).tolist()
            label_l = np.reshape(batch_data[12], [-1]).tolist()
            for k in range(len(uid_l)):
                uid = uid_l[k]
                skuid = skuid_l[k]
                predict_k = predict_l[k]
                label_k = label_l[k]
                if uid not in trace.keys():
                    trace[uid] = []
                trace[uid].append({'prediction': predict_k, 'ctr_label': label_k, 'skuid': skuid})
            global_index += input_format['batch_size']

        file_log.write(model_name +' model results:\n')
        print(model_name + ' model results:')

        auc, ndcg10, ndcg20 = eval_full_data(trace, sku_info_map, sku_emb, 'rank')
        file_log.write('auc: ' + str(auc) + ', ndcg10: ' + str(ndcg10) + ', ndcg20: ' + str(ndcg20) + '\n')
        print('auc:{}, ndcg10:{}, ndcg20:{}'.format(auc, ndcg10, ndcg20))


if len(sys.argv) > 1:
    data_type = sys.argv[1]
    file_path_train = sys.argv[2]
    file_path_valid = sys.argv[3]
    file_path_test = sys.argv[4]
    sku_info_file = sys.argv[5]
    user_info_file = sys.argv[6]
    mode = sys.argv[7]
    model_name = sys.argv[8]
    exp_name = sys.argv[9]
else:
    data_type = 'yelp'
    file_path_train = 'yelp_traindata_2019_negall.txt'
    file_path_valid = 'yelp_validdata_2019_08_200.txt'
    file_path_test = 'yelp_testdata_2019_08_200.txt'
    sku_info_file = 'yelp_2019_sku_info.npz'
    user_info_file = 'yelp_2019_user_info.npz'
    mode = 'train'
    model_name = 'MAIF'
    exp_name = 'maif_all_yelp_batch256_dim16'

# read file and construct training data and testing data

# data = np.load('20220810_sku_map_spu.npz', allow_pickle=True)
# sku_info_map = data['sku_info_map'].tolist()

# data_matrix = np.load('similar_matrix_0808_0809_cate3.npz', allow_pickle=True)
similar_matrix = []
new_index = []

SEED = 19
tf.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if mode == 'train':
    sku_info_map = load_sku_info_map(sku_info_file)
    user_info_map = load_user_info_map(user_info_file)
    read_train_file(file_path_train)
    read_valid_file(file_path_valid)
    read_test_file(file_path_test)

    train(sku_info_map, user_info_map, exp_name, model_name)

elif mode == 'test':
    best_model_path = 'sasrec_yelp_batch256_dim16/best_model/'
    read_test_file(file_path_test)

    sku_info_map = load_sku_info_map(sku_info_file)
    user_info_map = load_user_info_map(user_info_file)
    test(best_model_path, sku_info_map, user_info_map, model_name)
