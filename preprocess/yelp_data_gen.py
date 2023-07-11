import os
import random
from tqdm import tqdm
import time
import numpy as np
import math
# x = {'a':1,'b':2,'c':3}
# print(list(x.items()))
# for i in list(x.items()):


# print(set(x))

# for i in set(x):
#     print(i)
# for i in range(5):
#     sim_user = random.choice(x)
#     while sim_user in sim_users:
#         sim_user = random.choice(x)
#     sim_users.append(sim_user)
# print(sim_users)


# with open('UserBehavior.csv','r') as f:
#     for line in tqdm(f,total=100200000):
#         line_split = line.strip('\n').split(',')
#         # cnt += 1
#         user_id = line_split[0]
#         item_id = line_split[1]
#         cate_id = line_split[2]
#         # if user_id not in user_interaction.keys():
#         #     user_interaction[user_id] = []
#         # user_interaction[user_id].append(item_id)
#         # cate_map[cate_id] = 1
#         # item_cate[item_id] = cate_id
#         if item_id not in item_cnt.keys():
#             item_cnt[item_id] = 0
#         item_cnt[item_id] += 1
# sku_info = np.load('yelp_2019_sku_info.npz', allow_pickle=True)
# user_info = np.load('yelp_2019_user_info_all.npz', allow_pickle=True)

user_interaction = {}
user_his_cate = {}
cate_map = {}
cate_part = {}
item_cate = {}
item_cnt = {}
cnt = 0
dt_map = {}
with open('yelp_2019.txt','r') as f:
    first = True
    for line in tqdm(f,total=630000):
        if first == True:
            first = False
            continue
        line_split = line.strip('\n').split('\t')
        user_id = line_split[0]
        item_id = line_split[1]
        timestamp = line_split[2]
        dt = line_split[3]
        if item_id not in item_cnt.keys():
            item_cnt[item_id] = 0
        item_cnt[item_id] += 1

        if user_id not in user_interaction.keys():
            user_interaction[user_id] = []
        user_interaction[user_id].append({'item_id':item_id,'timestamp':timestamp,'dt':dt})


for user in tqdm(user_interaction.keys()):
    interaction_list = user_interaction[user]
    user_interaction[user] = sorted(interaction_list, key=lambda x:x['timestamp'])

# print(dt_map)
# print(cnt)

# cnt_user_10 = 0
# for user in user_interaction.keys():
#     if len(user_interaction[user]) >= 10:
#         cnt_user_10 += 1

# print(cnt_user_10)

print('all_user_cnt:',len(user_interaction))
# print('cate_cnt:',len(cate_map))
# data = np.load('taobao_user_part_1125_1203.npz', allow_pickle=True)
# user_part = data['users'].tolist()
print('begin filter item map:')
item_map = {'0':'0'}
for item in item_cnt.keys():
    if item_cnt[item] >= 5:
        item_map[item] = str(len(item_map))

file_item = open('yelp_item_map.txt', 'w')
file_item.write('item_id\tindex_id\n')
for item in item_map.keys():
    file_item.write(item+'\t'+item_map[item]+'\n')

sort_item_list = sorted(item_cnt.items(), key=lambda x:x[1], reverse=True)
top_50_item = [i[0] for i in sort_item_list[:50]]
top_100_item = [i[0] for i in sort_item_list[:200]]

print('filter_item_size:', len(item_map))
user_part = list(user_interaction.keys())
total_interaction = 0
# for user in tqdm(user_part):
#     # total_interaction += len(user_interaction[user])
#     for item in user_interaction[user]:
#         if item['item_id'] not in item_cnt.keys():
#             item_cnt[item['item_id']] = 0
#         item_cnt[item['item_id']] += 1
# if item_cate[item] not in cate_part.keys():
#     cate_part[item_cate[item]] = {}
# cate_part[item_cate[item]][item] = 1

user_cnt = 0
filter_user = []
for user in tqdm(user_part):
    # total_interaction += len(user_interaction[user])
    new_list = []
    # cate_list = []
    for item in user_interaction[user]:
        if item_cnt[item['item_id']] >= 5:
            new_list.append(item)
            # if item['item_id'] not in item_part.keys():
            #     item_part[item['item_id']] = 0
            # item_part[item['item_id']] += 1
            # cate_list.append(item_cate[item['item_id']])
            # if item_cate[item['item_id']] not in cate_part.keys():
            #     cate_part[item_cate[item['item_id']]] = {}
            # cate_part[item_cate[item['item_id']]][item['item_id']] = 1
    user_interaction[user] = new_list
    # user_his_cate[user] = list(set(cate_list))
    if len(user_interaction[user]) >= 5:
        user_cnt += 1
        total_interaction += len(user_interaction[user])
        filter_user.append(user)

print('filter_user_cnt:',user_cnt)
print('filter_total_interaction:',total_interaction)

user_map = {'0':'0'}
for user in filter_user:
    user_map[user] = str(len(user_map))
file_user = open('yelp_user_map.txt', 'w')
file_user.write('user_id\tindex_id\n')
for user in user_map.keys():
    file_user.write(user+'\t'+user_map[user]+'\n')

user_info = np.load('yelp_2019_user_info_all.npz', allow_pickle=True)
user_info = user_info['user_info'].tolist()

# data_item_info = np.load('yelp_2019_sku_info.npz', allow_pickle=True)
# item_info = data_item_info['sku_info'].tolist()
# item_index = data_item_info['item_index'].tolist()

split_train_valid = math.ceil(len(filter_user)*0.8)
split_valid_test = math.ceil(len(filter_user)*0.9)
train_user = filter_user[:split_train_valid]
valid_user = filter_user[split_train_valid:split_valid_test]
# valid_user = filter_user[split_train_valid:]
test_user = filter_user[split_valid_test:]


item_sort = sorted(item_cnt.items(), key=lambda x:x[1], reverse=True)
# data1 = np.load('top_1000_yelp.npz',allow_pickle=True)
# hot_item_list = set([str(item_map[x[0]]) for x in item_sort[:100]])

file_train = open('yelp_traindata_2019_negall_3.txt','w')
file_valid = open('yelp_validdata_2019_08_200.txt','w')
file_test = open('yelp_testdata_2019_08_200.txt','w')
file_train.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'dt'+'\n')
file_valid.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'dt'+'\n')
file_test.write('queryid'+'\t'+'uuid'+'\t'+'item_id'+'\t'+'his_item_seq'+'\t'+'label'+'\t'+'dt'+'\n')
query_id = 0

neg_all = []
for i in range(len(item_map)):
    neg_all.append(str(i))

train_sample_cnt = 0
valid_sample_cnt = 0
test_sample_cnt = 0
for user in tqdm(train_user):
    interaction_list = user_interaction[user]
    his_item_list = [str(item_map[i['item_id']]) for i in interaction_list]

    his_item_map = {}
    for ii in his_item_list:
        his_item_map[ii] = 1

    if user not in user_info.keys():
        continue
    for index in range(len(interaction_list)):
        query_id += 1
        item = interaction_list[index]
        item_id = item_map[item['item_id']]
        dt = item['dt']
        # if dt <= '2019-12-01':
        train_sample_cnt += 1
        file_train.write(str(query_id)+'\t'+user_map[user]+'\t'+str(item_id)+'\t'+'-'.join(his_item_list[:index])+'\t1\t'+dt+'\n')

        for j in range(2):
            train_sample_cnt += 1
            neg = random.choice(neg_all)
            while neg in his_item_map.keys():
                neg = random.choice(neg_all)
            file_train.write(str(query_id) + '\t' + user_map[user] + '\t' + str(neg) + '\t' + '-'.join(his_item_list[:index]) + '\t0\t' + dt + '\n')

for user in tqdm(valid_user):
    interaction_list = user_interaction[user]
    his_item_list = [str(item_map[i['item_id']]) for i in interaction_list]

    his_item_map = {}
    for ii in his_item_list:
        his_item_map[ii] = 1

    if user not in user_info.keys():
        continue

    query_id += 1
    split_index = math.ceil(len(his_item_list)*0.8)

    for index in range(split_index, len(interaction_list)):
        item = interaction_list[index]
        item_id = item_map[item['item_id']]
        dt = item['dt']
        # if dt <= '2019-12-01':
        valid_sample_cnt += 1
        file_valid.write(str(query_id)+'\t'+user_map[user]+'\t'+str(item_id)+'\t'+'-'.join(his_item_list[:split_index])+'\t1\t'+dt+'\n')

    for j in range(50):
        valid_sample_cnt += 1
        neg = random.choice(top_100_item)
        while neg in his_item_map.keys():
            neg = random.choice(top_100_item)
        # neg = item_map[top_50_item[j]]
        file_valid.write(str(query_id) + '\t' + user_map[user] + '\t' + str(item_map[neg]) + '\t' + '-'.join(his_item_list[:split_index]) + '\t0\t' + dt + '\n')

for user in tqdm(test_user):
    interaction_list = user_interaction[user]
    his_item_list = [str(item_map[i['item_id']]) for i in interaction_list]

    his_item_map = {}
    for ii in his_item_list:
        his_item_map[ii] = 1

    if user not in user_info.keys():
        continue

    split_index = math.ceil(len(his_item_list)*0.8)
    query_id += 1

    for index in range(split_index, len(interaction_list)):
        item = interaction_list[index]
        item_id = item_map[item['item_id']]
        dt = item['dt']
        # if dt <= '2019-12-01':
        test_sample_cnt += 1
        file_test.write(str(query_id)+'\t'+user_map[user]+'\t'+str(item_id)+'\t'+'-'.join(his_item_list[:split_index])+'\t1\t'+dt+'\n')

    for j in range(50):
        test_sample_cnt += 1
        neg = random.choice(top_100_item)
        while neg in his_item_map.keys():
            neg = random.choice(top_100_item)
        # neg = item_map[top_50_item[j]]
        file_test.write(str(query_id) + '\t' + user_map[user] + '\t' + str(item_map[neg]) + '\t' + '-'.join(his_item_list[:split_index]) + '\t0\t' + dt + '\n')



print('train_sample_cnt:', train_sample_cnt)
print('valid_sample_cnt:', valid_sample_cnt)
print('test_sample_cnt:', test_sample_cnt)