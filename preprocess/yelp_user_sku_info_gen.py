import numpy as np
from tqdm import tqdm
import json
import os

user_info = {}
sku_info = {0:{'city':0, 'review_count':0, 'cate1':0, 'cate2':0, 'stars':0}}

year_map = {'0':'0'}
item_map = {}
with open('yelp_item_map.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=30000):
        if first == True:
            first = False
            continue
        line_split = line.strip().split('\t')
        item_id = line_split[0]
        index_id = line_split[1]
        item_map[item_id] = index_id

user_map = {}
with open('yelp_user_map.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=20000):
        if first == True:
            first = False
            continue
        line_split = line.strip().split('\t')
        user_id = line_split[0]
        index_id = line_split[1]
        user_map[user_id] = index_id


with open('yelp_academic_dataset_user.json', 'r') as f:
    for line in tqdm(f, total=2000000):
        data = json.loads(line)
        user_id = data['user_id']
        if user_id not in user_map.keys():
            continue
        review_count = int(data['review_count'])
        yelping_since = str(data['yelping_since'][:4])
        if yelping_since not in year_map.keys():
            year_map[yelping_since] = len(year_map)

        average_stars = int(round(float(data['average_stars']),1) * 10)

        user_info[int(user_map[user_id])] = {'review_count':review_count, 'yelping_since':year_map[yelping_since], 'average_stars':average_stars}

np.savez('yelp_2019_user_info.npz', user_info=user_info)
print('len(user_info):', len(user_info))

city_index = {'0':0}
cate_index = {'0':0}

with open('yelp_academic_dataset_business.json', 'r') as f:
    for line in tqdm(f, total=160000):
        data = json.loads(line)
        business_id = data['business_id']

        if business_id not in item_map.keys():
            continue

        city = data['city']
        if city not in city_index.keys():
            city_index[city] = len(city_index)
        review_count = int(data['review_count'])

        categories = data['categories']
        if categories is None:
            categories = []
        else:
            categories = categories.split(',')
        if len(categories) == 0:
            cate1 = '0'
            cate2 = '0'
        elif len(categories) == 1:
            cate1 = categories[0]
            cate2 = '0'
        else:
            cate1 = categories[0]
            cate2 = categories[1]
        if cate1 not in cate_index.keys():
            cate_index[cate1] = len(cate_index)
        if cate2 not in cate_index.keys():
            cate_index[cate2] = len(cate_index)
        stars = int(round(float(data['stars']),1) * 10)
        # file_sku_info.write(str(item_index[business_id])+'\t'+str(name_index[name])+'\t'+str(city_index[city])+'\t'+str(review_count)+'\t'+str(cate1_index[cate1])+'\t'+str(cate2_index[cate2])+'\t'+str(stars)+'\n')
        sku_info[int(item_map[business_id])] = {'city':city_index[city], 'review_count':review_count, 'cate1':cate_index[cate1], 'cate2':cate_index[cate2], 'stars':stars}

np.savez('yelp_2019_sku_info.npz', sku_info=sku_info)
print('len(sku_info):', len(sku_info))

print('len(year_map):', len(year_map))
print('len(city_map):', len(city_index))
print('len(cate_map):', len(cate_index))