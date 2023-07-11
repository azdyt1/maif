import numpy as np
from tqdm import tqdm
import json
import os

user_info = {}
sku_info = {0:{'year':0, 'cate1':0, 'cate2':0, 'stars':0}}

year_map = {'0':'0'}
item_map = {}
with open('movie_item_map.txt', 'r') as f:
    first = True
    for line in tqdm(f, total=30000):
        if first == True:
            first = False
            continue
        line_split = line.strip().split('\t')
        item_id = line_split[0]
        index_id = line_split[1]
        item_map[item_id] = index_id

occupation_map = {}
with open('users.dat', 'r') as f:
    for line in tqdm(f, total=2000000):
        line_split = line.strip('\n').split('::')
        user_id = line_split[0]
        gender = line_split[1]
        if gender == 'M':
            gender = 0
        else:
            gender = 1
        age = int(line_split[2])
        occupation = int(line_split[3].split('-')[0])

        user_info[int(user_id)] = {'gender':gender, 'age':age, 'occupation':occupation}

np.savez('movie_user_info.npz', user_info=user_info)
print('len(user_info):', len(user_info))


movie_rates_map = {}
with open('ratings.dat', 'r') as f:
    for line in tqdm(f, total=1100000):
        line_split = line.strip('\n').split('::')
        user_id = line_split[0]
        item_id = line_split[1]
        rates = float(line_split[2])
        if item_id not in movie_rates_map.keys():
            movie_rates_map[item_id] = []
        movie_rates_map[item_id].append(rates)

for movie in movie_rates_map.keys():
    movie_rates_map[movie] = np.mean(movie_rates_map[movie])

cate_map = {'0':0}
year_map = {'0':0}
with open('movies.dat', 'r', encoding='ISO-8859-1') as f:
    for line in tqdm(f, total=160000):
        line_split = line.strip('\n').split('::')
        item_id = line_split[0]
        if item_id not in item_map.keys():
            continue
        name = line_split[1]
        year = name[-5:-1]
        if year not in year_map.keys():
            year_map[year] = len(year_map)
        categories = line_split[2]

        if categories is None or categories == '':
            categories = []
        else:
            categories = categories.split('|')
        if len(categories) == 0:
            cate1 = '0'
            cate2 = '0'
        elif len(categories) == 1:
            cate1 = categories[0]
            cate2 = '0'
        else:
            cate1 = categories[0]
            cate2 = categories[1]
        if cate1 not in cate_map.keys():
            cate_map[cate1] = len(cate_map)
        if cate2 not in cate_map.keys():
            cate_map[cate2] = len(cate_map)
        if item_id not in movie_rates_map.keys():
            stars = 0
        else:
            stars = int(round(float(movie_rates_map[item_id]),1) * 10)
        # file_sku_info.write(str(item_index[business_id])+'\t'+str(name_index[name])+'\t'+str(city_index[city])+'\t'+str(review_count)+'\t'+str(cate1_index[cate1])+'\t'+str(cate2_index[cate2])+'\t'+str(stars)+'\n')
        sku_info[int(item_map[item_id])] = {'year':year_map[year], 'cate1':cate_map[cate1], 'cate2':cate_map[cate2], 'stars':stars}

np.savez('movie_sku_info.npz', sku_info=sku_info)
print('len(sku_info):', len(sku_info))
print('len(year_map):', len(year_map))
print('len(cate_map):', len(cate_map))
