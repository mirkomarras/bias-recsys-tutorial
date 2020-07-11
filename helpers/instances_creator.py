#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

def generator(observed_relevance, categories, no_categories, category_per_item, categories_per_user, no_negatives=10, gen_mode='point', item_popularity=None):
    user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr, labels = [], [], [], [], [], [], []
    no_users, no_items = observed_relevance.shape[0], observed_relevance.shape[1]

    users, items = np.nonzero(observed_relevance)
    positive_set_list = [set() for _ in range(no_users)]
    for (user_id, item_id) in zip(users, items):
        positive_set_list[int(user_id)].add(int(item_id))

    negative_set_list = [set() for _ in range(no_users)]
    for user_id in range(no_users):
        negative_set_list[user_id] = list(set(range(no_items)) - set(positive_set_list[int(user_id)]))


def generate_popularity_paired(train_set, no_users, no_items, item_popularity, type='', n_negative = 10):

    positive_set_list = [set() for _ in range(no_users)]
    positive_count = 0
    for (user, item, rating) in train_set:
        positive_set_list[int(user)].add(int(item))
        positive_count += 1

    instance_matrix = []
    label_matrix = []

    start_time, end_time = 0, 0
    item_id_rank = np.array(item_popularity.index)
    mapping = {v:i for i,v in enumerate(item_id_rank)}
    for index, (user, item, rating) in enumerate(train_set):
        if index % 10000 == 0:
            print('\r> step', index, '/', len(train_set), '- eta', (end_time - start_time) * (len(train_set) - index) // 60, 'min', end='')

        start_time = time.time()
        user = int(user)
        item = int(item)
        max_indexes = item_id_rank[:mapping[item]]
        min_indexes = item_id_rank[mapping[item]+1:]
        more_popular_items = list(set(max_indexes) - set(positive_set_list[user]))
        less_popular_items = list(set(min_indexes) - set(positive_set_list[user]))
        for s in range(int(n_negative//2)):
            other = random.choice(less_popular_items if len(less_popular_items) > 0 else range(no_items))
            instance_matrix.append([int(user), int(item), int(other)])
            label_matrix.append(np.random.randint(2, size=1))
            other = random.choice(more_popular_items if len(more_popular_items) > 0 else range(no_items))
            instance_matrix.append([int(user), int(item), int(other)])
            label_matrix.append(np.random.randint(2, size=1))
        end_time = time.time()

    print()

    return np.array(instance_matrix), np.array(label_matrix)

def generator(observed_relevance, categories, no_categories, category_per_item, categories_per_user, no_negatives=10, gen_mode='point'):
    user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr, labels = [], [], [], [], [], [], []
    no_users, no_items = observed_relevance.shape[0], observed_relevance.shape[1]

    users, items = np.nonzero(observed_relevance)
    positive_set_list = [set() for _ in range(no_users)]
    for (user_id, item_id) in zip(users, items):
        positive_set_list[int(user_id)].add(int(item_id))

    negative_set_list = [set() for _ in range(no_users)]
    for user_id in range(no_users):
        negative_set_list[user_id] = list(set(range(no_items)) - set(positive_set_list[int(user_id)]))

    for index, (user_id, item_id) in enumerate(zip(users, items)):
        if (index % 10000) == 0:
            print('\r> Making instances for interaction', index, '/', len(users), 'of type', gen_mode, end='')

        if gen_mode == 'point':
            user_input.append(user_id)
            user_attr.append(categories_per_user[user_id])
            item_i_input.append(item_id)
            item_i_attr_instance = np.zeros(no_categories)
            item_i_attr_instance[categories.index(category_per_item[item_id])] = 1
            item_i_attr.append(item_i_attr_instance)
            labels.append(1)

            for _ in range(no_negatives):
                user_input.append(user_id)
                user_attr.append(categories_per_user[user_id])
                j = random.choice(negative_set_list[user_id])
                item_i_attr_instance = np.zeros(no_categories)
                item_i_attr_instance[categories.index(category_per_item[j])] = 1
                item_i_attr.append(item_i_attr_instance)
                item_i_input.append(j)
                labels.append(0)

        elif gen_mode == 'pair':
            for _ in range(no_negatives):
                user_input.append(user_id)
                user_attr.append(categories_per_user[user_id])
                item_i_input.append(item_id)
                item_i_attr_instance = np.zeros(no_categories)
                item_i_attr_instance[categories.index(category_per_item[item_id])] = 1
                item_i_attr.append(item_i_attr_instance)
                j = random.choice(negative_set_list[user_id])
                item_j_attr_instance = np.zeros(no_categories)
                item_j_attr_instance[categories.index(category_per_item[j])] = 1
                item_j_attr.append(item_j_attr_instance)
                item_j_input.append(j)
                labels.append(1)

        else:
            raise NotImplementedError('The generation type ' + gen_mode + ' is not implemented.')
    print()

    return (user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr), (labels)