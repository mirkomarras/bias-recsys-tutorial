#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse

def fixed_timestamp(interactions, min_train=4, min_test=1, min_time=None, max_time=None, step_time=10000, user_field='user_id', item_field='item_id', time_field='timestamp', rating_field='rating'):
    timestamps = [t for t in interactions[time_field].values if str(t) != 'nan']
    timestamps.sort()

    best_timestamp = -np.inf
    best_learners = -np.inf
    best_interactions = -np.inf
    best_list_learners = []

    min_time = min_time if min_time else 0
    max_time = max_time if max_time else len(timestamps)

    for i in range(min_time, max_time, step_time):

        current_timestamp = timestamps[i]
        train = interactions[interactions[time_field] <= current_timestamp]
        test = interactions[interactions[time_field] > current_timestamp]

        train_per_user = train.groupby([user_field]).count()[[item_field, rating_field]]
        train_per_user.columns = [item_field + '_train', rating_field + '_train']
        test_per_user = test.groupby([user_field]).count()[[item_field, rating_field]]
        test_per_user.columns = [item_field + '_test', rating_field + '_test']

        result = pd.concat([train_per_user, test_per_user], axis=1)[[item_field + '_train', item_field + '_test']]
        result.columns = ['no_train', 'no_test']

        result_learners = result[(result['no_train'] >= min_train) & (result['no_test'] >= min_test)].index
        no_learners = len(result_learners)
        no_interactions = len(interactions[interactions[user_field].isin(result_learners)])

        if no_learners > best_learners or (no_learners == best_learners and no_interactions > best_interactions):
            best_learners = no_learners
            best_timestamp = current_timestamp
            best_interactions = no_interactions
            best_list_learners = result_learners

        print('\r> Index:', i, 'Current:', current_timestamp, no_learners, no_interactions, 'Best:', best_timestamp, best_learners, best_interactions, end='')

    print()

    interactions = interactions[interactions[user_field].isin(best_list_learners)].copy()
    interactions[user_field + '_original'] = interactions[user_field]
    interactions[item_field + '_original'] = interactions[item_field]
    interactions[user_field] = interactions[user_field].astype('category').cat.codes
    interactions[item_field] = interactions[item_field].astype('category').cat.codes

    train = interactions[interactions[time_field] <= best_timestamp]
    test = interactions[interactions[time_field] > best_timestamp]

    train['set'] = 'train'
    test['set'] = 'test'

    traintest = pd.concat([train, test])

    print('> Bast timestamp found at', best_timestamp)
    print('> Mean number of train ratings per learner:', np.mean(traintest[traintest['set']=='train'].groupby([user_field]).count()[item_field].values))
    print('> Mean number of test ratings per learner:', np.mean(traintest[traintest['set']=='test'].groupby([user_field]).count()[item_field].values))

    return traintest

def user_timestamp(interactions, split=0.80, min_samples=10, user_field='user_id', item_field='item_id', time_field='timestamp'):
    train_set = []
    test_set = []
    groups = interactions.groupby([user_field])

    for i, (index, group) in enumerate(groups):
        if i % 1000 == 0:
            print('\r> Parsing user', i, 'of', len(groups), end='')
        if len(group.index) < min_samples:
            continue
        sorted_group = group.sort_values(time_field)
        n_rating_test = int(len(sorted_group.index) * (1.0 - split))
        train_set.append(sorted_group.head(len(sorted_group.index) - n_rating_test))
        test_set.append(sorted_group.tail(n_rating_test))
    print()

    train, test = pd.concat(train_set), pd.concat(test_set)

    train['set'] = 'train'
    test['set'] = 'test'

    traintest = pd.concat([train, test])

    traintest[user_field + '_original'] = traintest[user_field]
    traintest[item_field + '_original'] = traintest[item_field]
    traintest[user_field] = traintest[user_field].astype('category').cat.codes
    traintest[item_field] = traintest[item_field].astype('category').cat.codes

    print('> Mean number of train ratings per learner:', np.mean(traintest[traintest['set']=='train'].groupby([user_field]).count()[item_field].values))
    print('> Mean number of test ratings per learner:', np.mean(traintest[traintest['set']=='test'].groupby([user_field]).count()[item_field].values))

    return traintest

def user_random(interactions, split=0.80, min_samples=10, user_field='user_id', item_field='item_id'):
    train_set = []
    test_set = []
    groups = interactions.groupby([user_field])

    for i, (index, group) in enumerate(groups):
        if i % 1000 == 0:
            print('\r> Parsing user', i, 'of', len(groups), end='')
        if len(group.index) < min_samples:
            continue
        shuffled_group = group.sample(frac=1)
        n_rating_test = int(len(shuffled_group.index) * (1.0 - split))
        train_set.append(shuffled_group.head(len(shuffled_group.index) - n_rating_test))
        test_set.append(shuffled_group.tail(n_rating_test))
    print()

    train, test = pd.concat(train_set), pd.concat(test_set)

    train['set'] = 'train'
    test['set'] = 'test'

    traintest = pd.concat([train, test])

    traintest[user_field + '_original'] = traintest[user_field]
    traintest[item_field + '_original'] = traintest[item_field]
    traintest[user_field] = traintest[user_field].astype('category').cat.codes
    traintest[item_field] = traintest[item_field].astype('category').cat.codes

    print('> Mean number of train ratings per learner:', np.mean(traintest[traintest['set']=='train'].groupby([user_field]).count()[item_field].values))
    print('> Mean number of test ratings per learner:', np.mean(traintest[traintest['set']=='test'].groupby([user_field]).count()[item_field].values))

    return traintest

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', default='coco', type=str, action='store')
    parser.add_argument('--method', dest='method', default='per_user_timestamp', type=str, action='store')
    parser.add_argument('--min_train', dest='min_train', default=8, type=int, action='store')
    parser.add_argument('--min_test', dest='min_test', default=2, type=int, action='store')
    parser.add_argument('--min_time', dest='min_time', default=None, type=int, action='store')
    parser.add_argument('--max_time', dest='max_time', default=None, type=int, action='store')
    parser.add_argument('--step_time', dest='step_time', default=1000, type=int, action='store')
    parser.add_argument('--user_field', dest='user_field', default='user_id', type=str, action='store')
    parser.add_argument('--item_field', dest='item_field', default='item_id', type=str, action='store')
    parser.add_argument('--rating_field', dest='rating_field', default='rating', type=str, action='store')
    parser.add_argument('--time_field', dest='time_field', default='timestamp', type=str, action='store')
    parser.add_argument('--percentage', dest='percentage', default=0.80, type=float, action='store')

    args = parser.parse_args()

    print('Parameters summary')
    print('> Dataset:', args.dataset)
    print('> Method:', args.method)
    print('> Mininum train user samples:', args.min_train)
    print('> Mininum test user samples:', args.min_test)
    print('> Minimum timestamp:', args.min_time)
    print('> Maximum timestamp:', args.max_time)
    print('> Timestamp step:', args.step_time)
    print('> User field:', args.user_field)
    print('> Item field:', args.item_field)
    print('> Rating field:', args.rating_field)
    print('> Time field:', args.time_field)
    print('> Train percentage:', args.percentage)

    print('Step 1: Load interactions')
    data = pd.read_csv('./data/datasets/' + args.dataset + '.csv', encoding='utf8')
    print(data.head())

    print('Step 2: Splitting train and test')
    if args.method == 'uftime':
        traintest = fixed_timestamp(interactions=data, min_train=args.min_train, min_test=args.min_test, min_time=args.min_time, max_time=args.max_time, step_time=args.step_time, user_field=args.user_field, item_field=args.item_field, time_field=args.time_field, rating_field=args.rating_field)
    elif args.method == 'utime':
        traintest = user_timestamp(interactions=data, split=args.percentage, min_samples=args.min_train+args.min_test, user_field=args.user_field, item_field=args.item_field, time_field=args.time_field)
    elif args.method == 'urandom':
        traintest = user_random(interactions=data, split=args.percentage, min_samples=args.min_train+args.min_test, user_field=args.user_field, item_field=args.item_field)
    else:
        raise NotImplementedError('The split method ' + args.method + 'is not supported.')
