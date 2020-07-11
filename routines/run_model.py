#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import argparse
import os

from helpers.utils import save_obj, load_obj

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', dest='dataset', default='coco', type=str, action='store')
    parser.add_argument('--method', dest='method', default='random', type=str, action='store')
    parser.add_argument('--mode', dest='mode', default='implicit', type=str, action='store')
    parser.add_argument('--user_field', dest='user_field', default='user_id', type=str, action='store')
    parser.add_argument('--item_field', dest='item_field', default='item_id', type=str, action='store')
    parser.add_argument('--rating_field', dest='rating_field', default='rating', type=str, action='store')
    parser.add_argument('--type_field', dest='type_field', default='type_id', type=str, action='store')
    parser.add_argument('--model', dest='model', default='random', type=str, action='store')
    parser.add_argument('--cutoffs', dest='cutoffs', default='5,10,20,50,100,200', type=str, action='store')

    args = parser.parse_args()

    print('Parameters summary')
    print('> Dataset:', args.dataset)
    print('> Method:', args.method)
    print('> Mode:', args.mode)
    print('> User field:', args.user_field)
    print('> Item field:', args.item_field)
    print('> Rating field:', args.rating_field)
    print('> Type field:', args.type_field)
    print('> Model:', args.model)
    print('> Cutoffs:', args.cutoffs)

    print('Step 1: Loading interactions')
    traintest = pd.read_csv('./data/splits/' + args.dataset + '_' + args.method + '.csv', encoding='utf8')
    train = traintest[traintest['set']=='train'].copy()
    test = traintest[traintest['set']=='test'].copy()
    print('> Loading', len(train.index), 'train interactions')
    print('> Loading', len(test.index), 'test interactions')

    print('Step 2: Preparing users and items lists')
    users = list(np.unique(traintest[args.user_field].values))
    items = list(np.unique(traintest[args.item_field].values))
    users.sort()
    items.sort()
    print('> Loading', len(users), 'users -', np.min(users), '-', np.max(users), '-', len(np.unique(users)), '-', users[:10])
    print('> Loading', len(items), 'items -', np.min(items), '-', np.max(items), '-', len(np.unique(items)), '-', items[:10])

    print('Step 2: Loading item categories and content')
    items_metadata = traintest.drop_duplicates(subset=['item_id'], keep='first')
    print('> Retrieved', len(items_metadata.index), 'mapping indexes, one per course')
    category_per_item = items_metadata[args.type_field].values
    print('> Loading items descriptions and categories -', len(set(category_per_item)), 'categories like', category_per_item[:3])

    print('Step 3: Creating model')

    if args.mode == 'implicit':
        train[args.rating_field] = train[args.rating_field].apply(lambda x: 1.0)
        test[args.rating_field] = test[args.rating_field].apply(lambda x: 1.0)
        traintest[args.rating_field] = traintest[args.rating_field].apply(lambda x: 1.0)

    if args.model == 'bpr':
        from models.bpr import BPR
        print('> Set', args.model, 'model')
        gen_mode = 'pair'
        model = BPR(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'cfnet':
        from models.cfnet import CFNet
        gen_mode = 'point'
        print('> Set', args.model, 'model')
        model = CFNet(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'coupledcf':
        gen_mode = 'point'
        from models.coupledcf import CoupledCF
        print('> Set', args.model, 'model')
        model = CoupledCF(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'dmf':
        gen_mode = 'point'
        from models.dmf import DMF
        print('> Set', args.model, 'model')
        model = DMF(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'gmf':
        gen_mode = 'point'
        from models.gmf import GMF
        print('> Set', args.model, 'model')
        model = GMF(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'itemknn':
        gen_mode = 'point'
        from models.item_knn import ItemKNN
        model = ItemKNN(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'itemknncb':
        gen_mode = 'point'
        from models.item_knn_cb import ItemKNNCB
        model = ItemKNNCB(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'neumf':
        gen_mode = 'point'
        from models.neumf import NeuMF
        print('> Set', args.model, 'model')
        model = NeuMF(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'p3alpha':
        gen_mode = 'point'
        from models.p3alpha import P3Alpha
        model = P3Alpha(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'rp3beta':
        gen_mode = 'point'
        from models.rp3beta import RP3Beta
        model = RP3Beta(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'random':
        gen_mode = 'point'
        from models.random import Random
        print('> Set', args.model, 'model')
        model = Random(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'toppop':
        gen_mode = 'point'
        from models.top_popular import TopPopular
        print('> Set', args.model, 'model')
        model = TopPopular(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    elif args.model == 'userknn':
        gen_mode = 'point'
        from models.user_knn import UserKNN
        model = UserKNN(users, items, train, test, category_per_item, args.item_field, args.user_field, args.rating_field)
    else:
        model = None
        raise NotImplementedError('The model ' + args.model + ' is not implemented.')

    print('Step 4: Training model')
    model_path = './data/models/' + args.dataset + '_' + args.method + '_' + args.model + '_model.h5'
    if os.path.exists(model_path):
        print('> Loaded pre-computed model')
        model.set_model(model_path)
    else:
        model.train(model_path)
        model.get_model(model_path)

    print('Step 5: Predicting scores')
    pred_path = './data/predictions/' + args.dataset + '_' + args.method + '_' + args.model + '_pred.pkl'
    if os.path.exists(pred_path):
        print('> Loaded pre-computed predictions')
        model.set_predictions(load_obj(pred_path))
    else:
        model.predict()
        save_obj(model.get_predictions(), pred_path)
        print()

    print('Step 6: Computing metrics')
    metr_path = './data/metrics/' + args.dataset + '_' + args.method + '_' + args.model + '_metr.pkl'
    if os.path.exists(metr_path):
        print('> Loaded pre-computed metrics')
        model.set_metrics(load_obj(metr_path))
    else:
        model.test(cutoffs=np.array([int(k) for k in args.cutoffs.split(',')]))
        save_obj(model.get_metrics(), metr_path)
        print()

    print('Step 7: Show performance')
    model.show_metrics()
