#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

from helpers.instances_creator import generator
from helpers.utils import load_obj, save_obj
from models.model import Model
from helpers.utils import get_bpr_loss, get_dot_difference, get_dot_difference_shape

class BPR(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def __get_model(self, mf_dim=10):
        user_embedding = tf.keras.layers.Embedding(self.no_users + 1, mf_dim, name='UserEmb')
        item_embedding = tf.keras.layers.Embedding(self.no_items + 1, mf_dim, name='ItemEmb')

        user_input = tf.keras.layers.Input(shape=[1], name='UserInput')
        user_vec = tf.keras.layers.Flatten(name='FlatUserEmb')(user_embedding(user_input))

        i_item_input = tf.keras.layers.Input(shape=[1], name='PosItemInput')
        pos_item_vec = tf.keras.layers.Flatten(name='FlatPosItemEmb')(item_embedding(i_item_input))

        j_item_input = tf.keras.layers.Input(shape=[1], name='NegItemInput')
        neg_item_vec = tf.keras.layers.Flatten(name='FlatNegItemEmb')(item_embedding(j_item_input))

        dot_difference = tf.keras.layers.Lambda(get_dot_difference, output_shape=get_dot_difference_shape, name='Accuracy')([user_vec, pos_item_vec, neg_item_vec])

        return tf.keras.Model(inputs=[user_input, i_item_input, j_item_input], outputs=[dot_difference])

    def train(self, filepath, no_epochs=20, no_batches=1024, lr=0.001, no_factors=10, no_negatives=10, gen_mode='pair', val_split=0.1):

        print('Generating training instances', 'of type', gen_mode)
        x, y = generator(self.observed_relevance, self.categories, self.no_categories, self.category_per_item, self.categories_per_user, no_negatives=no_negatives, gen_mode=gen_mode)

        print('> Making training -', 'Epochs', no_epochs, 'Batch Size', no_batches, 'Learning Rate', lr, 'Factors', no_factors, 'Negatives', no_negatives, 'Mode', gen_mode)
        self.model = self.__get_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss=get_bpr_loss)

        user_input, user_attr, item_i_input, item_i_attr, item_j_input, item_j_attr = x
        labels = y
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1), tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='loss', verbose=1)]
        self.history = self.model.fit([np.array(user_input), np.array(item_i_input), np.array(item_j_input)], np.array(labels), batch_size=no_batches, epochs=no_epochs, verbose=1, shuffle=True, callbacks=callbacks).history


    def predict(self):
        self.predicted_relevance = np.zeros((self.no_users, self.no_items))
        item_pids = np.arange(self.no_items, dtype=np.int32)
        user_matrix = self.model.get_layer('UserEmb').get_weights()[0]
        item_matrix = self.model.get_layer('ItemEmb').get_weights()[0]
        for user_id in range(self.no_users):
            if (user_id % 1000) == 0:
               print('\r> Making predictions for user', user_id, '/', self.no_users, end='')
            user_vector = user_matrix[user_id]
            item_vectors = item_matrix[item_pids]
            self.predicted_relevance[user_id] = np.array(np.dot(user_vector, item_vectors.T))