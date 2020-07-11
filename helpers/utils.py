#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pickle

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_bpr_loss(y_true, y_pred):
    return 1.0 - tf.keras.backend.sigmoid(y_pred)

def get_dot_difference_shape(shapeVectorList):
    userEmbeddingShapeVector, itemPositiveEmbeddingShapeVector, itemNegativeEmbeddingShapeVector = shapeVectorList
    return userEmbeddingShapeVector[0], 1

def get_dot_difference(parameterMatrixList):
    userEmbeddingMatrix, itemPositiveEmbeddingMatrix, itemNegativeEmbeddingMatrix = parameterMatrixList
    return tf.keras.backend.batch_dot(userEmbeddingMatrix, itemPositiveEmbeddingMatrix, axes=1) - tf.keras.backend.batch_dot(userEmbeddingMatrix, itemNegativeEmbeddingMatrix, axes=1)