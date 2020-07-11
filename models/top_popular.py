#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from models.model import Model

class TopPopular(Model):

    def __init__(self, users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field):
        super().__init__(users, items, observed_relevance, unobserved_relevance, category_per_item, item_field, user_field, rating_field)

    def predict(self):
        self.predicted_relevance = np.zeros((self.no_users, self.no_items))
        for user_id in self.users:
            print('\r> Making predictions for user', user_id, '/', self.no_users, end='')
            self.predicted_relevance[user_id] = self.item_popularity / np.sum(self.item_popularity)
