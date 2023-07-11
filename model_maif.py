# coding: utf-8
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.ops import partitioned_variables
from tensorflow.contrib.layers import l1_l2_regularizer

import importlib
import numpy as np
import os


class MAIF(object):

    @property
    def hash_features(self):
        return set()

    def label_parse(self, label, mode):
        return tf.minimum(label, 1)

    def __init__(self, input_format):
        self.input_format = input_format
        self.optimizers = {}

        self.uid = tf.placeholder(tf.int32, shape=[None, None])
        self.review_count_id_u = tf.placeholder(tf.int32, shape=[None, None])
        self.year_id = tf.placeholder(tf.int32, shape=[None, None])
        self.stars_id_u = tf.placeholder(tf.int32, shape=[None, None])

        self.skuid = tf.placeholder(tf.int32, shape=[None, None])
        self.cate1id = tf.placeholder(tf.int32, shape=[None, None])
        self.cate2id = tf.placeholder(tf.int32, shape=[None, None])
        self.review_count_id = tf.placeholder(tf.int32, shape=[None, None])
        self.cityid = tf.placeholder(tf.int32, shape=[None, None])
        self.starsid = tf.placeholder(tf.int32, shape=[None, None])

        self.hisids = tf.placeholder(tf.int32, shape=[None, None])
        self.hiscate1ids = tf.placeholder(tf.int32, shape=[None, None])
        self.hiscate2ids = tf.placeholder(tf.int32, shape=[None, None])
        self.hisreviewids = tf.placeholder(tf.int32, shape=[None, None])
        self.hiscityids = tf.placeholder(tf.int32, shape=[None, None])
        self.hisstarsids = tf.placeholder(tf.int32, shape=[None, None])

        self.label = tf.placeholder(tf.float32, shape=[None, None])

        self.params = input_format
        self.drop_rate = tf.placeholder(tf.float32)

        self.emb_user = tf.get_variable('user_embedding', shape=[self.params['user_num'], self.params['dims']],
                                       trainable=True,
                                       initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                  mode='FAN_OUT',
                                                                                                  uniform=True))
        self.emb_review_count = tf.get_variable('review_count_embedding', shape=[self.params['review_count_num'], self.params['dims']],
                                        trainable=True,
                                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                   mode='FAN_OUT',
                                                                                                   uniform=True))
        self.emb_stars = tf.get_variable('stars_embedding', shape=[self.params['stars_num'], self.params['dims']],
                                        trainable=True,
                                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                   mode='FAN_OUT',
                                                                                                   uniform=True))

        self.emb_year = tf.get_variable('year_embedding', shape=[self.params['year_num'], self.params['dims']],
                                        trainable=True,
                                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                   mode='FAN_OUT',
                                                                                                   uniform=True))

        self.emb_city = tf.get_variable('city_embedding', shape=[self.params['city_num'], self.params['dims']],
                                        trainable=True,
                                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                   mode='FAN_OUT',
                                                                                                   uniform=True))

        self.emb_sku = tf.get_variable('sku_embedding', shape=[self.params['item_num'], self.params['dims']], trainable=True, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_OUT',
                                                                                                       uniform=True))

        self.emb_cate = tf.get_variable('cate_embedding', shape=[self.params['cate_num'], self.params['dims']], trainable=True, initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                       mode='FAN_OUT',
                                                                                                      uniform=True))

        #user_profile_embedding
        self.user_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_user, self.uid), shape=[-1, self.params['dims']])
        self.user_review_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_review_count, self.review_count_id_u), shape=[-1, self.params['dims']])
        self.user_stars_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_stars, self.stars_id_u), shape=[-1, self.params['dims']])
        self.user_year_emb = tf.reshape(tf.nn.embedding_lookup(self.emb_year, self.year_id),shape=[-1, self.params['dims']])
        self.user_emb_join = tf.concat([self.user_review_emb, self.user_stars_emb, self.user_year_emb], -1)

        # candidate embedding
        self.target_item_in = tf.reshape(tf.nn.embedding_lookup(self.emb_sku, self.skuid), shape=[-1, self.params['dims']])
        self.target_cate1_in = tf.reshape(tf.nn.embedding_lookup(self.emb_cate, self.cate1id), shape=[-1, self.params['dims']])
        self.target_cate2_in = tf.reshape(tf.nn.embedding_lookup(self.emb_cate, self.cate2id),
                                          shape=[-1, self.params['dims']])
        self.target_review_in = tf.reshape(tf.nn.embedding_lookup(self.emb_review_count, self.review_count_id),
                                          shape=[-1, self.params['dims']])
        self.target_city_in = tf.reshape(tf.nn.embedding_lookup(self.emb_city, self.cityid),
                                          shape=[-1, self.params['dims']])
        self.target_stars_in = tf.reshape(tf.nn.embedding_lookup(self.emb_stars, self.starsid),
                                          shape=[-1, self.params['dims']])
        self.target_join_in = tf.concat([self.target_item_in, self.target_cate1_in, self.target_cate2_in, self.target_review_in, self.target_city_in, self.target_stars_in], -1)

        # user interaction encoding
        self.seq_his_item_in = tf.nn.embedding_lookup(self.emb_sku, self.hisids)
        self.seq_his_item_attn = self.get_sequence_embedding_with_transformer('his_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_item_in, mode='attention', seq_embedding=self.seq_his_item_in, user_embedding=None)

        self.seq_his_cate1_in = tf.nn.embedding_lookup(self.emb_cate, self.hiscate1ids)
        self.seq_his_cate2_in = tf.nn.embedding_lookup(self.emb_cate, self.hiscate2ids)
        self.seq_his_review_in = tf.nn.embedding_lookup(self.emb_review_count, self.hisreviewids)
        self.seq_his_city_in = tf.nn.embedding_lookup(self.emb_city, self.hiscityids)
        self.seq_his_stars_in = tf.nn.embedding_lookup(self.emb_stars, self.hisstarsids)

        # self.seq_his_join_cate1_attn = self.get_sequence_embedding_with_transformer('his_short_seq2', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_cate1_in, mode='attention', seq_embedding=self.seq_his_cate1_in, user_embedding=None)
        # self.seq_his_join_cate2_attn = self.get_sequence_embedding_with_transformer('his_short_seq3', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_cate2_in, mode='attention', seq_embedding=self.seq_his_cate2_in, user_embedding=None)
        # self.seq_his_join_review_attn = self.get_sequence_embedding_with_transformer('his_short_seq4', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_review_in, mode='attention', seq_embedding=self.seq_his_review_in, user_embedding=None)
        # self.seq_his_join_city_attn = self.get_sequence_embedding_with_transformer('his_short_seq5', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_city_in, mode='attention', seq_embedding=self.seq_his_city_in, user_embedding=None)
        # self.seq_his_join_stars_attn = self.get_sequence_embedding_with_transformer('his_short_seq6', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_stars_in, mode='attention', seq_embedding=self.seq_his_stars_in, user_embedding=None)

        # self.seq_his_join_sum = tf.reduce_sum(self.seq_his_join_in, 1)
        self.seq_his_join_cate1_attn = self.get_sequence_embedding_with_transformer_feature('his_feature_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_cate1_in, mode='attention', seq_embedding=[self.seq_his_cate2_in, self.seq_his_review_in, self.seq_his_city_in, self.seq_his_stars_in], target_seq_embedding=self.seq_his_cate1_in, user_embedding=self.user_emb_join)
        self.seq_his_join_cate2_attn = self.get_sequence_embedding_with_transformer_feature('his_feature_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_cate2_in, mode='attention', seq_embedding=[self.seq_his_cate1_in, self.seq_his_review_in, self.seq_his_city_in, self.seq_his_stars_in], target_seq_embedding=self.seq_his_cate2_in, user_embedding=self.user_emb_join)
        self.seq_his_join_review_attn = self.get_sequence_embedding_with_transformer_feature('his_feature_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_review_in, mode='attention', seq_embedding=[self.seq_his_cate1_in, self.seq_his_cate2_in, self.seq_his_city_in, self.seq_his_stars_in], target_seq_embedding=self.seq_his_review_in, user_embedding=self.user_emb_join)
        self.seq_his_join_city_attn = self.get_sequence_embedding_with_transformer_feature('his_feature_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_city_in, mode='attention', seq_embedding=[self.seq_his_cate1_in, self.seq_his_cate2_in, self.seq_his_review_in, self.seq_his_stars_in], target_seq_embedding=self.seq_his_city_in, user_embedding=self.user_emb_join)
        self.seq_his_join_stars_attn = self.get_sequence_embedding_with_transformer_feature('his_feature_short_seq', self.hisids, T=self.params['his_short_seq_max_len'], target_embedding=self.target_stars_in, mode='attention', seq_embedding=[self.seq_his_cate1_in, self.seq_his_cate2_in, self.seq_his_review_in, self.seq_his_city_in], target_seq_embedding=self.seq_his_stars_in, user_embedding=self.user_emb_join)

        self.seq_side_attn_concat = tf.concat([tf.expand_dims(self.seq_his_join_cate1_attn, 1),tf.expand_dims(self.seq_his_join_cate2_attn, 1),tf.expand_dims(self.seq_his_join_review_attn, 1),tf.expand_dims(self.seq_his_join_city_attn, 1),tf.expand_dims(self.seq_his_join_stars_attn, 1)], 1)

        self.side_attn = tf.layers.dense(self.seq_side_attn_concat, 1, activation=tf.nn.relu)
        self.side_attn = tf.nn.softmax(tf.transpose(self.side_attn, [0,2,1]))
        self.seq_his_join_all_attn_sum = tf.reshape(tf.matmul(self.side_attn, self.seq_side_attn_concat), [-1, self.params['dims']])

        # predicting layer
        self.deep_out = tf.concat([self.target_join_in, self.seq_his_item_attn, self.seq_his_join_all_attn_sum], 1)

        # self.deep_out_bn = tf.layers.batch_normalization(self.deep_out)

        self.common_deep_out1 = tf.layers.dense(self.deep_out, units=64, activation=tf.nn.relu, name='dense1')

        self.common_deep_out2 = tf.layers.dense(self.common_deep_out1, units=32, activation=tf.nn.relu, name='dense2')

        self.final_layer_out = tf.layers.dense(self.common_deep_out2, units=1, name='dense3')

        # loss.
        self.label = tf.reshape(self.label, [-1, 1])

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
           labels=self.label, logits=self.final_layer_out))

        self.optimizer = self.params['optimizer']
        self.train_op = self.optimizer.minimize(self.loss)


    def layer_normalize(self, inputs, epsilon=1e-8):
        '''
        Applies layer normalization
        Args:
            inputs: A tensor with 2 or more dimensions
            epsilon: A floating number to prevent Zero Division
        Returns:
            A tensor with the same shape and data dtype
        '''
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

        return outputs

    def self_multi_head_attn(self, queries,
                             keys,
                             values,
                             num_units=None,
                             num_heads=1,
                             is_drop=False,
                             dropout_keep_prob=1,
                             is_training=True,
                             has_residual=True,
                             mask=None):

        _, T, input_num_units = queries.get_shape().as_list()
        if num_units is None:
            num_units = input_num_units

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
        if has_residual:
            if num_units != input_num_units:
                V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)
            else:
                V_res = queries

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

        # mask = tf.squeeze(mask, 2)  # [bsz, T]
        masks = tf.tile(mask, [num_heads, 1])  # [bsz * n_head, T]
        key_masks = tf.tile(tf.expand_dims(masks, 1), [1, tf.shape(Q_)[1], 1])  # [bsz * n_head, T, T]

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

        paddings = tf.ones_like(weights) * (-2 ** 32 + 1)
        weights = tf.where(tf.not_equal(key_masks, 0), weights, paddings)

        # Activation
        weights = tf.nn.softmax(weights)

        # Dropouts
        # if is_drop:
        weights = tf.layers.dropout(weights, rate=self.drop_rate)

        # Weighted sum
        outputs = tf.matmul(weights, V_)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        # Residual connection
        if has_residual:
            outputs += V_res

        outputs = tf.nn.relu(outputs)
        # Normalize
        outputs = self.layer_normalize(outputs)

        return outputs

    def get_sequence_embedding_with_transformer(self, scope_name, seq_ids, T=50, target_embedding=None,mode='mean',k=25,flag='attention', seq_embedding=None, user_embedding=None):

        mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        mask = (1 - mask) * (-1e9)
        seq_embedding = seq_embedding + mask
        paddings = tf.ones([tf.shape(seq_embedding)[0], T, tf.shape(seq_embedding)[2]]) * (-1e9)
        embedding_concat = tf.concat([seq_embedding, paddings], 1)
        seq_emb = tf.slice(embedding_concat, [0, 0, 0], [-1, T, -1])

        # self-attention encoding
        seq_emb = self.self_multi_head_attn(queries=seq_emb,
                                            keys=seq_emb,
                                            values=seq_emb,
                                            num_units=self.params['block_shape'][0],
                                            num_heads=self.params['heads'],
                                            is_drop=self.params['is_drop'],
                                            dropout_keep_prob=self.params['dropout_keep_prob'],
                                            mask=seq_ids
                                            )

        if mode == 'concat':
            flat = tf.reshape(seq_emb, shape=[-1, self.params['block_shape'][-1] * T])
        elif mode == 'sum':
            flat = tf.reduce_sum(seq_emb, axis=1)
        elif mode == 'mean':
            flat = tf.reduce_mean(seq_emb, axis=1)

        # target attention pooling
        elif mode == 'attention' and target_embedding != None:
            flat = self.attention_sum_pooling(target_embedding, seq_emb, dims=self.params['block_shape'][-1], flag=flag)
        else:
            flat = tf.reduce_mean(seq_emb, axis=1)
        return flat

    def get_sequence_embedding_with_transformer_feature(self, scope_name, seq_ids, T=50, target_embedding=None,mode='mean',k=25,flag='attention', seq_embedding=None, target_seq_embedding=None, user_embedding=None):

        mask = tf.cast(tf.not_equal(seq_ids, 0), tf.float32)
        mask = tf.expand_dims(mask, axis=-1)
        mask = (1 - mask) * (-1e9)
        for i in range(len(seq_embedding)):
            seq_embedding[i] = seq_embedding[i] + mask
            paddings = tf.ones([tf.shape(seq_embedding[i])[0], T, tf.shape(seq_embedding[i])[2]]) * (-1e9)
            embedding_concat = tf.concat([seq_embedding[i], paddings], 1)
            seq_embedding[i] = tf.slice(embedding_concat, [0, 0, 0], [-1, T, -1])
            seq_embedding[i] = tf.expand_dims(seq_embedding[i], 2)

        seq_emb = tf.concat(seq_embedding, 2)

        target_seq_embedding = target_seq_embedding + mask
        paddings = tf.ones([tf.shape(target_seq_embedding)[0], T, tf.shape(target_seq_embedding)[2]]) * (-1e9)
        embedding_concat = tf.concat([target_seq_embedding, paddings], 1)
        target_seq_embedding = tf.slice(embedding_concat, [0, 0, 0], [-1, T, -1])

        user_embedding = tf.tile(tf.expand_dims(user_embedding, 1), [1,T,1])
        seq_concat_targetseq_user_emb = tf.concat([target_seq_embedding, user_embedding], -1)

        seq_concat_targetseq_user_emb = tf.layers.dense(seq_concat_targetseq_user_emb, self.params['dims'], activation=tf.nn.relu)
        seq_attn = tf.nn.softmax(tf.matmul(tf.expand_dims(seq_concat_targetseq_user_emb, 2), tf.transpose(seq_emb, [0, 1, 3, 2])))

        # seq_concat_targetseq_user_emb = tf.tile(tf.expand_dims(seq_concat_targetseq_user_emb, 2), [1,1,4,1])
        # seq_concat_targetseq_user_sourceseq_emb = tf.concat([seq_concat_targetseq_user_emb, seq_emb], -1)
        # seq_attn = tf.transpose(tf.nn.softmax(tf.layers.dense(seq_concat_targetseq_user_sourceseq_emb, 1, activation=tf.nn.relu), 2), [0,1,3,2])



        seq_emb = tf.reshape(tf.matmul(seq_attn, seq_emb), [-1,T,self.params['dims']])

        # self-attention encoding
        seq_emb = self.self_multi_head_attn(queries=seq_emb,
                                            keys=seq_emb,
                                            values=seq_emb,
                                            num_units=self.params['block_shape'][0],
                                            num_heads=self.params['heads'],
                                            is_drop=self.params['is_drop'],
                                            dropout_keep_prob=self.params['dropout_keep_prob'],
                                            mask=seq_ids
                                            )

        if mode == 'concat':
            flat = tf.reshape(seq_emb, shape=[-1, self.params['block_shape'][-1] * T])
        elif mode == 'sum':
            flat = tf.reduce_sum(seq_emb, axis=1)
        elif mode == 'mean':
            flat = tf.reduce_mean(seq_emb, axis=1)

        # target attention pooling
        elif mode == 'attention' and target_embedding != None:
            flat = self.attention_sum_pooling(target_embedding, seq_emb, dims=self.params['block_shape'][-1], flag=flag)
        else:
            flat = tf.reduce_mean(seq_emb, axis=1)
        return flat


    def attention_sum_pooling(self, query, key, dims, flag):
        """
        :param query: [batch_size, query_size] -> [batch_size, time, query_size]
        :param key:   [batch_size, time, key_size]
        :return:      [batch_size, 1, time]
            query_size should keep the same dim with key_size
        """
        query = tf.expand_dims(query, 1)
        key_transpose = tf.transpose(key, [0, 2, 1])
        align = tf.matmul(query, key_transpose)
        align = tf.nn.softmax(align)

        output = tf.matmul(align, key)  # [batch_size, 1, time] * [batch_size, time, key_size] -> [batch_size, 1, key_size]
        # output = tf.squeeze(output)
        output = tf.reshape(output, [-1, dims])
        return output

    def multilabel_parse(self, label, features):
        msLabel = features['isMeishiIntent']
        labels = tf.where(tf.equal(msLabel, 1),
                          label,
                          tf.minimum(label, 3))
        labels = tf.cast(labels, tf.float32)
        return tf.minimum(tf.divide(labels, 5.0), 1.0)

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)

    def get_sku_emb(self, sess):
        return sess.run(self.emb_sku)


    def train(self, features, sess, data_type):


        loss, train_op = sess.run([self.loss, self.train_op],
                                  feed_dict={self.skuid: features[0], self.cate1id: features[1], self.cate2id:features[2],
                                  self.cityid:features[3], self.review_count_id:features[4], self.starsid:features[5],
                                  self.hisids:features[6], self.hiscate1ids:features[7], self.hiscate2ids:features[8],
                                  self.hiscityids:features[9],self.hisreviewids:features[10],self.hisstarsids:features[11],self.label:features[12],
                                  self.uid:features[13],self.review_count_id_u:features[14],self.year_id:features[15],self.stars_id_u:features[16],
                                    self.drop_rate: 0.0})

        return loss, train_op

    def test(self, features, sess, data_type):

        return sess.run([tf.nn.sigmoid(self.final_layer_out)], feed_dict={self.skuid: features[0], self.cate1id: features[1], self.cate2id:features[2],
                                  self.cityid:features[3], self.review_count_id:features[4], self.starsid:features[5],
                                  self.hisids:features[6], self.hiscate1ids:features[7], self.hiscate2ids:features[8],
                                  self.hiscityids:features[9],self.hisreviewids:features[10],self.hisstarsids:features[11],self.label:features[12],
                                  self.uid:features[13],self.review_count_id_u:features[14],self.year_id:features[15],self.stars_id_u:features[16],
                                    self.drop_rate: 0.0})
