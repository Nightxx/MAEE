import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K
from Visualization_SingleTask.model.moe import Moe
from Visualization_SingleTask.model.maee import Maee
import re
from datetime import datetime
import math
import scipy.stats as stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.random.set_seed(SEED)

# Fix op-parallelism parameters for reproducibility
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=12, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

nDCG_res = []
update_gate_res = []


# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data, model_name):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]
        self.cur_best_validation_auc = 0
        self.cur_best_test_auc = 0
        self.best_epoch = 0
        self.name = model_name
        self.res1 = None
        self.res2 = None

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        print('################### Final Results ##################')
        print(
            '\nbest epoch: {}--best-AUC-Validation: {}, bset-AUC-Test: {}'.format(
                self.best_epoch,
                round(self.cur_best_validation_auc, 4),
                round(self.cur_best_test_auc, 4)
            )
        )
        update_gate_res.append(np.mean(self.res1[self.res2[:, 0] == True, 0], axis=0))
        update_gate_res.append(np.mean(self.res1[self.res2[:, 1] == True, 0], axis=0))
        update_gate_res.append(np.mean(self.res1[self.res2[:, 2] == True, 1], axis=0))
        update_gate_res.append(np.mean(self.res1[self.res2[:, 3] == True, 1], axis=0))
        update_gate_res.append(np.mean(self.res1[self.res2[:, 4] == True, 1], axis=0))

        print(update_gate_res)
        plot_gate_stats()
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)
        output_name = self.model.output_names[0]
        validation_roc_auc = roc_auc_score(self.validation_Y, validation_prediction)
        test_roc_auc = roc_auc_score(self.test_Y, test_prediction)
        print(
            '\nROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                output_name, round(validation_roc_auc, 4),
                output_name, round(test_roc_auc, 4)
            )
        )


        output1 = self.model.get_layer(self.name).output[2]
        model2 = Model(inputs=[self.model.input], outputs=output1)
        res1 = model2.predict(self.test_X)

        output2 = self.model.get_layer(self.name).output[3]
        model3 = Model(inputs=[self.model.input], outputs=output2)
        res2 = model3.predict(self.test_X)
        if self.cur_best_validation_auc <= validation_roc_auc:
            self.cur_best_validation_auc = validation_roc_auc
            self.cur_best_test_auc = test_roc_auc
            self.best_epoch = epoch
            self.res1 = res1
            self.res2 = res2

        print(
            '\nbest epoch: {}, Task {}--best-AUC-Validation: {}, bset-AUC-Test: {}'.format(
                self.best_epoch,
                output_name,
                round(self.cur_best_validation_auc, 4),
                round(self.cur_best_test_auc, 4)
            )
        )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def data_preparation():
    # The column names are from
    # http://files.grouplens.org/datasets/movielens/ml-1m.zip

    # User data set
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./data/users.dat', sep='::', encoding='latin1', header=None, names=users_title,
                          engine='python')

    # coding zipcode
    zip_map = {val: ii for ii, val in enumerate(sorted(set(users['Zip-code'])))}
    num_zip = max(zip_map.values()) + 1
    users['Zip-code'] = users['Zip-code'].map(zip_map)
    num_uid = max(users['UserID']) + 1

    # Movie data set
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./data/movies.dat', sep='::', encoding='latin1', header=None, names=movies_title,
                           engine='python')
    num_mid = max(movies['MovieID']) + 1

    # movie title transform
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(sorted(set(movies['Title'])))}
    year = {val: int(int(pattern.match(val).group(2)) / 5) * 5 for ii, val in enumerate(sorted(set(movies['Title'])))}
    movies['MovieYear'] = movies['Title'].map(year)
    movies['Title'] = movies['Title'].map(title_map)
    movie_id_list = movies['MovieID'].values
    print(movie_id_list)
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)

    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(sorted(title_set))}

    num_title_word = max(title2int.values()) + 1

    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(sorted(set(movies['Title'])))}

    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])

    for loc in range(title_count):
        movies['Title_' + str(loc)] = movies['Title'].map(title_map).map(lambda x: x[loc])

    movies = movies.drop(['Title'], axis=1)

    # movie genre transform
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    for genres in sorted(genres_set):
        movies['Genres_' + genres] = movies['Genres'].str.contains(genres).astype(int)
    movies = movies.drop(['Genres'], axis=1)

    # rating data set
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./data/ratings.dat', sep='::', encoding='latin1', header=None, names=ratings_title,
                            engine='python')
    rating_year_map = {val: datetime.fromtimestamp(val).year for ii, val in enumerate(ratings['timestamps'])}
    rating_month_map = {val: datetime.fromtimestamp(val).month for ii, val in enumerate(ratings['timestamps'])}
    rating_day_map = {val: datetime.fromtimestamp(val).day for ii, val in enumerate(ratings['timestamps'])}
    rating_hour_map = {val: datetime.fromtimestamp(val).hour for ii, val in enumerate(ratings['timestamps'])}
    ratings['rating_year'] = ratings['timestamps'].map(rating_year_map)
    ratings['rating_month'] = ratings['timestamps'].map(rating_month_map)
    ratings['rating_day'] = ratings['timestamps'].map(rating_day_map)
    ratings['rating_hour'] = ratings['timestamps'].map(rating_hour_map)
    ratings = ratings.drop(['timestamps'], axis=1)

    # merge three tables and split into training, validation, and test set
    raw_data = pd.merge(pd.merge(ratings, users), movies)

    categorical_columns = ['Gender', 'Age', 'JobID', 'MovieYear', 'rating_year', 'rating_month', 'rating_day',
                           'rating_hour']
    transformed_raw_data = pd.get_dummies(raw_data, columns=categorical_columns)

    categorical_columns.append('Genres')
    embedding_columns = ['UserID', 'MovieID', 'Zip-code', 'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Title_4',
                         'Title_5', 'Title_6', 'Title_7', 'Title_8', 'Title_9', 'Title_10', 'Title_11', 'Title_12',
                         'Title_13', 'Title_14']

    label_field = ['ratings']
    data, label_raw = transformed_raw_data.drop(label_field, axis=1), transformed_raw_data[label_field]
    label = to_categorical((label_raw['ratings'] > 4).astype(int), num_classes=2)

    indices = np.random.permutation(data.index)
    N = len(indices)
    train_indices = indices[:int(N * 0.8)]
    val_indices = indices[int(N * 0.8): int(N * 0.9)]
    test_indices = indices[int(N * 0.9):]
    train_data = data.iloc[train_indices]
    train_label = np.array(label[train_indices])
    val_data = data.iloc[val_indices]
    val_label = np.array(label[val_indices])
    test_data = data.iloc[test_indices]
    test_label = np.array(label[test_indices])

    # calculate each movie's ipv(number of 5-stars) and pv(number of ratings) on each group(F/M/Y/A/E)
    F_pv = {}
    F_ipv = {}
    M_pv = {}
    M_ipv = {}
    Youth_pv = {}
    Youth_ipv = {}
    Adult_pv = {}
    Adult_ipv = {}
    Elder_pv = {}
    Elder_ipv = {}

    movie_id_list = test_data['MovieID'].values
    gender_list = test_data['Gender_F'].values
    rating_list = test_label[:, 1]
    age_lists = []
    age_map = {0: "1", 1: "18", 2: "25", 3: "35", 4: "45", 5: "50", 6: "56"}
    for i in range(7):
        age_lists.append(test_data['Age_' + age_map[i]].values)

    for i in range(len(movie_id_list)):
        if gender_list[i] == 1:
            if movie_id_list[i] not in F_pv:
                F_pv[movie_id_list[i]] = 1
                F_ipv[movie_id_list[i]] = rating_list[i]
            else:
                F_pv[movie_id_list[i]] += 1
                F_ipv[movie_id_list[i]] += rating_list[i]
        else:
            if movie_id_list[i] not in M_pv:
                M_pv[movie_id_list[i]] = 1
                M_ipv[movie_id_list[i]] = rating_list[i]
            else:
                M_pv[movie_id_list[i]] += 1
                M_ipv[movie_id_list[i]] += rating_list[i]
        if age_lists[0][i] == 1 or age_lists[1][i] == 1:
            if movie_id_list[i] not in Youth_pv:
                Youth_pv[movie_id_list[i]] = 1
                Youth_ipv[movie_id_list[i]] = rating_list[i]
            else:
                Youth_pv[movie_id_list[i]] += 1
                Youth_ipv[movie_id_list[i]] += rating_list[i]
        elif age_lists[2][i] == 1 or age_lists[3][i] == 1 or age_lists[4][i] == 1:
            if movie_id_list[i] not in Adult_pv:
                Adult_pv[movie_id_list[i]] = 1
                Adult_ipv[movie_id_list[i]] = rating_list[i]
            else:
                Adult_pv[movie_id_list[i]] += 1
                Adult_ipv[movie_id_list[i]] += rating_list[i]
        elif age_lists[5][i] == 1 or age_lists[6][i] == 1:
            if movie_id_list[i] not in Elder_pv:
                Elder_pv[movie_id_list[i]] = 1
                Elder_ipv[movie_id_list[i]] = rating_list[i]
            else:
                Elder_pv[movie_id_list[i]] += 1
                Elder_ipv[movie_id_list[i]] += rating_list[i]

    # calculate each group's (female/male/youth/adult/elder) average rating scores
    F_score_list = []
    M_score_list = []
    Youth_score_list = []
    Adult_score_list = []
    Elder_score_list = []

    for movie_id in movie_id_list:
        F_score = 0
        M_score = 0
        Youth_score = 0
        Adult_score = 0
        Elder_score = 0

        if movie_id in F_pv:
            F_score = F_ipv[movie_id] / F_pv[movie_id]
        F_score_list.append(F_score)

        if movie_id in M_pv:
            M_score = M_ipv[movie_id] / M_pv[movie_id]
        M_score_list.append(M_score)

        if movie_id in Youth_pv:
            Youth_score = Youth_ipv[movie_id] / Youth_pv[movie_id]
        Youth_score_list.append(Youth_score)

        if movie_id in Adult_pv:
            Adult_score = Adult_ipv[movie_id] / Adult_pv[movie_id]
        Adult_score_list.append(Adult_score)

        if movie_id in Elder_pv:
            Elder_score = Elder_ipv[movie_id] / Elder_pv[movie_id]
        Elder_score_list.append(Elder_score)

    F_score_idx = [i[0] for i in sorted(enumerate(F_score_list), key=lambda x: x[1], reverse=True)]
    M_score_idx = [i[0] for i in sorted(enumerate(M_score_list), key=lambda x: x[1], reverse=True)]
    Youth_score_idx = [i[0] for i in sorted(enumerate(Youth_score_list), key=lambda x: x[1], reverse=True)]
    Adult_score_idx = [i[0] for i in sorted(enumerate(Adult_score_list), key=lambda x: x[1], reverse=True)]
    Elder_score_idx = [i[0] for i in sorted(enumerate(Elder_score_list), key=lambda x: x[1], reverse=True)]

    rank_Youth = stats.rankdata(Youth_score_list, method='min')
    rank_Adult = stats.rankdata(Adult_score_list, method='min')
    rank_Elder = stats.rankdata(Elder_score_list, method='min')
    r_ya = stats.kendalltau(rank_Youth, rank_Adult)[0]
    r_ye = stats.kendalltau(rank_Youth, rank_Elder)[0]
    r_ae = stats.kendalltau(rank_Adult, rank_Elder)[0]

    print(r_ya, r_ye, r_ae)
    Youth_oppo_agg_score_list = []
    for i in range(len(Adult_score_list)):
        Youth_oppo_agg_score_list.append(1 / (r_ya + r_ye) * (r_ya * Adult_score_list[i] + r_ye * Elder_score_list[i]))
    Adult_oppo_agg_score_list = []
    for i in range(len(Youth_score_list)):
        Adult_oppo_agg_score_list.append(1 / (r_ya + r_ae) * (r_ya * Youth_score_list[i] + r_ae * Elder_score_list[i]))
    Elder_oppo_agg_score_list = []
    for i in range(len(Youth_score_list)):
        Elder_oppo_agg_score_list.append(1 / (r_ae + r_ye) * (r_ae * Adult_score_list[i] + r_ye * Youth_score_list[i]))

    Youth_oppo_score_idx = [i[0] for i in
                            sorted(enumerate(Youth_oppo_agg_score_list), key=lambda x: x[1], reverse=True)]
    Adult_oppo_score_idx = [i[0] for i in
                            sorted(enumerate(Adult_oppo_agg_score_list), key=lambda x: x[1], reverse=True)]
    Elder_oppo_score_idx = [i[0] for i in
                            sorted(enumerate(Elder_oppo_agg_score_list), key=lambda x: x[1], reverse=True)]

    DCG_M = 0
    IDCG_M = 0

    DCG_F = 0
    IDCG_F = 0

    DCG_Y = 0
    IDCG_Y = 0

    DCG_A = 0
    IDCG_A = 0

    DCG_E = 0
    IDCG_E = 0

    K = [20, 40, 60]
    for k in K:
        for i in range(k):
            DCG_M += M_score_list[F_score_idx[i]] / math.log2(2 + i)
            IDCG_M += M_score_list[M_score_idx[i]] / math.log2(2 + i)
            DCG_F += F_score_list[M_score_idx[i]] / math.log2(2 + i)
            IDCG_F += F_score_list[F_score_idx[i]] / math.log2(2 + i)
            DCG_Y += Youth_score_list[Youth_oppo_score_idx[i]] / math.log2(2 + i)
            IDCG_Y += Youth_score_list[Youth_score_idx[i]] / math.log2(2 + i)
            DCG_A += Adult_score_list[Adult_oppo_score_idx[i]] / math.log2(2 + i)
            IDCG_A += Adult_score_list[Adult_score_idx[i]] / math.log2(2 + i)
            DCG_E += Elder_score_list[Elder_oppo_score_idx[i]] / math.log2(2 + i)
            IDCG_E += Elder_score_list[Elder_score_idx[i]] / math.log2(2 + i)
        nDCG_M = DCG_M / IDCG_M
        nDCG_F = DCG_F / IDCG_F
        nDCG_Y = DCG_Y / IDCG_Y
        nDCG_A = DCG_A / IDCG_A
        nDCG_E = DCG_E / IDCG_E
        nDCG_res.append([nDCG_F, nDCG_M, nDCG_Y, nDCG_A, nDCG_E])

    output_info = [label.shape[1], 'ratings']

    return train_data, train_label, val_data, val_label, test_data, test_label, categorical_columns, embedding_columns, num_uid, num_mid, num_zip, num_title_word, output_info

def plot_gate_stats():

    fig, ((ax_gender_gate, ax_gender_20, ax_gender_40, ax_gender_60),
          (ax_age_gate, ax_age_20, ax_age_40, ax_age_60)) = plt.subplots(figsize=(8,6), nrows=2,ncols=4)
    x_gender = ['Female', 'Male']
    x_age = ['Youth', 'Adult', 'Elder']
    y_gender_gate = update_gate_res[0:2]
    y_gender_20 = nDCG_res[0][0:2]
    y_gender_40 = nDCG_res[1][0:2]
    y_gender_60 = nDCG_res[2][0:2]
    y_age_gate = update_gate_res[2:5]
    y_age_20 = nDCG_res[0][2:5]
    y_age_40 = nDCG_res[1][2:5]
    y_age_60 = nDCG_res[2][2:5]
    ax_gender_gate.bar(x_gender, y_gender_gate, width=1, color=('silver', 'gray'), edgecolor="black", linewidth=1)
    ax_gender_gate.title.set_text('Update Gate(Gender)')
    ax_gender_20.bar(x_gender, y_gender_20, width=1, color=('silver', 'gray'), edgecolor="white", linewidth=2)
    ax_gender_20.title.set_text('nDCG@20(Gender)')
    ax_gender_40.bar(x_gender, y_gender_40, width=1, color=('silver', 'gray'), edgecolor="white", linewidth=2)
    ax_gender_40.title.set_text('nDCG@40(Gender)')
    ax_gender_60.bar(x_gender, y_gender_60, width=1, color=('silver', 'gray'), edgecolor="white", linewidth=2)
    ax_gender_60.title.set_text('nDCG@60(Gender)')
    ax_age_gate.bar(x_age, y_age_gate, width=1, color=('gainsboro', 'darkgray', 'dimgray'), edgecolor="black",
                  linewidth=1)
    ax_age_gate.title.set_text('Update Gate(Age)')
    ax_age_20.bar(x_age, y_age_20, width=1, color=('gainsboro', 'darkgray', 'dimgray'), edgecolor="white",
                  linewidth=2)
    ax_age_20.title.set_text('nDCG@20(Age)')
    ax_age_40.bar(x_age, y_age_40, width=1, color=('gainsboro', 'darkgray', 'dimgray'), edgecolor="white",
                  linewidth=2)
    ax_age_40.title.set_text('nDCG@40(Age)')
    ax_age_60.bar(x_age, y_age_60, width=1, color=('gainsboro', 'darkgray', 'dimgray'), edgecolor="white",
                  linewidth=2)
    ax_age_60.title.set_text('nDCG@60(Age)')
    fig.tight_layout()
    fig.savefig('update_gate.pdf', format='pdf', dpi=1000)
    plt.show()

def generate_expert_layers(model_config, input_layer, train_data, categorical_columns, embedding_columns, embedding_dim):
    model_name = model_config["name"]

    if model_name == 'dnn':
        expert_layer = input_layer
    elif model_name == 'moe':
        expert_layer = Moe(
            units=model_config["units"],
            num_experts=model_config["num_experts"],
            num_tasks=1
        )(input_layer)
    elif model_name == 'maee':
        field_expert_index_list = []
        field_expert_num_list = []
        field_expert_type_list = []
        field_expert_boundaries = {}
        field_expert_names = []
        num_fields = len(model_config["field_names"])
        for index, field_name in enumerate(model_config["field_names"]):
            if model_config["field_types"][index] == "discrete":
                index_list_net = []
                for j, field_net in enumerate(model_config["field_nets"][index]):
                    index_list_value = []
                    for k, field_value in enumerate(model_config["field_values"][index][j]):
                        feature_name = field_name + "_" + field_value
                        index_list_value.append(get_feature_index(train_data, feature_name, "discrete", categorical_columns, embedding_columns, embedding_dim))
                    index_list_net.append(index_list_value)
                field_expert_index_list.append(index_list_net)
                field_expert_num_list.append(len(index_list_net))
                field_expert_type_list.append(model_config["field_types"][index])
                field_expert_names.append(field_name)
            elif model_config["field_types"][index] == "continuous":
                index_list = [get_feature_index(train_data, field_name, "continuous", categorical_columns, embedding_columns, embedding_dim)]
                field_expert_index_list.append(index_list)
                field_expert_num_list.append(len(model_config["boundaries"][field_name])+1)
                field_expert_type_list.append(model_config["field_types"][index])
                field_expert_boundaries[field_name] = model_config["boundaries"][field_name]
                field_expert_names.append(field_name)

        print(field_expert_index_list)
        print(field_expert_num_list)
        print("MAEE total num of experts: {}".format(1+sum(field_expert_num_list)))
        expert_layer = Maee(
            units=model_config["units"],
            num_fields=num_fields,
            field_expert_num_list=field_expert_num_list,
            field_expert_index_list=field_expert_index_list,
            field_expert_type_list=field_expert_type_list,
            field_expert_boundaries=field_expert_boundaries,
            field_expert_names=field_expert_names,
            num_tasks=1
        )(input_layer)
    else:
        expert_layer = input_layer
    return expert_layer


def get_feature_index(train_data, feature_name, type, categorical_columns, embedding_columns, embedding_dim):
    loc = 0
    for index, key in enumerate(train_data.keys()):
        if feature_name == key:
            return loc
        if key in embedding_columns:
            loc += embedding_dim
        else:
            loc += 1
    return -1


def main(model_config, train_config):
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, categorical_columns, embedding_columns, num_uid, num_mid, num_zip, num_title_word, output_info = data_preparation()
    num_features = train_data.shape[1]
    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layers and embedding layers
    embedding_dim = 16

    embedding_layers = []
    input_layer = Input(shape=(num_features,),name="Input")
    # user id embedding
    input_layer_uid = input_layer[:,0]
    embedding_layer_uid = Reshape(target_shape=(embedding_dim, ))(Embedding(input_dim=num_uid,
                                    output_dim=embedding_dim,
                                    embeddings_initializer='VarianceScaling',
                                    input_length=1)(input_layer_uid))
    embedding_layers.append(embedding_layer_uid)

    # movie id embedding
    input_layer_mid = input_layer[:,1]
    embedding_layer_mid = Reshape(target_shape=(embedding_dim, ))(Embedding(input_dim=num_mid,
                                    output_dim=embedding_dim,
                                    embeddings_initializer='VarianceScaling',
                                    input_length=1)(input_layer_mid))
    embedding_layers.append(embedding_layer_mid)

    # zip code embedding
    input_layer_zip = input_layer[:,2]
    embedding_layer_zip = Reshape(target_shape=(embedding_dim, ))(Embedding(input_dim=num_zip,
                                    output_dim=embedding_dim,
                                    embeddings_initializer='VarianceScaling',
                                    input_length=1)(input_layer_zip))
    embedding_layers.append(embedding_layer_zip)

    # title word embedding
    title_embedding = Embedding(input_dim=num_title_word,
              output_dim=embedding_dim,
              embeddings_initializer='VarianceScaling',
              input_length=1)
    title_embeddings = []
    for i in range(15):
        input_layer_title = input_layer[:,i+3]
        embedding_layer_title = Reshape(target_shape=(embedding_dim, ))(title_embedding(input_layer_title))
        title_embeddings.append(embedding_layer_title)
    embedding_layers.append(tf.concat(title_embeddings, axis=1))

    # other one-hot features
    input_layer_others = input_layer[:,18:]
    embedding_layer_others = input_layer_others
    embedding_layers.append(embedding_layer_others)

    embedding_layer = tf.concat(embedding_layers, axis=1)

    # Set up expert layer
    expert_layer = generate_expert_layers(model_config, embedding_layer, train_data, categorical_columns, embedding_columns, embedding_dim)
    print(expert_layer)


    if model_config["name"] == 'moe':
        expert_layer = expert_layer[0]
    elif model_config["name"] == 'maee':
        expert_layer = expert_layer[0]
    # Build tower layer from expert layers
    tower_layer_1 = Dense(
        units=8,
        activation='relu',
        kernel_initializer=VarianceScaling())(expert_layer)
    tower_layer_2 = Dense(
        units=8,
        activation='relu',
        kernel_initializer=VarianceScaling())(tower_layer_1)
    output_layer = Dense(
        units=output_info[0],
        name=output_info[1],
        activation='softmax',
        kernel_initializer=VarianceScaling())(tower_layer_2)

    # Compile model
    model = Model(inputs=[input_layer], outputs=[output_layer])
    adam_optimizer = Adam(learning_rate=train_config["learning_rate"])
    model.compile(
        loss={'ratings': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()
    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label),
                model_name=model_config["name"]
            )
        ],
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"]
    )




if __name__ == '__main__':

    # 5. MAEE(2 field, 6 experts): test AUC=0.8194, lr=0.001, units=16
    model_config = {
        "name": "maee",
        "units": 16,
        "num_fields": 2,
        "field_names": ["Gender", "Age"],
        "field_nets": [["F", "M"],["1", "25", "50"]],
        "field_values": [[["F"], ["M"]], [["1", "18"], ["25", "35", "45"], ["50", "56"]]],
        "field_types": ["discrete", "discrete"],
    }

    # tune training params accordingly
    train_config = {
        "learning_rate": 0.001,
        "batch_size": 256,
        "epochs": 10
    }


    main(model_config, train_config)
    print(model_config, train_config)
