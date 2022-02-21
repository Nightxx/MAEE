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

from SingleTask.model.moe import Moe
from SingleTask.model.maee import Maee
import re
from datetime import datetime
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

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]
        self.cur_best_validation_auc = 0
        self.cur_best_test_auc = 0
        self.best_epoch = 0

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

        if self.cur_best_validation_auc <= validation_roc_auc:
            self.cur_best_validation_auc = validation_roc_auc
            self.cur_best_test_auc = test_roc_auc
            self.best_epoch = epoch

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
    # The dataset is downloaded form
    # https://files.grouplens.org/datasets/movielens/ml-1m.zip

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

    output_info = [label.shape[1], 'ratings']

    return train_data, train_label, val_data, val_label, test_data, test_label, categorical_columns, embedding_columns, num_uid, num_mid, num_zip, num_title_word, output_info


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
    input_layer = Input(shape=(num_features,))
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
                test_data=(test_data, test_label)
            )
        ],
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"]
    )


if __name__ == '__main__':

    # Set up model_config for different models

    # 1. DNN: test AUC=0.8102, lr=0.005
    '''
    model_config = {
        "name": "dnn"
    }
    '''

    # 2. MOE(3 experts): AUC=0.8176, lr=0.001, units=16
    '''
    model_config = {
        "name": "moe",
        "units": 16,
        "num_experts": 3
    }
    '''

    # 3. MOE(6 experts): AUC=0.8184, lr=0.001, units=4
    '''
    model_config = {
        "name": "moe",
        "units": 4,
        "num_experts": 6
    }
    '''

    # 4. MAEE(1 field, 3 experts): test AUC=0.8178, lr=0.001, units=16
    '''
    model_config = {
        "name": "maee",
        "units": 16,
        "num_fields": 1,
        "field_names": ["Gender"],
        "field_nets": [["F", "M"]],
        "field_values": [[["F"], ["M"]]],
        "field_types": ["discrete"],
    }
    '''

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
        "epochs": 30
    }

    main(model_config, train_config)
    print(model_config, train_config)
