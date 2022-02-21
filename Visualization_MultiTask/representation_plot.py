import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from Visualization_MultiTask.model.mmoe import Mmoe
from Visualization_MultiTask.model.maee import Maee

import re
from datetime import datetime
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


import seaborn as sns
sns.set_style('darkgrid')
sns.set_context("notebook", font_scale=0.5,
                rc={"lines.linewidth": 0.5})
def scatter(x, colors, name):

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc0 = ax.scatter(x[colors == 0, 0], x[colors == 0, 1], lw=0, s=2,
                    c='r')
    sc1 = ax.scatter(x[colors == 1, 0], x[colors == 1, 1], lw=0, s=2,
                     c='b')
    sc2 = ax.scatter(x[colors == 2, 0], x[colors == 2, 1], lw=0, s=2,
                     c='g')
    sc3 = ax.scatter(x[colors == 3, 0], x[colors == 3, 1], lw=0, s=2,
                     c='y')
    sc4 = ax.scatter(x[colors == 4, 0], x[colors == 4, 1], lw=0, s=2,
                     c='m')
    sc5 = ax.scatter(x[colors == 5, 0], x[colors == 5, 1], lw=0, s=2,
                     c='c')
    if name == 'maee':
        ax.legend((sc0,sc1,sc2,sc3,sc4,sc5),('Self_Employed_Y', 'Self_Employ_N', 'In_College_Y', 'In_College_N', 'In_College_U', 'Common'), loc="upper left", bbox_to_anchor=(-0.1,1.1), prop={'size': 16}, markerscale=5)
    elif name == 'mmoe':
        ax.legend((sc0, sc1, sc2, sc3, sc4, sc5), ('E1', 'E2', 'E3', 'E4', 'E5', 'E6'), loc="upper left", bbox_to_anchor=(-0.1,1.1),
                  prop={'size': 16}, markerscale=5)

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    expert_name = {"0": "Self_Employed_Y", "1": "Self_Employed_N", "2": "In_College_Y", "3": "In_College_N", "4": "In_College_U", "5": "Common"}
    # We add the labels for each digit.

    txts = []
    #offsets = [(-2,-2),(1,0),(1,1),(-1,0),(0,0),(0,0)]
    scale = [1.7, 1, 0.15, 1.7, 1.1, 1]
    for i in range(6):
        # Position of each label.
        xtext, ytext = np.mean(x[colors == i, :], axis=0)
        print(xtext,ytext)

        if name == 'maee':
            txt = ax.text(xtext, ytext, expert_name[str(i)], fontsize=16)
        elif name == 'mmoe':
            txt = ax.text(xtext * scale[i], ytext * scale[i], "E" + str(i+1), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=7.5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc0, sc1, sc2, sc3, sc4, sc5, txts

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data, model_name):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]
        self.cur_best_validation_auc_0 = 0
        self.cur_best_test_auc_0 = 0
        self.cur_best_validation_auc_1 = 0
        self.cur_best_test_auc_1 = 0
        self.best_epoch = 0
        self.name = model_name
        self.res = None

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        print('################### Final Results ##################')
        print(
            '\nbest epoch: {}, Task {}--best-AUC-Validation: {}, bset-AUC-Test: {}\n Task {}--best-AUC-Validation: {}, bset-AUC-Test: {}'.format(
                self.best_epoch,
                self.model.output_names[0],
                round(self.cur_best_validation_auc_0, 4),
                round(self.cur_best_test_auc_0, 4),
                self.model.output_names[1],
                round(self.cur_best_validation_auc_1, 4),
                round(self.cur_best_test_auc_1, 4)
            )
        )
        X = []
        Y = []
        sample_indices = np.random.permutation(np.arange(np.shape(self.res)[0]))[0:10000]
        for i in range(6):
            expert_representation = self.res[sample_indices, :, i]
            expert_label = np.zeros((np.shape(expert_representation)[0],)) + i
            X.append(expert_representation)
            Y.append(expert_label)
        X_sample = np.concatenate(X, axis=0)
        Y_sample = np.concatenate(Y, axis=0)


        digits_proj = TSNE(random_state=SEED).fit_transform(X_sample)
        print(digits_proj)
        scatter(digits_proj, Y_sample, self.name)
        foo_fig = plt.gcf()  # 'get current figure'
        foo_fig.savefig(self.name + '6_multi.pdf', format='pdf', dpi=1000)
        plt.show()


        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)
        index = 0
        output_name_0 = self.model.output_names[index]
        validation_roc_auc_0 = roc_auc_score(self.validation_Y[index], validation_prediction[index])
        test_roc_auc_0 = roc_auc_score(self.test_Y[index], test_prediction[index])
        print(
            '\nROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                output_name_0, round(validation_roc_auc_0, 4),
                output_name_0, round(test_roc_auc_0, 4)
            )
        )
        index = 1
        output_name_1 = self.model.output_names[index]
        validation_roc_auc_1 = roc_auc_score(self.validation_Y[index], validation_prediction[index])
        test_roc_auc_1 = roc_auc_score(self.test_Y[index], test_prediction[index])
        print(
            '\nROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                output_name_1, round(validation_roc_auc_1, 4),
                output_name_1, round(test_roc_auc_1, 4)
            )
        )

        output_representation = self.model.get_layer(self.name).output[1]
        model2 = Model(inputs=[self.model.input], outputs=output_representation)
        res = model2.predict(self.test_X)
        print(np.shape(res))


        if self.cur_best_validation_auc_0 <= validation_roc_auc_0:
            self.cur_best_validation_auc_0 = validation_roc_auc_0
            self.cur_best_test_auc_0 = test_roc_auc_0
            self.cur_best_validation_auc_1 = validation_roc_auc_1
            self.cur_best_test_auc_1 = test_roc_auc_1
            self.best_epoch = epoch
            self.res = res

        print(
            '\nbest epoch: {}, Task {}--best-AUC-Validation: {}, bset-AUC-Test: {}\n Task {}--best-AUC-Validation: {}, bset-AUC-Test: {}'.format(
                self.best_epoch,
                output_name_0,
                round(self.cur_best_validation_auc_0, 4),
                round(self.cur_best_test_auc_0, 4),
                output_name_1,
                round(self.cur_best_validation_auc_1, 4),
                round(self.cur_best_test_auc_1, 4)
            )
        )


        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


def data_preparation():
    # datasets are downloaded from
    # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
    # https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
    # see https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29 for details

    column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour', 'hs_college',
                    'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member',
                    'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses', 'stock_dividends',
                    'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat', 'det_hh_summ',
                    'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                    'num_emp', 'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                    'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked', 'year', 'income_50k']

    # Load the dataset in Pandas
    train_df = pd.read_csv(
        './data/census-income.data.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )
    other_df = pd.read_csv(
        './data/census-income.test.gz',
        delimiter=',',
        header=None,
        index_col=None,
        names=column_names
    )

    # First group of tasks according to the paper
    label_columns = ['income_50k', 'marital_stat']

    # One-hot encoding categorical columns
    categorical_columns = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
    train_raw_labels = train_df[label_columns]
    other_raw_labels = other_df[label_columns]
    transformed_train = pd.get_dummies(train_df.drop(label_columns, axis=1), columns=categorical_columns)
    transformed_other = pd.get_dummies(other_df.drop(label_columns, axis=1), columns=categorical_columns)
    # Filling the missing column in the other set
    transformed_other['det_hh_fam_stat_ Grandchild <18 ever marr not in subfamily'] = 0
    # One-hot encoding categorical labels
    train_income = to_categorical((train_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)
    other_income = to_categorical((other_raw_labels.income_50k == ' 50000+.').astype(int), num_classes=2)

    train_marital = to_categorical((train_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)
    other_marital = to_categorical((other_raw_labels.marital_stat == ' Never married').astype(int), num_classes=2)

    dict_outputs = {
        'income': train_income.shape[1],
        'marital': train_marital.shape[1]
    }
    dict_train_labels = {
        'income': train_income,
        'marital': train_marital
    }
    dict_other_labels = {
        'income': other_income,
        'marital': other_marital
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    # Split the other dataset into 1:1 validation to test according to the paper
    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info


def generate_expert_layers(model_config, input_layer, train_data):
    model_name = model_config["name"]

    if model_name == 'mmoe':
        expert_layer = Mmoe(
            units=model_config["units"],
            num_experts=model_config["num_experts"],
            num_tasks=model_config["num_tasks"]
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
                index_list = get_feature_index(train_data, field_name, "discrete")
                field_expert_index_list.append(index_list)
                field_expert_num_list.append(len(index_list))
                field_expert_type_list.append(model_config["field_types"][index])
                field_expert_names.append(field_name)
            elif model_config["field_types"][index] == "continuous":
                index_list = get_feature_index(train_data, field_name, "continuous")
                field_expert_index_list.append(index_list)
                field_expert_num_list.append(len(model_config["boundaries"][field_name])+1)
                field_expert_type_list.append(model_config["field_types"][index])
                field_expert_boundaries[field_name] = model_config["boundaries"][field_name]
                field_expert_names.append(field_name)
        print("MAEE total num of experts: {}".format(1+sum(field_expert_num_list)))
        expert_layer = Maee(
            units=model_config["units"],
            num_fields=num_fields,
            field_expert_num_list=field_expert_num_list,
            field_expert_index_list=field_expert_index_list,
            field_expert_type_list=field_expert_type_list,
            field_expert_boundaries=field_expert_boundaries,
            field_expert_names=field_expert_names,
            num_tasks=model_config["num_tasks"]
        )(input_layer)
    else:
        expert_layer = input_layer
    return expert_layer


def get_feature_index(train_data, feature_name, type):
    index_list = []
    if type == "continuous":
        for index, key in enumerate(train_data.keys()):
            if feature_name == key:
                index_list.append(index)
                break
    elif type == "discrete":
        for index, key in enumerate(train_data.keys()):
            feature_name_prefix = feature_name + "_"
            if feature_name_prefix in key:
                index_list.append(index)
    return index_list


def main(model_config, train_config):
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_features = train_data.shape[1]
    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))


    # Set up the input layer
    input_layer = Input(shape=(num_features,))
    # Set up expert layer
    expert_layers = generate_expert_layers(model_config, input_layer, train_data)[0]

    output_layers = []
    # Build tower layer from expert layers for each task
    for index, task_layer in enumerate(expert_layers):
        tower_layer_1 = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        tower_layer_2 = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(tower_layer_1)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer_2)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam(learning_rate=train_config["learning_rate"])
    model.compile(
        loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()
    print(train_data)
    print(train_label)
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

    # Set up model_config for different models

    # 1. MMOE(6 experts): income task test AUC=0.9417, marital task test AUC=0.9360, lr=0.01, units=8
    '''
    model_config = {
        "name": "mmoe",
        "units": 8,
        "num_experts": 6,
        "num_tasks": 2
    }
    '''

    # 4. MAEE (2 fields, 6 experts): income task test AUC=0.9460, marital task test AUC=0.9635, lr=0.005, units=16

    model_config = {
        "name": "maee",
        "units": 16,
        "num_fields": 2,
        "field_names": ["own_or_self", "hs_college"],
        "field_types": ["continuous", "discrete"],
        "boundaries": {
            "own_or_self": [0.5]
        },
        "num_tasks": 2
    }


    # tune training params accordingly
    train_config = {
        "learning_rate": 0.005,
        "batch_size": 256,
        "epochs": 100
    }

    main(model_config, train_config)
    print(model_config, train_config)
