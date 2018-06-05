# coding:utf8
# 原始数据经过get_criteo_feature.py后，生成类似libsvm的格式
# 特征维度不变（40）
# continous特征为 number:value
# categorical特征为 code:1

import pandas as pd
import tensorflow as tf

df = pd.read_table('va.libsvm', sep=' ', header=None)

# continous特征取value
for i in range(2, 14):
    df[i] = df[i].map(lambda x: x.split(':')[1])

# categorical特征取code
for i in range(14, 40):
    df[i] = df[i].map(lambda x: x.split(':')[0])

df.to_csv('sample.csv', header=0, index=0)

C_COLUMNS = ['I' + str(i) for i in range(1, 14)]
D_COLUMNS = ['C' + str(i) for i in range(14, 40)]
LABEL_COLUMN = 'is_click'
CSV_COLUMNS = [LABEL_COLUMN] + C_COLUMNS + D_COLUMNS

C_COLUMN_DEFAULTS = [[0.0] for i in range(13)]
D_COLUMN_DEFAULTS = [[0] for i in range(26)]
CSV_COLUMN_DEFAULTS = CSV_COLUMN_DEFAULTS + C_COLUMN_DEFAULTS + D_COLUMN_DEFAULTS

def input_fn(filenames, num_epochs, batch_size=1):
    def parse_csv(line):
        print('Parsing', filenames)
        columns = tf.decode_csv(line, record_defaults=CSV_COLUMN_DEFAULTS)
        features = dict(zip(CSV_COLUMNS, columns))
        labels = features.pop(LABEL_COLUMN)
        return features, labels
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.map(parse_csv, num_parallel_calls=10).prefetch(500000)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return features, labels

# test
#features, label = input_fn('sample.csv', 1, 1)
#sess = tf.Session()
#sess.run(features)

def build_feature():
    deep_cbc = [tf.feature_column.numeric_column(colname) for colname in C_COLUMNS]
    deep_dbc = [tf.feature_column.categorical_column_with_identity(key=colname, num_buckets=10000, default_value=0) for colname in D_COLUMNS]
    deep_emb = [tf.feature_column.embedding_column(c, dimension=100)  for c in deep_dbc]
    wide_columns = deep_cbc + deep_dbc
    deep_columns = deep_cbc + deep_emb
    
    return wide_columns, deep_columns

def build_estimator(wide_columns, deep_columns):
    hidden_units = [128, 64, 32]
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units)
    return estimator


wide_columns, deep_columns = build_feature()

w_n_d = build_estimator(wide_columns, deep_columns)

tf.logging.set_verbosity(tf.logging.INFO)

train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn("sample.csv", 1, batch_size=5))
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn("sample.csv", 1, batch_size=1))

tf.estimator.train_and_evaluate(w_n_d, train_spec, eval_spec)