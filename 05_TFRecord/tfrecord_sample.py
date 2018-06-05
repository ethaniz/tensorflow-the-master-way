# coding:utf8

import tensorflow as tf
from tfrecorder import TFrecorder

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

df = pd.DataFrame({'name':['image', 'label'],
                  'type':['float32', 'int64'],
                  'shape':[(784,),()],
                  'isbyte':[False, False],
                  'length_type':['fixed', 'fixed'],
                  'default':[np.NaN, np.NaN]})

tfr = TFrecorder()

path = 'mnist_tfrecord/train/train'

num_examples_per_file = 1000
num_so_far = 0

writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' % (path, num_so_far, num_examples_per_file))

for i in np.arange(dataset.num_examples):
    features = {}
    tfr.feature_writer(df.iloc[0], dataset.images[i], features)
    tfr.feature_writer(df.iloc[1], dataset.labels[i], features)
    
    tf_features = tf.train.Features(feature=features)
    tf_example = tf.train.Example(features=tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)
    
    if i%num_examples_per_file == 0 and i!=0:
        writer.close()
        num_so_far = i
        writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
        print('saved %s%s_%s.tfrecord' % (path, num_so_far, i+num_examples_per_file))

writer.close()
data_info_path = 'mnist_tfrecord/data_info.csv'
df.to_csv(data_info_path, index=False)


# 用该方法写测试集的tfrecord文件
dataset = mnist.test
path = 'mnist_tfrecord/test/test'
# 每个tfrecord文件写多少个样本
num_examples_per_file = 1000
# 当前写的样本数
num_so_far = 0
# 要写入的文件
writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, num_examples_per_file))
# 写多个样本
for i in np.arange(dataset.num_examples):
    # 要写到tfrecord文件中的字典
    features = {}
    # 写一个样本的图片信息存到字典features中
    tfr.feature_writer(df.iloc[0], dataset.images[i], features)
    # 写一个样本的标签信息存到字典features中
    tfr.feature_writer(df.iloc[1], dataset.labels[i], features)
    
    tf_features = tf.train.Features(feature= features)
    tf_example = tf.train.Example(features = tf_features)
    tf_serialized = tf_example.SerializeToString()
    writer.write(tf_serialized)
    # 每写了num_examples_per_file个样本就令生成一个tfrecord文件
    if i%num_examples_per_file ==0 and i!=0:
        writer.close()
        num_so_far = i
        writer = tf.python_io.TFRecordWriter('%s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
        print('saved %s%s_%s.tfrecord' %(path, num_so_far, i+num_examples_per_file))
writer.close()

def input_fn_maker(path, data_info_path, shuffle=False, batch_size=1, epoch=1, padding=None):
    def input_fn():
        filenames = tfr.get_filenames(path, shuffle=shuffle)
        dataset = tfr.get_dataset(paths=filenames, data_info=data_info_path, shuffle=shuffle,
                                 batch_size=batch_size, epoch=epoch, padding=padding)
        return dataset.make_one_shot_iterator().get_next()
    return input_fn

padding_info = ({'image': [784,], 'label':[]})

test_input_fn = input_fn_maker('mnist_tfrecord/test/', 'mnist_tfrecord/data_info.csv', padding=padding_info)
train_input_fn = input_fn_maker('mnist_tfrecord/train/', 'mnist_tfrecord/data_info.csv', shuffle=True, batch_size=512, padding=padding_info)
train_eval_fn = input_fn_maker('mnist_tfrecord/train/', 'mnist_tfrecord/data_info.csv', batch_size=512, padding=padding_info)

