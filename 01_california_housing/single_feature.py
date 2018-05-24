# -*- coding:utf8 -*-

import math
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.ERROR)

california_housing_dataframe = pd.read_csv(
    "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", 
    sep=","
    )

# np.random.permutation相比np.random.shuffle，前者会生成新对象
california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index)
)

california_housing_dataframe['median_house_value'] /= 1000

#print(california_housing_dataframe)

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
    periods = 10
    step_per_period = steps / periods

    my_feature = input_feature
    my_feature_data = california_housing_dataframe[[my_feature]]
    my_label = 'median_house_value'
    targets = california_housing_dataframe[my_label]

    feature_columns = [tf.feature_column.numeric_column(my_feature)]

    training_input_fn = lambda: my_input_fn(my_feature_data, targets, batch_size, shuffle=True, num_epochs=None)
    prediction_input_fn = lambda: my_input_fn(my_feature_data, targets, 1, False, 1)

    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )

    print("Training model...")
    print("RMSE (on training data):")

    root_mean_squared_errors = []
    for period in range(0, periods):
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=step_per_period
        )
        predictions = linear_regressor.predict(input_fn=prediction_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])

        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets)
        )
        print("period %02d: %0.2f" % (period, root_mean_squared_error))
        
        root_mean_squared_errors.append(root_mean_squared_error)

    print("Model training finished!")

    #plt.subplot(1, 2, 2)
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_errors)
    plt.show()

train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5
)