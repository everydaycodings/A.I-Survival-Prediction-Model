"@Author: @everydaycodings"
"A.I with Tensorflow To Predict Survival Chance with Given Data."
"Build a linear model with Estimators"

# Importing The Module
import pandas as pd
import tensorflow as tf
from IPython.display import clear_output

# Load The Database
dftrain = pd.read_csv("datasets/train.csv")
dfeval = pd.read_csv("datasets/eval.csv")
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# Making Base Feature Columns
CATEGORICAL_COLUMNS = ["sex", "n_siblings_spouses", "parch", "class", "deck",
                       "embark_town", "alone"]
NUMERIC_COLUMNS = ["age", "fare"]

feature_colums = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_colums.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_colums.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# Code block which breaks our datasets into batches and epochs
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Training a model is just a single command using the tf.estimator API
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_colums)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# Printing Out The Result
clear_output()
print("Accuracy_Rate:",result["accuracy"])

result = list(linear_est.predict(eval_input_fn))
print(dfeval.loc[4])
sorn = y_eval.loc[4]
print()
if sorn == 1:
    print("Not Survived")
elif sorn == 2:
    print("Survived")

print("Prediction_Rate: ",result[4]["probabilities"][1])
