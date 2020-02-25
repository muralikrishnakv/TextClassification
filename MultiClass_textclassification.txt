######################## LIBRARY IMPORTING BLOCK #####################################

# Deep NN - NLP Text Classifier Model building Template using using Tensorflow and Keras
# RANDOM INITIALIZATION TECHNIQUE
# Importing the Frequently used libraries for Deep NN - NLP Text Classifier Model building
# All libraries needed to be used in the program are initialized here

# Set __IS_LOCAL_RUN__ variable to TRUE, if script is needed to be executed in the current zeppelin notebook-
# -for unit testing purpose
# Set this variable to FALSE, post unit testing the script to make it ready for the training phase

__IS_LOCAL_RUN__ = True

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf
import keras

# import matplotlib.pyplot as plt
# import seaborn as sns


if __IS_LOCAL_RUN__:

    epoch = 50
    batch_size = 128
    optimizer_func = 'Adam'
    loss_func = 'binary_crossentropy'
    dataset_dir = '/mnt/ai/shared/datasets/worldbank_demo/'
    export_path_base = '/mnt/ai/shared/zeppelin_local_run/95:2/47:0/1536647579869'
    model_checkpoint_dir = '/mnt/ai/shared/models_repo/saved_model/keras/local_predict'
    summary_log_dir = '/mnt/ai/shared/models_repo/summary_log/95:2/47:0/tj99999999999999999'
    tensorboard_dir = '/mnt/ai/shared/tensorboard/95:2/47:0/1536647579869/'

else:

    # Arguments for the model execution
    parser = argparse.ArgumentParser(description='custom Deep NN model for Text Classification')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--optimizer_func', type=str, required=True, default='',
                        help='optimizer function for the model')
    parser.add_argument('--loss_func', type=str, required=True, default='', help='loss function for the model')
    parser.add_argument('--dataset_dir', type=str, required=True, default='', help='data directory')
    parser.add_argument('--export_path_base', type=str, required=True, default='', help='export directory')
    parser.add_argument('--model_checkpoint_dir', type=str, required=True,
                        default='/mnt/ai/shared/models_repo/saved_model/keras/local_predict',
                        help='model checkpoint directory')
    parser.add_argument('--summary_log_dir', type=str, required=True, default='',
                        help='tensorflow summary writer directory')
    parser.add_argument('--tensorboard_dir', type=str, required=True, default='', help='tensorboard directory')

    args = parser.parse_args()

    batch_size = args.batch_size
    epoch = args.epoch
    optimizer_func = args.optimizer_func
    loss_func = args.loss_func
    dataset_dir = args.dataset_dir
    export_path_base = args.export_path_base
    model_checkpoint_dir = args.model_checkpoint_dir
    summary_log_dir = args.summary_log_dir
    tensorboard_dir = args.tensorboard_dir


# for reproducibility . value can be edited.
np.random.seed(7)


####################### DATA DEFINITION BLOCK #####################################


# default values
# Path where the input dataset is present
working_directory = dataset_dir
input_data = working_directory + 'trainV7.csv'


####################### DATA PREPROCESSING BLOCK #################################

vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

# Dataset Pre-processing steps
# Read data from input file using pandas
data = pd.read_csv(input_data, sep=',', encoding="ISO-8859-1")
# drop rows with all the 'na' values
data.dropna(axis='index', inplace=True)

# define stopwords so that they can be removed from our input text
stopwords = ["a", "****abstract", ":" "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
             "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
             "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it",
             "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
             "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
             "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
             "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
             "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll",
             "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
             "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
             "yours", "yourself", "yourselves"]

# use concatenated column from the input file as X
X = data["Concatenated"].values.astype(np.str)
# remove stopwords from X using the above defined object
X = [" ".join([w for w in line.lower().split() if w not in stopwords]) for line in X]
# to consider last 11 columns as labels for our input X data
y = data.iloc[:, -11:].values.astype(np.int32)
# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# build a tokenizer using X_train and predefined parameters
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, lower="true")
tokenizer.fit_on_texts(X_train)
# object which gives all the word list from our train data with index
word_index = tokenizer.word_index
# tokenize and pad train data
sequences = tokenizer.texts_to_sequences(X_train)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
# tokenize and pad test data
testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
# get the total number of words which are tokenized
vocab_size = len(word_index) + 1


####################### WORD EMBEDDING LOADING BLOCK ##############################


# load the glove word embeddings/weights for the words present in our train corpus into a dictionary object
embeddings_index = {}
embedding_dim = 100
glove_dir = '/mnt/ai/shared/datasets/worldbank_demo/glove.6B.100d.txt'


with open(glove_dir) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector


####################### MODEL ARCHITECTURE BLOCK #############################


input = keras.layers.Input(shape=(max_length,))
x = keras.layers.Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], trainable=True)(input)
x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x)
x = keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
avg_pool = keras.layers.GlobalAveragePooling1D()(x)
max_pool = keras.layers.GlobalMaxPooling1D()(x)
x = keras.layers.concatenate([avg_pool, max_pool])
preds = keras.layers.Dense(11, activation="sigmoid")(x)
model = keras.Model(input, preds)
model.summary()
model.compile(loss=loss_func, optimizer=keras.optimizers.optimizer_func(lr=1e-3), metrics=['accuracy'])


####################### MODEL CALLBACK & TRAINING BLOCK ##################################


# Model Fitting # This will fit the data to the model # This takes the epochs, training data, validation data

# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = keras.callbacks.ModelCheckpoint(model_checkpoint_dir+"/sample2_cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [keras.callbacks.ReduceLROnPlateau(),
             keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
             keras.callbacks.TensorBoard(log_dir=tensorboard_dir),
             cp_callback]

model.fit(padded, y_train, validation_split=0.2, batch_size=batch_size,
          epochs=epoch, callbacks=callbacks, verbose=1)


####################### MODEL EXPORTING BLOCK #####################################


model.save(export_path_base+'/sample2_save.h5')

# Explicitly define the stop of the Learning phase by disabling the flag
K.set_learning_phase(0)  # Mandatory
# Define the exporter function and the path to export the servable
sess = K.get_session()
# Depends on the number of input variables and output variables
x = tf.placeholder(tf.float32, shape=(None))
# Depends on the number of input variables and output variables
pred = tf.placeholder(tf.float32, shape=(None))
# Example version, Change it accordingly.
model_version = "00000002"
# model_saved is a sample name for saving the model # this is editable accordingly
path = os.path.join(export_path_base, str(model_version))
builder = tf.saved_model.builder.SavedModelBuilder(path)
with tf.Session() as tf_sess:
    tf_sess.run(tf.global_variables_initializer())
    print(tf.saved_model.tag_constants.SERVING)

    # Set the logs writer to the tensorflow logs folder
    summary_writer = tf.summary.FileWriter(summary_log_dir, graph_def=sess.graph_def)

    builder.add_meta_graph_and_variables(
        tf_sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": tf.saved_model.signature_def_utils.predict_signature_def(
                # signature alias for the model input
                inputs={"x": model.input},
                outputs={"pred": model.output})
        })
builder.save()
print("Model Build is ready. Please use the .py file in Training phase "
      "for 'Training & exporting the model as Pb2 form'!!")


####################### MODEL TESTING BLOCK #####################################


# test the model on test data to find out accuracy
result = (model.predict(testing_padded) > 0.5) == np.array(y_test)
print("Total Accuracy")
print(np.array([all(r) for r in result]).mean())

