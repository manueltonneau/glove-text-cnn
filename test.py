import os
import time
import datetime

import tensorflow as tf
import numpy as np
import data_utils as utils

from tensorflow.contrib import learn
from text_cnn import TextCNN
from data_utils import IMDBDataset

import argparse
import pandas as pd
import pickle

def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()

    # necessary
    parser.add_argument("--checkpoint_dir", type=str, help="Path to the models", default="/home/manuto/Documents/world_bank/bert_twitter_labor/code/glove-text-cnn/runs/default_run_name/checkpoints")
    parser.add_argument("--eval_data_path", type=str, help="Path to the evaluation data. Must be in csv format.", default="/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced")
    parser.add_argument("--vocab_path", type=str, help="Path pickle file.", default="/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_embeddings/vocab.pckl")
    parser.add_argument("--label", default='is_unemployed', type=str)

    args = parser.parse_args()
    return args

def prepare_filepath_for_storing_pred(eval_data_path: str) -> str:
    path_to_store_pred = os.path.join(os.path.dirname(eval_data_path), 'glove_predictions')
    if not os.path.exists(path_to_store_pred):
        os.makedirs(path_to_store_pred)
    return path_to_store_pred

def tokenizer(text):
    return [wdict.get(w.lower(), 0) for w in text.split(' ')]

def pad_dataset(dataset, maxlen):
    return np.array(
        [np.pad(r, (0, maxlen - len(r)), mode='constant') if len(r) < maxlen else np.array(r[:maxlen])
         for r in dataset])

def create_label(label):
    if label == 1:
        return [0, 1]
    elif label == 0:
        return [1, 0]

args = get_args_from_command_line()

print ("Intialising test parameters ...")

batch_size = 64
# Checkpoint directory from training run
checkpoint_dir = args.checkpoint_dir
# Evaluate on all training data
eval_train = False

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

print ("Loading test data ...")
eval_df = pd.read_csv(args.eval_data_path, lineterminator='\n')


with open(args.vocab_path, 'rb') as dfile:
    wdict = pickle.load(dfile)

eval_df = eval_df[eval_df['text'].apply(lambda x: isinstance(x, str))]
eval_df['text_tokenized'] = eval_df['text'].apply(tokenizer)

x_test = pad_dataset(eval_df.text_tokenized.values.tolist(), 128)

#y_test = np.array((eval_df['class'].apply(create_label)).values.tolist())

# Evaluation
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        predictions_proba = graph.get_operation_by_name("output/predictions_proba").outputs[0]
        #predictions_proba = predictions_proba[:, 1]
        # Generate batches for one epoch
        batches = utils.batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_predictions_proba = []
        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            batch_predictions_proba = sess.run(predictions_proba, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_predictions_proba = np.concatenate([all_predictions_proba, batch_predictions_proba[:, 1]])



# Print accuracy if y_test is defined
if all_predictions is not None:
    print("Predictions done")
    #y_test = [col[1] for col in y_test]
    #correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(all_predictions_proba)))
    eval_df['glove_cnn_class_pred'] = all_predictions_proba
    #print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

output_dir = prepare_filepath_for_storing_pred(args.eval_data_path)
output_path = os.path.join(output_dir, os.path.split(args.eval_data_path)[1])
eval_df.to_csv(output_path, index=False)
print("Predictions saved at:", output_path)