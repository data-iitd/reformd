#!/usr/bin/env python
import os, sys, pickle
import utils
import reformd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def_opts = reformd.def_opts
scale = 0.1

source = "data/" + sys.argv[1] + "/"
event_train_file = source+"Train_Cat"
event_test_file = source+"Test_Cat"
time_train_file = source+"Train_Time"
time_test_file = source+"Test_Time"
dist_train_file = source+"Train_Dist"
dist_test_file = source+"Test_Dist"

data_src = utils.read_data(
    event_train_file=event_train_file,
    event_test_file=event_test_file,
    time_train_file=time_train_file,
    time_test_file=time_test_file,
    dist_train_file=dist_train_file,
    dist_test_file=dist_test_file
)

data_src['train_time_out_seq'] /= scale
data_src['train_time_in_seq'] /= scale
data_src['test_time_out_seq'] /= scale
data_src['test_time_in_seq'] /= scale

target = "data/" + sys.argv[2] + "/"
event_train_file = target+"Train_Cat"
event_test_file = target+"Test_Cat"
time_train_file = target+"Train_Time"
time_test_file = target+"Test_Time"
dist_train_file = target+"Train_Dist"
dist_test_file = target+"Test_Dist"

data_tgt = utils.read_data(
    event_train_file=event_train_file,
    event_test_file=event_test_file,
    time_train_file=time_train_file,
    time_test_file=time_test_file,
    dist_train_file=dist_train_file,
    dist_test_file=dist_test_file
)

data_tgt['train_time_out_seq'] /= scale
data_tgt['train_time_in_seq'] /= scale
data_tgt['test_time_out_seq'] /= scale
data_tgt['test_time_in_seq'] /= scale

tf.reset_default_graph()
sess = tf.Session()

rfrmd_mdl = reformd.REFORMD(
    sess=sess,
    num_categories=500,
    batch_size=512,
    bptt=5,
    learning_rate=0.01,
    cpu_only=False,
    _opts=reformd.def_opts
)

rfrmd_mdl.initialize(finalize=False)
rfrmd_mdl.train(training_data=data_src, epochs=10)
rfrmd_mdl.train(training_data=data_tgt, epochs=20)

print('Results on Test data:')
test_time_preds, test_event_preds = rfrmd_mdl.predict_test(data=data_tgt)
rfrmd_mdl.eval(test_time_preds, data_tgt['test_time_out_seq'], test_event_preds, data_tgt['test_event_out_seq'])