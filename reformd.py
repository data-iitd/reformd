import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os, pdb, pickle
import decorated_options as Deco
from utils import MAE, ACC
from scipy.integrate import quad
import multiprocessing as MP
import logging
tf.get_logger().setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

__EMBED_SIZE = 16
__HIDDEN_LAYER_SIZE = 64 

def_opts = Deco.Options(
    momentum=0.9,
    decay_steps=100,
    decay_rate=0.001,
    l2_penalty=0.001,
    float_type=tf.float32,
    seed=1234,
    scope='REFORMD',
    device_gpu='/gpu:0',
    device_cpu='/cpu:0',
    embed_size=__EMBED_SIZE,
    
    Wem=lambda num_categories: np.random.RandomState(42).randn(num_categories, __EMBED_SIZE) * 0.01,
    Wt=np.random.RandomState(42).randn(1, __HIDDEN_LAYER_SIZE)* 0.1,
    Wd=np.random.RandomState(42).randn(1, __HIDDEN_LAYER_SIZE)* 0.1,
    Wh=np.random.RandomState(42).randn(__HIDDEN_LAYER_SIZE)* 0.1,
    bh=np.random.RandomState(42).randn(1, __HIDDEN_LAYER_SIZE)* 0.1,
    wt=1.0,
    wd=1.0,
    Wy=np.random.RandomState(42).randn(__EMBED_SIZE, __HIDDEN_LAYER_SIZE)* 0.1,
    Vy=lambda num_categories: np.random.RandomState(42).randn(__HIDDEN_LAYER_SIZE, num_categories)* 0.1,
    Vt=np.random.RandomState(42).randn(__HIDDEN_LAYER_SIZE, 1)* 0.1,
    Vd=np.random.RandomState(42).randn(__HIDDEN_LAYER_SIZE, 1)* 0.1,
    bt=np.log(1.0),
    bd=np.log(1.0),
    bk=lambda num_categories: np.random.RandomState(42).randn(1, num_categories)* 0.1
)

def softplus(x):
    return np.log1p(np.exp(x))

def quad_func(t, c, w):
    return c * t * np.exp(-w * t + (c / w) * (np.exp(-w * t) - 1))

class REFORMD:
    @Deco.optioned()
    def __init__(self, sess, num_categories, batch_size,
                 learning_rate, momentum, l2_penalty, embed_size,
                 float_type, bptt, seed, scope, decay_steps, decay_rate,
                 device_gpu, device_cpu, cpu_only,
                 Wt, Wem, Wh, bh, wt, Wy, Vy, Vt, bk, bt, Wd, wd, Vd, bd):
        self.HIDDEN_LAYER_SIZE = Wh.shape[0]
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.MOMENTUM = momentum
        self.L2_PENALTY = l2_penalty
        self.EMBED_SIZE = embed_size
        self.BPTT = bptt

        self.NUM_CATEGORIES = num_categories
        self.FLOAT_TYPE = float_type

        self.DEVICE_CPU = device_cpu
        self.DEVICE_GPU = device_gpu

        self.sess = sess
        self.seed = seed
        self.last_epoch = 0

        self.rs = np.random.RandomState(seed + 42)

        with tf.variable_scope(scope):
            with tf.device(device_gpu if not cpu_only else device_cpu):
                # Make input variables
                self.events_in = tf.placeholder(tf.int32, [None, self.BPTT], name='events_in')
                self.times_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_in')
                self.dists_in = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='dists_in')

                self.events_out = tf.placeholder(tf.int32, [None, self.BPTT], name='events_out')
                self.times_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='times_out')
                self.dists_out = tf.placeholder(self.FLOAT_TYPE, [None, self.BPTT], name='dists_out')

                self.batch_num_events = tf.placeholder(self.FLOAT_TYPE, [], name='bptt_events')

                self.inf_batch_size = tf.shape(self.events_in)[0]

                # Make variables
                with tf.variable_scope('hidden_state'):
                    self.Wt = tf.get_variable(name='Wt', shape=(1, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wt))
                    self.Wd = tf.get_variable(name='Wd', shape=(1, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wd))
                    self.Wem = tf.get_variable(name='Wem', shape=(self.NUM_CATEGORIES, self.EMBED_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wem(self.NUM_CATEGORIES)))
                    self.Wh = tf.get_variable(name='Wh', shape=(self.HIDDEN_LAYER_SIZE, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE,initializer=tf.constant_initializer(Wh))
                    self.bh = tf.get_variable(name='bh', shape=(1, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bh))

                with tf.variable_scope('output'):
                    self.wt = tf.get_variable(name='wt', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(wt))
                    self.wd = tf.get_variable(name='wd', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(wd))
                    self.Wy = tf.get_variable(name='Wy', shape=(self.EMBED_SIZE, self.HIDDEN_LAYER_SIZE), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Wy))

                    # The first column of Vy is merely a placeholder (will not be trained).
                    self.Vy = tf.get_variable(name='Vy', shape=(self.HIDDEN_LAYER_SIZE, self.NUM_CATEGORIES), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Vy(self.NUM_CATEGORIES)))
                    self.Vt = tf.get_variable(name='Vt', shape=(self.HIDDEN_LAYER_SIZE, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Vt))
                    self.Vd = tf.get_variable(name='Vd', shape=(self.HIDDEN_LAYER_SIZE, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(Vd))
                    self.bt = tf.get_variable(name='bt', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bt))
                    self.bd = tf.get_variable(name='bd', shape=(1, 1), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bd))
                    self.bk = tf.get_variable(name='bk', shape=(1, self.NUM_CATEGORIES), dtype=self.FLOAT_TYPE, initializer=tf.constant_initializer(bk(num_categories)))

                self.all_vars = [self.Wt, self.Wd, self.Wem, self.Wh, self.bh, self.wt, self.wd, self.Wy, self.Vy, self.Vt, self.Vd, self.bt, self.bd, self.bk]

                self.initial_state = state = tf.zeros([self.inf_batch_size, self.HIDDEN_LAYER_SIZE], dtype=self.FLOAT_TYPE, name='initial_state')
                self.initial_time = last_time = tf.zeros((self.inf_batch_size,), dtype=self.FLOAT_TYPE, name='initial_time')
                self.initial_dist = last_dist = tf.zeros((self.inf_batch_size,), dtype=self.FLOAT_TYPE, name='initial_dist')

                self.loss = 0.0
                ones_2d = tf.ones((self.inf_batch_size, 1), dtype=self.FLOAT_TYPE)

                self.hidden_states = []
                self.event_preds = []

                self.time_LLs = []
                self.dist_LLs = []
                self.mark_LLs = []
                self.log_lambdas = []
                self.log_lambdas_d = []
                self.times = []
                self.dists = []

                with tf.name_scope('BPTT'):
                    for i in range(self.BPTT):
                        events_embedded = tf.nn.embedding_lookup(self.Wem, tf.mod(self.events_in[:, i] - 1, self.NUM_CATEGORIES))
                        time = self.times_in[:, i]
                        time_next = self.times_out[:, i]

                        dist = self.dists_in[:, i]
                        dist_next = self.dists_out[:, i]

                        delta_t_prev = tf.expand_dims(time - last_time, axis=-1)
                        delta_t_next = tf.expand_dims(time_next - time, axis=-1)

                        delta_d_prev = tf.expand_dims(dist - last_time, axis=-1)
                        delta_d_next = tf.expand_dims(time_next - time, axis=-1)

                        last_time = time
                        last_dist = dist
                        time_2d = tf.expand_dims(time, axis=-1)
                        dist_2d = tf.expand_dims(dist, axis=-1)
                        
                        type_delta_t = True
                        type_delta_d = True

                        with tf.name_scope('state_recursion'):
                            new_state = tf.tanh( tf.matmul(state, self.Wh) + tf.matmul(events_embedded, self.Wy) +
                                (tf.matmul(delta_t_prev, self.Wt) if type_delta_t else tf.matmul(time_2d, self.Wt)) +
                                (tf.matmul(delta_d_prev, self.Wd) if type_delta_d else tf.matmul(dist_2d, self.Wd)) +
                                tf.matmul(ones_2d, self.bh),
                                name='h_t')
                            state = tf.where(self.events_in[:, i] > 0, new_state, state)

                        with tf.name_scope('loss_calc'):
                            base_intensity = tf.matmul(ones_2d, self.bt)
                            base_intensity_d = tf.matmul(ones_2d, self.bd)
                            wt_soft_plus = tf.nn.softplus(self.wt)
                            wd_soft_plus = tf.nn.softplus(self.wd)
                            log_lambda_ = (tf.matmul(state, self.Vt) + (-delta_t_next * wt_soft_plus) + base_intensity)
                            log_lambda_d = (tf.matmul(state, self.Vd) + (-delta_d_next * wd_soft_plus) + base_intensity_d)

                            lambda_ = tf.exp(tf.minimum(50.0, log_lambda_), name='lambda_')
                            lambda_d = tf.exp(tf.minimum(50.0, log_lambda_d), name='lambda_d')
                            log_f_star = (log_lambda_ - (1.0 / wt_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(state, self.Vt) + base_intensity)) + (1.0 / wt_soft_plus) * lambda_)
                            log_f_star_d = (log_lambda_d - (1.0 / wd_soft_plus) * tf.exp(tf.minimum(50.0, tf.matmul(state, self.Vd) + base_intensity_d)) + (1.0 / wd_soft_plus) * lambda_d)
                            
                            events_pred = tf.nn.softmax(tf.minimum(50.0, tf.matmul(state, self.Vy) + ones_2d * self.bk), name='Pr_events' )
                            events_pred = tf.nn.dropout(events_pred, keep_prob=0.5)
                            time_LL = log_f_star
                            dist_LL = log_f_star_d
                            mark_LL = tf.expand_dims(
                                tf.log(tf.maximum(1e-6,tf.gather_nd(events_pred,tf.concat([
                                                tf.expand_dims(tf.range(self.inf_batch_size), -1),
                                                tf.expand_dims(tf.mod(self.events_out[:, i] - 1, self.NUM_CATEGORIES), -1)], axis=1, name='Pr_next_event')))), axis=-1, name='log_Pr_next_event')
                            step_LL = time_LL + mark_LL # + dist_LL 
                            num_events = tf.reduce_sum(tf.where(self.events_in[:, i] > 0, tf.ones(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE), tf.zeros(shape=(self.inf_batch_size,), dtype=self.FLOAT_TYPE)), name='num_events')
                            self.loss -= tf.reduce_sum(tf.where(self.events_in[:, i] > 0, tf.squeeze(step_LL) / self.batch_num_events, tf.zeros(shape=(self.inf_batch_size,))))

                        self.time_LLs.append(time_LL)
                        self.dist_LLs.append(dist_LL)
                        self.mark_LLs.append(mark_LL)
                        self.log_lambdas.append(log_lambda_)
                        self.log_lambdas_d.append(log_lambda_d)

                        self.hidden_states.append(state)
                        self.event_preds.append(events_pred)

                        self.times.append(time)
                        self.dists.append(dist)

                self.final_state = self.hidden_states[-1]

                with tf.device(device_cpu):
                    self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.learning_rate = tf.train.inverse_time_decay(self.LEARNING_RATE, global_step=self.global_step, decay_steps=decay_steps, decay_rate=decay_rate)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.MOMENTUM)
                self.gvs = self.optimizer.compute_gradients(self.loss)
                grads, vars_ = list(zip(*self.gvs))
                self.norm_grads, self.global_norm = tf.clip_by_global_norm(grads, 10.0)
                capped_gvs = list(zip(self.norm_grads, vars_))
                self.update = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
                self.tf_init = tf.global_variables_initializer()
    
    def initialize(self, finalize=False):
        self.sess.run(self.tf_init)
        if finalize:
            self.sess.graph.finalize()

    def train(self, training_data, epochs):
        num_epochs = epochs
        train_event_in_seq = training_data['train_event_in_seq']
        train_time_in_seq = training_data['train_time_in_seq']
        train_dist_in_seq = training_data['train_dist_in_seq']
        train_event_out_seq = training_data['train_event_out_seq']
        train_time_out_seq = training_data['train_time_out_seq']
        train_dist_out_seq = training_data['train_dist_out_seq']

        idxes = list(range(len(train_event_in_seq)))
        n_batches = len(idxes) // self.BATCH_SIZE

        for epoch in range(self.last_epoch, self.last_epoch + num_epochs):
            self.rs.shuffle(idxes)
            total_loss = 0.0

            for batch_idx in range(n_batches):
                batch_idxes = idxes[batch_idx * self.BATCH_SIZE:(batch_idx + 1) * self.BATCH_SIZE]
                batch_event_train_in = train_event_in_seq[batch_idxes, :]
                batch_event_train_out = train_event_out_seq[batch_idxes, :]
                batch_time_train_in = train_time_in_seq[batch_idxes, :]
                batch_time_train_out = train_time_out_seq[batch_idxes, :]
                batch_dist_train_in = train_dist_in_seq[batch_idxes, :]
                batch_dist_train_out = train_dist_out_seq[batch_idxes, :]

                cur_state = np.zeros((self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE))
                batch_loss = 0.0

                batch_num_events = np.sum(batch_event_train_in > 0)
                for bptt_idx in range(0, len(batch_event_train_in[0]) - self.BPTT, self.BPTT):
                    bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
                    bptt_event_in = batch_event_train_in[:, bptt_range]
                    bptt_event_out = batch_event_train_out[:, bptt_range]
                    bptt_time_in = batch_time_train_in[:, bptt_range]
                    bptt_time_out = batch_time_train_out[:, bptt_range]
                    bptt_dist_in = batch_dist_train_in[:, bptt_range]
                    bptt_dist_out = batch_dist_train_out[:, bptt_range]

                    if np.all(bptt_event_in[:, 0] == 0):
                        break

                    if bptt_idx > 0:
                        initial_time = batch_time_train_in[:, bptt_idx - 1]
                        initial_dist = batch_dist_train_in[:, bptt_idx - 1]
                    else:
                        initial_time = np.zeros(batch_time_train_in.shape[0])
                        initial_dist = np.zeros(batch_dist_train_in.shape[0])

                    feed_dict = {
                        self.initial_state: cur_state,
                        self.initial_time: initial_time,
                        self.initial_dist: initial_dist,
                        self.events_in: bptt_event_in,
                        self.events_out: bptt_event_out,
                        self.times_in: bptt_time_in,
                        self.times_out: bptt_time_out,
                        self.dists_in: bptt_dist_in,
                        self.dists_out: bptt_dist_out,
                        self.batch_num_events: batch_num_events
                    }

                    _, cur_state, loss_ = self.sess.run([self.update, self.final_state, self.loss], feed_dict=feed_dict)
                    batch_loss += loss_

                total_loss += batch_loss
            print('Loss after epoch {:.4f}'.format(total_loss / n_batches))
        self.last_epoch += num_epochs

    def predict(self, event_in_seq, time_in_seq, dist_in_seq, event_out_seq, time_out_seq, dist_out_seq, single_threaded=False):
        all_hidden_states = []
        all_event_preds = []
        cur_state = np.zeros((len(event_in_seq), self.HIDDEN_LAYER_SIZE))

        for bptt_idx in range(0, len(event_in_seq[0]) - self.BPTT, self.BPTT):
            bptt_range = range(bptt_idx, (bptt_idx + self.BPTT))
            bptt_event_in = event_in_seq[:, bptt_range]
            bptt_time_in = time_in_seq[:, bptt_range]
            bptt_dist_in = dist_in_seq[:, bptt_range]

            if bptt_idx > 0:
                initial_time = event_in_seq[:, bptt_idx - 1]
                initial_dist = event_in_seq[:, bptt_idx - 1]
            else:
                initial_time = np.zeros(bptt_time_in.shape[0])
                initial_dist = np.zeros(bptt_dist_in.shape[0])

            feed_dict = {self.initial_state: cur_state, self.initial_time: initial_time, self.initial_dist: initial_dist, self.events_in: bptt_event_in, self.times_in: bptt_time_in, self.dists_in: bptt_dist_in}
            bptt_hidden_states, bptt_events_pred, cur_state = self.sess.run([self.hidden_states, self.event_preds, self.final_state], feed_dict=feed_dict)
            all_hidden_states.extend(bptt_hidden_states)
            all_event_preds.extend(bptt_events_pred)

        [Vt, Vd, bt, bd, wt, wd]  = self.sess.run([self.Vt, self.Vd, self.bt, self.bd, self.wt, self.wd])
        [Wem, Vy, bk, Wh, Wy, Wt, Wd, bh]  = self.sess.run([self.Wem, self.Vy, self.bk, self.Wh, self.Wy, self.Wt, self.Wd, self.bh])
        wt = softplus(wt)
        wd = softplus(wd)

        global _quad_worker
        def _quad_worker(params):
            idx, h_i = params
            preds_i = []
            C = np.exp(np.dot(h_i, Vt) + bt).reshape(-1)

            for c_, t_last in zip(C, time_in_seq[:,idx]):
                args = (c_, wt)
                val, _err = quad(quad_func, 0, np.inf, args=args)
                preds_i.append(t_last + val)

            return preds_i

        if single_threaded:
            all_time_preds = [_quad_worker((idx, x)) for idx, x in enumerate(all_hidden_states)]
        else:
            with MP.Pool() as pool:
                all_time_preds = pool.map(_quad_worker, enumerate(all_hidden_states))
        
        return np.asarray(all_time_preds).T, np.asarray(all_event_preds).swapaxes(0, 1)

    def eval(self, time_preds, time_true, event_preds, event_true):
        mae, _ = MAE(time_preds, time_true, event_true)
        print('** MAE = {:.4f}; ACC = {:.4f}'.format(
            mae, ACC(event_preds, event_true)))

    def predict_test(self, data, single_threaded=False):
        return self.predict(event_in_seq=data['test_event_in_seq'],
                            time_in_seq=data['test_time_in_seq'],
                            dist_in_seq=data['test_dist_in_seq'],
                            event_out_seq=data['test_event_out_seq'],
                            time_out_seq=data['test_time_out_seq'],
                            dist_out_seq=data['test_dist_out_seq'],
                            single_threaded=single_threaded)
