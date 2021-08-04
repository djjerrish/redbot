import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.contrib import rnn

from tensorflow.python.util.nest import flatten

import numpy as np

class PartitionedMultiRNNCell(rnn_cell.RNNCell):
    def __init__(self, cell_fn, partition_size=128, partitions=1, layers=2):
        super(PartitionedMultiRNNCell, self).__init__()

        self._cells = []
        for i in range(layers):
            self._cells.append([cell_fn(partition_size) for _ in range(partitions)])
        self._partitions = partitions

    @property
    def state_size(self):

        return tuple(((layer[0].state_size,) * len(layer)) for layer in self._cells)

    @property
    def output_size(self):

        return self._cells[-1][0].output_size * len(self._cells[-1])

    def zero_state(self, batch_size, dtype):

        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return tuple(tuple(cell.zero_state(batch_size, dtype) for cell in layer) for layer in self._cells)

    def call(self, inputs, state):
        layer_input = inputs
        new_states = []
        for l, layer in enumerate(self._cells):
            if l > 0:
                offset_width = layer[0].output_size // 2
                layer_input = tf.concat((layer_input[:, -offset_width:], layer_input[:, :-offset_width]),
                    axis=1, name='concat_offset_%d' % l)

            p_inputs = tf.split(layer_input, len(layer), axis=1, name='split_%d' % l)
            p_outputs = []
            p_states = []
            for p, p_inp in enumerate(p_inputs):
                with vs.variable_scope("cell_%d_%d" % (l, p)):
                    p_state = state[l][p]
                    cell = layer[p]
                    p_out, new_p_state = cell(p_inp, p_state)
                    p_outputs.append(p_out)
                    p_states.append(new_p_state)
            new_states.append(tuple(p_states))
            layer_input = tf.concat(p_outputs, axis=1, name='concat_%d' % l)
        new_states = tuple(new_states)
        return layer_input, new_states

def _rnn_state_placeholders(state):
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder(c.dtype, c.shape, c.op.name)
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return h
    else:
        structure = [_rnn_state_placeholders(x) for x in state]
        return tuple(structure)

class Model():
    def __init__(self, args, infer=False): 
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.lr = tf.Variable(args.learning_rate, name="learning_rate", trainable=False)
        self.global_epoch_fraction = tf.Variable(0.0, name="global_epoch_fraction", trainable=False)
        self.global_seconds_elapsed = tf.Variable(0.0, name="global_seconds_elapsed", trainable=False)


        cell = PartitionedMultiRNNCell(cell_fn, partitions=args.num_blocks,
            partition_size=args.block_size, layers=args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        self.zero_state = cell.zero_state(args.batch_size, tf.float32)

        self.initial_state = _rnn_state_placeholders(self.zero_state)
        self._flattened_initial_state = flatten(self.initial_state)

        layer_size = args.block_size * args.num_blocks

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [layer_size, args.vocab_size])

            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])

            embedding = tf.get_variable("embedding", [args.vocab_size, layer_size])

            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs,
                initial_state=self.initial_state, scope='rnnlm')

        output = tf.reshape(outputs, [-1, layer_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        if infer:
            self.probs = tf.nn.softmax(self.logits)
        else:
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

            loss = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]), logits=self.logits)

            self.cost = tf.reduce_mean(loss)
            
            tf.summary.scalar("cost", self.cost)

            tvars = tf.trainable_variables() 
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                     args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr) 
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.summary_op = tf.summary.merge_all()

    def add_state_to_feed_dict(self, feed_dict, state):
        for i, tensor in enumerate(flatten(state)):
            feed_dict[self._flattened_initial_state[i]] = tensor

    def save_variables_list(self):
        save_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnnlm'))
        save_vars.update({self.lr, self.global_epoch_fraction, self.global_seconds_elapsed})
        return list(save_vars)

    def forward_model(self, sess, state, input_sample):
        shaped_input = np.array([[input_sample]], np.float32)
        inputs = {self.input_data: shaped_input}
        self.add_state_to_feed_dict(inputs, state)
        [probs, state] = sess.run([self.probs, self.final_state], feed_dict=inputs)
        return probs[0], state

    def trainable_parameter_count(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters
