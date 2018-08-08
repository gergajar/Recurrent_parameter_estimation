import numpy as np
import tensorflow as tf
from layers import *
import pickle
import os
import time
from network_input import SequenceDataset
import time


class RecurrentModel(object):

    def __init__(self, **kwargs):
        self.max_length = kwargs["max_length"]
        self.batch_size = kwargs["batch_size"]
        self.logdir = kwargs["logdir"]
        self.learning_rate = kwargs["learning_rate"]
        self.n_outputs = kwargs["n_outputs"]
        self.tensor_length = kwargs["tensor_length"]
        self.output_feedback = kwargs["output_feedback"]
        self.input_data_path = kwargs["input_data_path"]
        self.sequence_data_path = kwargs["sequence_data_path"]

        tf.reset_default_graph()
        self.sess = tf.Session()
        self.dataset = SequenceDataset(data_path=self.sequence_data_path,
                                       batch_size=self.batch_size,
                                       n_inputs=self.tensor_length,
                                       input_data_path=self.input_data_path,
                                       rebuild_data=False)
        self.iterator = {}
        self.init_iterator = {}
        # Create a coordinator and run all QueueRunner objects
        self.create_variables()
        self.build_graph()

        self.training_results = {}
        self.training_results["train"] = {"accuracy": [], "loss": [], "iteration": []}
        self.training_results["validation"] = {"accuracy": [], "loss": [], "iteration": []}

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_all_params = self.optimizer.minimize(self.loss)

        self.summ = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.aux_saver = tf.train.Saver()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        self.writer1 = tf.summary.FileWriter(self.logdir+"/train")
        self.writer2 = tf.summary.FileWriter(self.logdir+"/val")
        self.writer1.add_graph(self.sess.graph)

    def create_variables(self):

        print("----- creating variables -----")

        self.batch_norm_layer_0 = BatchNormalizationLayer(output_channels=self.tensor_length*5,
                                                          layer_name='batch_norm_layer_0')

        self.lstm_layer_1 = LSTMLayer(hidden_units=32, layer_name="lstm_layer_1", activation="relu")

        self.lstm_layer_2 = LSTMLayer(hidden_units=32, layer_name="lstm_layer_2", activation="relu")

        self.fc_layer_3 = FullyConnectedLayer([32, self.n_outputs],
                                               'linear',
                                               'fc_layer_3')

    def build_graph(self):

        print("---- Using tf records ----")

        self.input_sequence = tf.placeholder(tf.float32,
                                             shape=[self.batch_size, self.max_length, self.tensor_length],
                                             name="input_sequence")

        self.error_sequence = tf.placeholder(tf.float32,
                                             shape=[self.batch_size, self.max_length, self.tensor_length],
                                             name="error_sequence")

        self.time = tf.placeholder(tf.float32,
                                   shape=[self.batch_size, self.max_length, self.tensor_length],
                                   name="time")

        self.target = tf.placeholder(tf.float32,
                                     shape=[self.batch_size],
                                     name="target")


        self.seq_length = tf.placeholder(tf.int64,
                                        shape=[self.batch_size],
                                        name="seq_length")

        self.train_phase = tf.placeholder(tf.bool, name="train_phase")

        tf.add_to_collection(name="placeholders", value=self.input_sequence)
        tf.add_to_collection(name="placeholders", value=self.error_sequence)
        tf.add_to_collection(name="placeholders", value=self.time)
        tf.add_to_collection(name="placeholders", value=self.target)
        tf.add_to_collection(name="placeholders", value=self.seq_length)
        tf.add_to_collection(name="placeholders", value=self.train_phase)

        print("----- building grpah -----")

        init_state1 = tf.zeros([self.batch_size, self.lstm_layer_1.lstm.state_size], name="init_state")
        init_state2 = tf.zeros([self.batch_size, self.lstm_layer_2.lstm.state_size], name="init_state")
        init_output = tf.zeros([self.batch_size, self.n_outputs], name="init_output")

        self.stacked_features = tf.concat([self.input_sequence, self.error_sequence], axis=2)
        input_sequence = tf.unstack(self.stacked_features, axis=1)
        input_time = tf.unstack(self.time, axis=1)

        variable_size_mask = tf.sequence_mask(lengths=self.seq_length, maxlen=self.max_length, dtype=tf.float32)

        estimation = []
        loss = []

        self.loss = 0

        for i in range(self.max_length):
            with tf.variable_scope("time_"+str(i)):
                sequence_i = tf.identity(input_sequence[i], name="seq_"+str(i))
                time_i = tf.identity(input_time[i], name="t_"+str(i))
                sequence_i = tf.placeholder_with_default(sequence_i,
                                                      shape=[self.batch_size,
                                                      self.tensor_length*2],
                                                      name="image_"+str(i)+"_ph")
                time_i = tf.placeholder_with_default(time_i,
                                                    shape=[self.batch_size,
                                                           self.tensor_length],
                                                    name="day_"+str(i)+"_ph")
                tf.add_to_collection("input_per_time", sequence_i)
                tf.add_to_collection("input_per_time", time_i)

                if i == 0:
                    state1 = tf.identity(init_state1, name="state1_"+str(i))
                    state2 = tf.identity(init_state2, name="state2_"+str(i))
                    pivot_time = time_i[:, 0]

                    if self.output_feedback:
                        feedback_out = init_output

                state1 = tf.placeholder_with_default(state1,
                                                     shape=[self.batch_size,
                                                            self.lstm_layer_1.lstm.state_size],
                                                     name="state_" + str(i) + "_ph")
                state2 = tf.placeholder_with_default(state2,
                                                     shape=[self.batch_size,
                                                            self.lstm_layer_2.lstm.state_size],
                                                     name="state_" + str(i) + "_ph")

                if self.output_feedback:
                    complete_input = tf.concat([sequence_i, time_i - tf.expand_dims(pivot_time, 1), feedback_out], axis=1)
                else:
                    complete_input = tf.concat([sequence_i, time_i - tf.expand_dims(pivot_time, 1)], axis=1)

                # Forward step

                output1 = self.batch_norm_layer_0(input_tensor=complete_input,
                                                  phase_train=self.train_phase,
                                                  over_dim=[0, ])

                output2, state1 = self.lstm_layer_1(input_tensor=output1, state=state1)

                output3, state2 = self.lstm_layer_2(input_tensor=output2, state=state2)

                self.model_output = self.fc_layer_3(input_tensor=output3)

                state1 = tf.identity(state1, name="output_state1")
                state2 = tf.identity(state2, name="output_state2")

                with tf.name_scope("model_output"):

                    self.model_output = tf.identity(self.model_output, name="model_output_"+str(i))
                    estimation.append(self.model_output)

                    tf.add_to_collection(name="metrics", value=self.model_output)

                    loss_per_example = tf.metrics.mean_squared_error(labels=self.target,
                                                                     predictions=self.model_output,
                                                                     name='loss_per_example')

                    masked_loss = tf.multiply(variable_size_mask[:, i], loss_per_example)
                    self.loss_value = tf.reduce_mean(masked_loss, name="average_loss_"+str(i))
                    loss.append(self.loss_value)
                    self.loss += self.loss_value

        self.all_estimations = tf.stack(estimation, name="list_estimations")
        self.all_loss = tf.stack(loss, name="list_loss")

        tf.add_to_collection(name="metrics", value=self.all_estimations)
        tf.add_to_collection(name="metrics", value=self.all_loss)

        with tf.name_scope("MSE"):
            self.loss_summ = tf.summary.scalar("loss", self.loss)

        tf.add_to_collection(name="metrics", value=self.loss)

    def train_iterations(self, n_iterations):

        train_step = self.train_all_params
        self.best_loss_value = 1000
        start = time.time()

        for iteration in range(n_iterations):

            batch = self.dataset.next_batch(subset_name="training")

            _ = self.sess.run(train_step,
                              feed_dict={self.train_phase: True,
                                         self.input_sequence: batch["sequences"],
                                         self.error_sequence: batch["noise"],
                                         self.time: batch["time"],
                                         self.target: batch["params"],
                                         self.seq_length: batch["lengths"]})


            if iteration % 100 == 0:
                train_loss, s_t = self.sess.run([self.loss, self.summ],
                                                feed_dict={self.train_phase: True,
                                                self.input_sequence: batch["sequences"],
                                                self.error_sequence: batch["noise"],
                                                self.time: batch["time"],
                                                self.target: batch["params"],
                                                self.seq_length: batch["lengths"]})

                val_loss, s_v = self.sess.run([self.loss, self.summ],
                                               feed_dict={self.train_phase: True,
                                               self.input_sequence: batch["sequences"],
                                               self.error_sequence: batch["noise"],
                                               self.time: batch["time"],
                                               self.target: batch["params"],
                                               self.seq_length: batch["lengths"]})

                self.writer1.add_summary(s_t, iteration)
                self.writer2.add_summary(s_v, iteration)
                print("------------------")
                print("iteration = "+str(iteration),)
                print("training loss = " + str(train_loss))
                print("validation loss =" + str(val_loss))
                self.saver.save(self.sess, os.path.join(self.logdir, "model.ckpt"), iteration)
                # start = time.time()

        self.saver.save(self.sess, os.path.join(self.logdir, "final_model"))
        self.saver.export_meta_graph(os.path.join(self.logdir, "final_model.meta"))
        self.sess.close()


if __name__ == "__main__":

    max_length = 50
    batch_size = 128
    logdir = "./network_training/"
    experiment_name = "gaussian_noise/"
    logdir += experiment_name
    learning_rate = 0.001
    n_outputs = 2
    tensor_length = 1
    output_feedback = True
    data_path = "./data/sinusoidal_gaussian.pkl"
    input_data_path = "./data/network_input.pkl"

    model = RecurrentModel(max_length=max_length,
                           batch_size=batch_size,
                           logdir=logdir,
                           learning_rate=learning_rate,
                           n_outputs=n_outputs,
                           tensor_length=tensor_length,
                           output_feedback=output_feedback,
                           input_data_path=input_data_path,
                           sequence_data_path=data_path)

    start = time.time()
    model.train_iterations(2001)
    end = time.time()
    print("total time: " + str(end - start))
    # print(train_accuracy)