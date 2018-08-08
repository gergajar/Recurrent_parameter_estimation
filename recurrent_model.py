import numpy as np
import tensorflow as tf
from layers_v2 import *
import pickle
import os
import time

class LightCurveClassifier(object):

    def __init__(self, **kwargs):
        self.sequence_length = kwargs["sequence_length"]
        self.batch_size = kwargs["batch_size"]
        self.logdir = kwargs["logdir"]
        self.learning_rate = kwargs["learning_rate"]
        self.n_classes = kwargs["n_outputs"]

        self.buffer_size = kwargs["buffer_size"]
        self.tensor_length = kwargs["tensor_length"]
        self.with_dropout = kwargs["with_dropout"]

        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 1})
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=config)
        self.dataset = {}
        self.iterator = {}
        self.init_iterator = {}
        # Create a coordinator and run all QueueRunner objects
        with tf.device(self.device):
            self.create_variables()
            self.build_graph()
            self.validation_graph()

            self.training_results = {}
            self.training_results["train"] = {"accuracy": [], "loss": [], "iteration": []}
            self.training_results["validation"] = {"accuracy": [], "loss": [], "iteration": []}

            with tf.name_scope("train"):
                #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)  # 5e-4
                self.optimizer = tf.train.AdamOptimizer()
                self.train_all_params = self.optimizer.minimize(self.loss)

            with tf.name_scope("accuracy"):
                self.correct_predictions = tf.equal(tf.argmax(self.model_output, 1),
                                                    tf.argmax(self.one_hot_target, 1), name="correct_pred")
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32), name='accuracy')
                self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)
                tf.add_to_collection(name="metrics", value=self.correct_predictions)
                tf.add_to_collection(name="metrics", value=self.accuracy)

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

        self.batch_norm_layer_0 = BatchNormalizationLayer(self.tensor_length*2,
                                                          'batch_norm_layer_0')

        self.fc_layer_1 = FullyConnectedLayer([self.tensor_length*2, 6*6*64],
                                              "relu",
                                              "fc_layer_1")

        self.fc_layer_2 = FullyConnectedLayer([6*6*64, 1024],
                                               'relu',
                                               'fc_layer_2')

        self.lstm_layer_3 = LSTMLayer(hidden_units=512, layer_name="lstm_layer_3")

        self.fc_layer_4 = FullyConnectedLayer([512, self.n_classes],
                                               'linear',
                                               'fc_layer_4')

    def build_graph(self):

        print("---- Using tf records ----")

        self.train_set = tf.placeholder(tf.bool, name="train_set")
        tf.add_to_collection(name="placeholders", value=self.train_set)

        train_counts, train_errors, train_labels, train_days, train_seq_l, train_var_flag = self.get_record_tensor("train")
        val_counts, val_errors, val_labels, val_days, val_seq_l, val_var_flag = self.get_record_tensor("val")
        counts, errors, labels, days, seq_l, var_flag = tf.cond(self.train_set,
                                                                lambda: [train_counts, train_errors, train_labels,
                                                                         train_days, train_seq_l, train_var_flag],
                                                                lambda: [val_counts, val_errors, val_labels,
                                                                         val_days, val_seq_l, val_var_flag])

        self.count_sequence = tf.placeholder_with_default(counts,
                                                          shape=[self.batch_size, self.sequence_length, self.tensor_length],
                                                          name="count_sequence")
        self.error_sequence = tf.placeholder_with_default(errors,
                                                          shape=[self.batch_size, self.sequence_length,
                                                                 self.tensor_length],
                                                          name="error_sequence")
        self.target = tf.placeholder_with_default(labels,
                                                  shape=[self.batch_size],
                                                  name="target")

        self.days= tf.placeholder_with_default(days,
                                               shape=[self.batch_size, self.sequence_length, self.tensor_length],
                                               name="days")

        self.seq_l= tf.placeholder_with_default(seq_l,
                                                shape=[self.batch_size],
                                                name="seq_l")

        self.var_flag = tf.placeholder_with_default(var_flag,
                                                    shape=[self.batch_size],
                                                    name="var_flag")

        tf.add_to_collection(name="placeholders", value=self.target)
        tf.add_to_collection(name="placeholders", value=self.count_sequence)
        tf.add_to_collection(name="placeholders", value=self.error_sequence)
        tf.add_to_collection(name="placeholders", value=self.days)
        tf.add_to_collection(name="placeholders", value=self.seq_l)
        tf.add_to_collection(name="placeholders", value=self.var_flag)
        # tf.summary.scalar("label", self.target)
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
        tf.add_to_collection(name="placeholders", value=self.keep_prob)

        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        tf.add_to_collection(name="placeholders", value=self.train_phase)

        print("----- building grpah -----")

        init_state = tf.zeros([self.batch_size, self.lstm_layer_3.lstm.state_size], name="init_state")
        init_prob = tf.zeros([self.batch_size, self.n_classes], name="init_prob")

        self.stacked_photometry = tf.concat([self.count_sequence, self.error_sequence], axis=2)
        input_sequence = tf.unstack(self.stacked_photometry, axis=1)
        input_days = tf.unstack(self.days, axis=1)

        variable_size_mask = tf.sequence_mask(lengths=self.seq_l, maxlen=self.sequence_length, dtype=tf.float32)

        probabilities = []
        accuracies = []
        correct_predictions = []
        flag_out_list = []

        self.loss = 0

        for i in range(self.sequence_length):
            with tf.variable_scope("time_"+str(i)):
                sequence_i = tf.identity(input_sequence[i], name="seq_"+str(i))
                day_i = tf.identity(input_days[i], name="day_"+str(i))
                sequence_i = tf.placeholder_with_default(sequence_i,
                                                      shape=[self.batch_size,
                                                      self.tensor_length*2],
                                                      name="image_"+str(i)+"_ph")
                day_i = tf.placeholder_with_default(day_i,
                                                    shape=[self.batch_size,
                                                           self.tensor_length],
                                                    name="day_"+str(i)+"_ph")
                tf.add_to_collection("input_per_time", sequence_i)
                tf.add_to_collection("input_per_time", day_i)
                if i == 0:
                    state=tf.identity(init_state, name="state_"+str(i))
                    pivot_day = day_i[:, 0]
                    if self.prob_feedback:
                        feedback_prob = init_prob
                state = tf.placeholder_with_default(state,
                                                    shape=[self.batch_size,
                                                           self.lstm_layer_3.lstm.state_size],
                                                    name="state_" + str(i) + "_ph")
                output1 = self.batch_norm_layer_0(input_tensor=sequence_i,
                                                  phase_train=self.train_phase,
                                                  over_dim=[0, ])
                output2 = self.fc_layer_1(output1)
                output3 = self.fc_layer_2(output2)
                if self.with_dropout:
                    output3 = tf.nn.dropout(output3, self.keep_prob)
                if self.prob_feedback:
                    concat_with_days = tf.concat([output3, day_i - tf.expand_dims(pivot_day, 1), feedback_prob],
                                                 axis=1,
                                                 name="adding_days")
                else:
                    concat_with_days = tf.concat([output3, day_i - tf.expand_dims(pivot_day, 1)], axis=1,
                                                 name="adding_days")
                output4, state = self.lstm_layer_3(input_tensor=concat_with_days,
                                                   state=state)
                state = tf.identity(state, name="output_state")
                self.model_output = self.fc_layer_4(input_tensor=output4)
                if self.prob_feedback:
                    print("----USING PROB FEEDBACK----")
                    feedback_prob = tf.nn.softmax(self.model_output, name="prob_feedback")

                with tf.name_scope("model_output"):

                    self.softmax_output = tf.nn.softmax(self.model_output, name="output_probabilities")
                    self.one_hot_target = tf.one_hot(indices=self.target, depth=self.n_classes, name="One_hot_target")
                    correct_predictions_i = tf.equal(tf.argmax(self.model_output, 1),
                                                     tf.argmax(self.one_hot_target, 1), name="output_correct_pred")
                    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='output_accuracy')

                    probabilities.append(self.softmax_output)
                    accuracies.append(accuracy)
                    correct_predictions.append(correct_predictions_i)

                    tf.add_to_collection(name="metrics", value=correct_predictions_i)
                    tf.add_to_collection(name="metrics", value=accuracy)
                    tf.add_to_collection(name="metrics", value=self.softmax_output)
                    tf.add_to_collection(name="metrics", value=state)

                    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
                                       logits=self.model_output,
                                       labels=self.one_hot_target,#self.target,
                                       name='loss_per_example')

                    masked_loss = tf.multiply(variable_size_mask[:, i], loss_per_example)
                    self.loss_value = tf.reduce_mean(masked_loss, name="average_loss")
                    self.loss += self.loss_value

        self.all_probabilities = tf.stack(probabilities, name="list_probabilities")
        self.all_accuracies = tf.stack(accuracies, name="list_accuracies")
        self.all_correct_pred = tf.stack(correct_predictions, name="list_correct_pred")
        if self.flag_loss_coef:
            all_flag_out = tf.stack(flag_out_list, name="list_flag_out")
            tf.add_to_collection(name="metrics", value=all_flag_out)

        tf.add_to_collection(name="metrics", value=self.all_probabilities)
        tf.add_to_collection(name="metrics", value=self.all_accuracies)
        tf.add_to_collection(name="metrics", value=self.all_correct_pred)

        with tf.name_scope("cross_entropy"):
            self.loss_summ = tf.summary.scalar("loss", self.loss)

        tf.add_to_collection(name="metrics", value=self.loss)

    def train_iterations(self, n_iterations):

        train_step = self.train_all_params
        self.best_val_accuracy = 0
        self.best_loss_value = 1000
        start = time.time()

        print("----- doing training with tf records -----")
        for iteration in range(n_iterations):
            if iteration == 0:
                print("----- runinning first sess -----")
                print(self.tf_record_path+"train" + "_seql_" + str(self.sequence_length)+".tfrecord")
            _ = self.sess.run(train_step, feed_dict={
                self.keep_prob: 0.5,
                self.train_phase: True,
                self.train_set: True})
            if iteration % 200 == 0:
                train_accuracy, train_loss, s = self.sess.run([self.accuracy, self.loss, self.summ], feed_dict={
                    self.keep_prob: 1.0,
                    self.train_phase: False,
                    self.train_set: True
                })
                self.writer1.add_summary(s, iteration)
                print("iteration = "+str(iteration))
                print("train accuracy = " + str(train_accuracy))
                print("train loss =" + str(train_loss))
                self.training_results["train"]["accuracy"].append(train_accuracy)
                self.training_results["train"]["loss"].append(train_loss)
                self.training_results["train"]["iteration"].append(iteration)
                self.saver.save(self.sess, os.path.join(self.logdir, "model.ckpt"), iteration)
                print("time_running: "+str((time.time()-start)/float(60))+"min")
                # start = time.time()

            if iteration % 500 == 0:
                self.run_validation(iteration=iteration)

        self.saver.save(self.sess, os.path.join(self.logdir, "final_model"))
        self.saver.export_meta_graph(os.path.join(self.logdir, "final_model.meta"))
        self.sess.close()
        pickle.dump(self.training_results,
                    open(self.logdir+"train_results.pkl", "wb"),
                    protocol=2)
        return self.best_loss_value, self.best_val_accuracy

    def validation_graph(self):
        with tf.variable_scope("validation_graph"):
            self.val_accuracy_placeholder = tf.placeholder_with_default(tf.zeros([]),
                                                                        shape=(),
                                                                        name="val_accuracy")
            self.loss_accuracy_placeholder = tf.placeholder_with_default(tf.zeros([]),
                                                                         shape=(),
                                                                         name="val_loss")

            acc_sum = tf.summary.scalar("accuracy", self.val_accuracy_placeholder)
            loss_sum = tf.summary.scalar("loss", self.loss_accuracy_placeholder)
            self.val_summ = tf.summary.merge([acc_sum,
                                              loss_sum])

    def run_validation(self, iteration, n_batches=50):
        correct_pred_list = []
        loss_list = []
        for i in range(n_batches):
            correct_pred, loss = self.sess.run([self.correct_predictions, self.loss], feed_dict={
                    self.keep_prob: 1.0,
                    self.train_phase: False,
                    self.train_set: False
                })
            correct_pred_list.append(correct_pred)
            #print(loss)
            loss_list.append(loss)
        correct_pred_list = np.concatenate(correct_pred_list)
        loss_list = np.array(loss_list)
        val_loss = np.mean(loss_list)
        val_accuracy = np.mean(correct_pred_list)
        if val_loss < self.best_loss_value:
            self.best_loss_value = val_loss
            self.aux_saver.save(self.sess, os.path.join(self.logdir+"/best_model/", "best_model"))
            self.aux_saver.export_meta_graph(os.path.join(self.logdir+"/best_model/", "best_model.meta"))
            print("updating model")
        val_acc_sum = self.sess.run(self.val_summ, feed_dict={
            self.val_accuracy_placeholder: val_accuracy.astype(np.float32),
            self.loss_accuracy_placeholder: val_loss.astype(np.float32)
        })
        self.writer2.add_summary(val_acc_sum, iteration)
        print("iteration = " + str(iteration))
        print("val accuracy = " + str(val_accuracy))
        print("val loss = " + str(val_loss))
        self.training_results["validation"]["accuracy"].append(val_accuracy)
        self.training_results["validation"]["loss"].append(val_loss)
        self.training_results["validation"]["iteration"].append(iteration)

    def get_record_tensor(self, data_set="train"):

        with tf.name_scope("Input_pipeline_"+data_set):

            with tf.device("/cpu:0"):
                data_path = self.tf_record_path + data_set + "_inlength_" + str(self.tensor_length) \
                + "_seql_" + str(self.sequence_length) + ".tfrecord"
                self.dataset[data_set] = tf.data.TFRecordDataset(data_path)

                # Use `tf.parse_single_example()` to extract data from a `tf.Example`
                # protocol buffer, and perform any additional per-record preprocessing.
                def parser(record):
                    feature = {'est_counts': tf.FixedLenFeature([], tf.string, default_value=""),
                               'est_errors': tf.FixedLenFeature([], tf.string, default_value=""),
                               "label": tf.FixedLenFeature((), tf.int64,
                                                           default_value=tf.zeros([], dtype=tf.int64)),
                               "days": tf.FixedLenFeature([], tf.string, default_value=""),
                               "seq_length": tf.FixedLenFeature((), tf.int64,
                                                                default_value=tf.zeros([], dtype=tf.int64)),
                               "variable_flag": tf.FixedLenFeature((), tf.int64,
                                                                   default_value=tf.zeros([], dtype=tf.int64))}
                    parsed = tf.parse_single_example(record, feature)

                    # Perform additional preprocessing on the parsed data.
                    # TODO:FIX THIS FROM "TO TF RECORD"
                    est_counts = tf.decode_raw(parsed["est_counts"], tf.float32)
                    est_errors = tf.decode_raw(parsed["est_errors"], tf.float32)
                    obs_days = tf.decode_raw(parsed["days"], tf.float32)
                    # obs_days = tf.cast(obs_days, dtype=tf.float32)
                    # image = tf.cast(image, dtype=tf.float32)
                    # print(image.get_shape())
                    label = tf.cast(parsed["label"], tf.int64)
                    variable_flag = tf.cast(parsed["variable_flag"], tf.int64)
                    seq_len = tf.cast(parsed["seq_length"], tf.int64)
                    obs_days = tf.reshape(obs_days, [self.sequence_length, self.tensor_length], name="day_decoder")
                    est_counts = tf.reshape(est_counts, [self.sequence_length, self.tensor_length], name="counts_decoder")
                    est_errors = tf.reshape(est_errors, [self.sequence_length, self.tensor_length], name="errors_decoder")
                    # label = tf.decode_raw(parsed["label"], tf.float32)
                    # print(label.get_shape)
                    # label = tf.reshape(label, [2])
                    return est_counts, est_errors, label, obs_days, seq_len, variable_flag

                # Use `Dataset.map()` to build a pair of a feature dictionary and a label
                # tensor for each example.
                self.dataset[data_set] = self.dataset[data_set].map(parser, num_parallel_calls=1)
                self.dataset[data_set] = self.dataset[data_set].prefetch(buffer_size=self.buffer_size)
                if data_set == "train":
                    self.dataset[data_set] = self.dataset[data_set].shuffle(buffer_size=self.buffer_size)
                self.dataset[data_set] = self.dataset[data_set].batch(self.batch_size)
                #if self.just_variables:
                #    self.dataset[data_set] = self.dataset[data_set].filter(lambda imgs, labl, ds, sl: tf.less(labl, 4))
                self.dataset[data_set] = self.dataset[data_set].filter(lambda f, errf, labl, ds, sl, fl: tf.equal(tf.shape(f)[0],
                                                                                                                  self.batch_size))
                self.dataset[data_set] = self.dataset[data_set].repeat()
                # iterator = dataset.make_one_shot_iterator()
                self.iterator[data_set] = self.dataset[data_set].make_initializable_iterator()
                with tf.name_scope("initializer"):
                    self.init_iterator[data_set] = self.iterator[data_set].initializer
                    tf.add_to_collection("buffer", value=self.init_iterator[data_set])
                self.sess.run(self.init_iterator[data_set])
                counts, errors, labels, days, seq_l, var_flag = self.iterator[data_set].get_next()
                return counts, errors, labels, days, seq_l, var_flag

if __name__ == "__main__":
    #gpu_id, n_runs, sequence_length, tensor_length, four_class_problem = sys.argv[1:]
    #n_runs = 2
    gpu_id = str(2)
    n_runs = str(1)
    sequence_length = str(20)
    tensor_length = str(3)
    four_class_problem = str(0)
    device = "/gpu:"+gpu_id
    if int(four_class_problem) == 1:
        print("DOING SIMPLIFIED PROBLEM")
        tf_record_path = "/home/rcarrasco/simulated_data/tf_record/simplfified_no_nan/"
        log_path = "/home/rcarrasco/network_train/lightcurve_classifier/june10/seql_" + sequence_length + "/"
        experiment_name = "simplified_"+sequence_length
        n_classes = 4
    else:
        print("DOING COMPLETE PROBLEM")
        tf_record_path = "/home/rcarrasco/simulated_data/tf_record/complete_no_nan/"
        log_path = "/home/rcarrasco/network_train/lightcurve_classifier/june10/seql_" + sequence_length + "/"
        experiment_name = "complete_"+sequence_length
        n_classes = 7

    band = "g"
    just_variables = False #VERY VERY HARDCODED
    #sequence_length_array = [20, 22, 18]
    # sequence_length = 20
    batch_size = 256
    extend_sequences = True
    same_batch = False
    static_batch = False
    # set_prop = [0.8, 0.1, 0.1]
    with_tf_record = True
    buffer_size = 2048
    probability_feedback = True
    # with_dropout = [True, ]
    #n_runs = [1, ]
    val_results = []
    variable_flag_coef = np.array([0.01, 0.1, 1])*(n_classes)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    for i in range(int(n_runs)):
    #for var_coef in variable_flag_coef:

        sequence_dir = "_run"+str(i)
        print("-----------------------------------" + sequence_dir + "-----------------------------------")
        convnet = LightCurveClassifier(sequence_length=int(sequence_length),
                                       batch_size=batch_size,
                                       static_batch=static_batch,
                                       logdir=log_path+experiment_name+sequence_dir,
                                       experiment_name=experiment_name,
                                       adam_step=True,
                                       learning_rate=5e-4,
                                       same_batch=same_batch,
                                       extend_sequences=extend_sequences,
                                       n_classes=n_classes,
                                       band=band,
                                       with_tf_record=with_tf_record,
                                       tf_record_path=tf_record_path,
                                       buffer_size=buffer_size,
                                       tensor_length=int(tensor_length),
                                       with_dropout=True,
                                       device=device,
                                       just_variables=just_variables,
                                       prob_feedback=probability_feedback,
                                       flag_loss_coef=0)

        start = time.time()
        loss, acc = convnet.train_iterations(30000)
        val_results.append([loss, acc])
        end = time.time()
        convnet.sess.close()
        print("total time: " + str(end - start))
    print(val_results)
    print("wena")
    # print(train_accuracy)