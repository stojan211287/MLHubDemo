import importlib

import numpy as np
import pandas as pd
import tensorflow as tf


class ConvergenceDetected(Exception):
    pass


class LogReg:

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def __init__(self, n_classes, num_steps, learning_rate=1, tolerance=1e-18):

        self.n_classes = n_classes
        self.num_steps = num_steps

        self.learning_rate = learning_rate
        self.tolerance = tolerance

        # MODEL PARAMETERS

        self.betas = None
        self._activation = lambda x: LogReg._softmax(x)
        self._linear = lambda x, betas: np.dot(x, betas)

        self._f = lambda x, betas:  self._activation(self._linear(x, betas))

        self._log_loss = lambda h_x: (h_x.sum(axis=1) - np.log(np.exp(h_x).sum(axis=1))).mean(axis=0)

    def fit(self, features, target):

        # INITIALIZE BETAS
        self.betas = np.zeros(shape=(features.shape[1], self.n_classes))

        try:
            iteration = 0

            while iteration < self.num_steps:
                loss, loss_gradient = self._mle_loss_and_gradient(features=features,
                                                                  target=target)

                print("Loss: %0.6f" % (loss, ))
                print(loss_gradient)
                print(self.betas)

                self.betas = self._update_params(params=self.betas,
                                                 gradient=loss_gradient)

                new_loss, new_loss_gradient = self._mle_loss_and_gradient(features=features,
                                                                          target=target)

                self._check_convergence(loss=loss,
                                        new_loss=new_loss,
                                        gradient=loss_gradient,
                                        new_gradient=new_loss_gradient)

                iteration += 1

        except ConvergenceDetected:
            return self.betas

    def _update_params(self, params, gradient):
        return params - self.learning_rate*gradient

    def _check_convergence(self, loss, new_loss, gradient, new_gradient):
        if (np.linalg.norm(gradient.ravel() - new_gradient.ravel()) <= self.tolerance) or \
           (np.linalg.norm(loss.ravel() - new_loss.ravel()) <= self.tolerance):
            raise ConvergenceDetected
        else:
            pass

    def _mle_loss_and_gradient(self, features, target):

        class_hist = np.histogram(target, bins=self.n_classes)[0]
        target_indices = np.argsort(target)

        loss_gradient = np.zeros(self.betas.shape)

        class_beginning_index = 0

        probability_dist = self._f(x=features,
                                   betas=self.betas)

        loss = self._log_loss(probability_dist)

        for class_index, class_size in enumerate(class_hist):

            indices = target_indices[class_beginning_index: class_beginning_index+class_size]

            subset_of_prob = probability_dist[indices, class_index][:, np.newaxis]

            loss_gradient[:, class_index] += (features[indices, :] - subset_of_prob*features[indices, :]).mean(axis=0)
            loss_gradient[:, class_index] -= (probability_dist[:, class_index][:, np.newaxis]*features).mean(axis=0)

            class_beginning_index = class_size

        return loss, loss_gradient

    def predict(self, features):
        return self._f(x=features,
                       betas=self.betas)


class TFModel:

    def __init__(self, tf_estimator_class, model_parameters, feature_parser, model_export_directory):

        self.tf_class_name = tf_estimator_class
        self.model_params = model_parameters
        self.parser = feature_parser
        self.export_dir = model_export_directory

        self.model_instance = None

        try:
            self.model_class = getattr(importlib.import_module("tensorflow.estimator"), self.tf_class_name)
        except ImportError:
            raise Exception("Cannot import tensorflow.estimator")
        except AttributeError:
            raise Exception("There does not seem to be a %s class in tensorflow.estimator" % (self.tf_class_name, ))

    def train(self, features, target, num_steps):

        try:
            assert isinstance(features, pd.DataFrame)
            assert isinstance(target, str)

        except AssertionError:
            raise ValueError("The TFModel.train has the signature features: pd.DataFrame, target: String")

        train_features, test_features = TFModel._train_test_split(features=features)

        train_target = train_features[target]
        test_target = test_features[target]

        # SEPARATE TARGET COLUMN AND REST OF FEATURES
        training_set = train_features.drop(target, axis=1)
        test_set = test_features.drop(target, axis=1)

        training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_set,
                                                                y=train_target,
                                                                batch_size=training_set.shape[0],
                                                                num_epochs=num_steps,
                                                                shuffle=False)

        test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_set,
                                                            y=test_target,
                                                            batch_size=test_set.shape[0],
                                                            num_epochs=1,
                                                            shuffle=False)

        instance_model_params = self.model_params.copy()
        instance_model_params["feature_columns"] = self.parser.get_tf_feature_columns(data=features)

        self.model_instance = self.model_class(**instance_model_params)

        train_spec = tf.estimator.TrainSpec(input_fn=training_input_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn)

        evaluations, _ = tf.estimator.train_and_evaluate(self.model_instance, train_spec, eval_spec)

        class_predictions = []

        for prediction_wrapper in self.model_instance.predict(input_fn=test_input_fn):
            class_predictions.append(prediction_wrapper["class_ids"][0])

        test_features.loc[:, self.tf_class_name+"_predictions"] = class_predictions

        return evaluations, test_features

    @staticmethod
    def _train_test_split(features, test_set_size=0.1):

        train_indices = np.random.choice(features.index.values,
                                         size=int((1.-test_set_size)*features.shape[0]))

        test_indices = np.setdiff1d(features.index.values,
                                    train_indices,
                                    assume_unique=True)

        return features.iloc[train_indices, :], features.iloc[test_indices, :]

    def deploy(self):

        production_features = {}

        for feature_definition in self.parser.features:

            tf_feature_type = tf.float32 if feature_definition.kind == "numeric" else tf.string
            production_features[feature_definition.name] = tf.FixedLenFeature(shape=[1],
                                                                              dtype=tf_feature_type)

        serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec=production_features,
                                                                                       default_batch_size=None)

        self.model_instance.export_savedmodel(export_dir_base=self.export_dir,
                                              serving_input_receiver_fn=serving_input_fn)


if __name__ == "__main__":

    N = 10000
    d = 2
    n_classes = 3

    data = np.random.normal(loc=0., scale=1.0, size=(N, d))
    target = np.random.randint(low=0, high=n_classes, size=(N, ))

    logreg = LogReg(n_classes=n_classes,
                    num_steps=1000)

    fitted_betas = logreg.fit(features=data, target=target)

    print(fitted_betas)




