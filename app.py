from flask import Flask, Response, render_template, request
from flask_bootstrap import Bootstrap

from utils import default_feature_code, model_param_lookup, construct_parser, encode_target, error_with_traceback
from constants import NO_OF_ROWS_TO_SHOW, DATASETS, DATA_ERRORS, TRAINING_ERRORS

from user import User
from data import DataLoadingError, DataLoader, MalformedDataUrl, DataNotFoundRemotly
from models import TFModel

import pandas as pd


def create_app():
    application = Flask(__name__)
    Bootstrap(application)
    return application


# TODO: GET RID OF THIS
data_with_model_predictions = None

# MAKE USER
user = User(datasets=DATASETS)

# MAKE DATA LOADER
data_loader = DataLoader(local_data_dir="./data")

# INIT APP
app = create_app()
app.config['SECRET_KEY'] = "ThisIsSuperSecret"


@app.route(rule="/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route(rule="/datasets", methods=["GET", "POST"])
def datasets():

    backend_response = {"data_loading_error": None,
                        "available_data_features": None,
                        "loaded_data_features": None}

    if request.method == "POST":
        # HANDLES POST REQUEST FROM Download BUTTON
        try:
            data_url = request.form.get("data_url")

            new_dataset_name = data_url.split("/")[-1].split(".")[0]

            user.loaded_data = data_loader.load_data(data_path=data_url)
            user.loaded_data_name = new_dataset_name

            user.add_dataset(url=data_url,
                             dataset_name=new_dataset_name)

            backend_response["loaded_data_features"] = user.loaded_data.columns

        except (MalformedDataUrl, DataNotFoundRemotly) as error:
            backend_response["data_loading_error"] = {"error_message": error}
    else:
        dataset_name = request.args.get("dataset")

        if dataset_name:

            try:
                user.loaded_data_name = dataset_name
                data_url = user.available_datasets[dataset_name]["URL"]
                user.loaded_data = data_loader.load_data(data_path=data_url)

                backend_response["available_data_features"] = user.loaded_data.describe().\
                                                                   head(NO_OF_ROWS_TO_SHOW).to_json()
            except DataNotFoundRemotly as error:
                backend_response["data_loading_error"] = {"error_message": error}
        else:
            pass

    return render_template("datasets.html",
                           user=user,
                           backend_response=backend_response)


@app.route(rule="/features", methods=["GET", "POST"])
def features():

    backend_response = {"raw_data_preview": None,
                        "feature_preview": None,
                        "transform_error": None,
                        "committed_feature_hash": None,
                        "data_loading_error": None}

    if request.method == "POST":

        user.feature_code = request.form.get("code_box")
        user.loaded_data_name = request.form.get("select_data")

        commit_was_pressed = request.form.get("commit_button")

        try:
            user.loaded_data = data_loader.load_data(data_path=user.available_datasets[user.loaded_data_name]["URL"])
            parser, completed_code = construct_parser(code_raw_string=user.feature_code)
            used_feature_generators = parser.get_all_generators()
            raw_features = set(user.loaded_data.columns.values) - used_feature_generators

            backend_response["raw_data_preview"] = user.loaded_data.head(NO_OF_ROWS_TO_SHOW).to_json()

            parsed_df = parser.parse_to_df(data=user.loaded_data)
            all_features = pd.concat([parsed_df,
                                      user.loaded_data.loc[:, list(raw_features)]],
                                     axis=1)
            feature_preview = all_features.head(NO_OF_ROWS_TO_SHOW).to_json()

            if commit_was_pressed:
                feature_hash = user.commit_features(feature_preview=feature_preview)
                backend_response["committed_feature_hash"] = feature_hash
            else:
                backend_response["feature_preview"] = feature_preview

        except DATA_ERRORS as error:
            backend_response["transform_error"] = error_with_traceback(error=error)

        except DataLoadingError as data_error:
            backend_response["data_loading_error"] = error_with_traceback(error=data_error)
    else:
        user.feature_code = default_feature_code

        if request.args.get("dataset"):
            user.loaded_data_name = request.args.get("dataset")

    return render_template("feature_transform.html",
                           user=user,
                           backend_response=backend_response)


@app.route(rule="/training", methods=["GET", "POST"])
def training():

    global data_with_model_predictions

    backend_response = {"model_class": None,
                        "model_eval": None,
                        "model_docs": None,
                        "training_error": None}

    if request.method == "POST":

        user.feature_code = request.form.get("code_box")
        user.loaded_data_name = request.form.get("select_data")
        model_class = request.form.get("select_model")

        try:
            user.loaded_data = data_loader.load_data(data_path=user.available_datasets[user.loaded_data_name]["URL"])

            parser, completed_code = construct_parser(code_raw_string=user.feature_code)
            parsed_data_df = parser.parse_to_df(user.loaded_data)

            # ATTACH TARGET COLUMN TO TRANSFORMED FEATURES
            target_name = user.available_datasets[user.loaded_data_name]["target"]

            encoded_target, n_classes = encode_target(raw_target=user.loaded_data[target_name])
            parsed_data_df[target_name] = encoded_target

            model_params = model_param_lookup[model_class]
            model_params["n_classes"] = n_classes

            model = TFModel(tf_estimator_class=model_class,
                            model_parameters=model_params,
                            feature_parser=parser,
                            model_export_directory="./model_export")

            model_eval, data_with_model_predictions = model.train(features=parsed_data_df,
                                                                  target=target_name,
                                                                  num_steps=100)

            # TODO: TAKE A LOOK AT THIS
            model_docs = model.model_instance.__init__.__doc__.split("\n")

            backend_response["model_class"] = model_class
            backend_response["model_eval"] = model_eval
            backend_response["model_docs"] = model_docs

        except TRAINING_ERRORS as error:
            backend_response["training_error"] = error_with_traceback(error=error)
    else:
        user.feature_code = default_feature_code

    return render_template("model_training.html",
                           user=user,
                           backend_response=backend_response)


@app.route(rule="/mlhub-model-predictions", methods=["GET"])
def get_predictions():

    if data_with_model_predictions is not None:

        def prediction_generator():

            row_number = 0

            for _, row_entries in data_with_model_predictions.iterrows():
                if row_number == 0:
                    yield ','.join(data_with_model_predictions.columns.values) + '\n'
                else:
                    yield ','.join(row_entries.values.astype(str)) + '\n'

                row_number += 1

        return Response(prediction_generator(),
                        mimetype='text/csv')
    else:
        raise ValueError("No model prediction data available. Train a model first.")


@app.route(rule="/docs", methods=["GET", "POST"])
def docs():
    return render_template("documentation.html")


@app.route(rule="/commits", methods=["GET"])
def commits():
    return render_template("commits.html",
                           user=user)


@app.route(rule="/deployment", methods=["GET"])
def deployment():
    return render_template("deployment.html",
                           user=user)


if __name__ == "__main__":

    app.run(host="0.0.0.0",
            port=5000,
            debug=True)
