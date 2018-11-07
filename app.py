import os
import traceback

from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from wtforms import TextAreaField

from utils import exec_user_code, complete_user_code, default_feature_code

from data import DataLoader, DataLoadingError
from features import FeatureParser
from models import TFModel


def create_app():
    application = Flask(__name__)
    Bootstrap(application)
    return application


def get_data(url):
    data_loader = DataLoader(local_data_dir="./data")
    return data_loader.load_data(data_path=url)


def parse_feature_code(code_raw_string):

    complete_code = complete_user_code(user_code=code_raw_string)
    module_with_user_code = exec_user_code(code=complete_code)
    feature_list = getattr(module_with_user_code, "features")

    return FeatureParser(features=feature_list)


# INPUT FORM FOR FEATURE TRANSFORMATION CODE
class CodeInputForm(FlaskForm):
    code = TextAreaField("Put your feature transformation code in the text box below:")


# CONSTANTS AND GLOBALS
BASE_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
NO_OF_ROWS_TO_SHOW = 5
TRACEBACK_LIMIT = 2

DATASETS = {
    "RedWineQuality": {"URL": BASE_DATA_URL+"wine-quality/winequality-red.csv",
                       "target": "quality"},
    "WhiteWineQuality": {"URL": BASE_DATA_URL+"wine-quality/winequality-white.csv",
                         "target": "quality"},
    "BreastCancerWisconsinDataset": {"URL":BASE_DATA_URL+"breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                                     "target": 10},
    "ErrorDataset": {"URL": BASE_DATA_URL+"this-is-not-a-dataset.fsv",
                     "target": "there_is_no_target"}
}

# POINTER TO A LOADED PANDAS DATAFRAME OF DATA TO PLAY WITH
loaded_data = None
loaded_data_name = None


# INIT APP
app = create_app()
app.config['SECRET_KEY'] = "any secret string"


@app.route(rule="/", methods=["GET", "POST"])
def index():

    if request.method == "POST":
        selected_dataset = request.form.get("select_data")
        return redirect(url_for("features", dataset=selected_dataset))
    else:
        return render_template("index.html",
                               datasets=DATASETS)


@app.route(rule="/features", methods=["GET", "POST"])
def features():

    global loaded_data
    global loaded_data_name

    form = CodeInputForm()

    if request.method == "POST":

        code_raw_string = form.code.data
        loaded_data_name = request.form.get("select_data")

        raw_data = None

        try:
            loaded_data = get_data(url=DATASETS[loaded_data_name]["URL"])
            raw_data = loaded_data.head(NO_OF_ROWS_TO_SHOW).to_json()

            parser = parse_feature_code(code_raw_string=code_raw_string)
            parsed_data_df = parser.parse_to_df(data=loaded_data)
            transformed_features = parsed_data_df.head(NO_OF_ROWS_TO_SHOW).to_json()

            return render_template("feature_transform.html",
                                   selected_dataset=loaded_data_name,
                                   form=form,
                                   datasets=DATASETS,
                                   raw_data=raw_data,
                                   features=transformed_features)

        except (NameError, SyntaxError, AttributeError, KeyError, ValueError) as execution_error:
            transform_error = {"error": execution_error,
                               "traceback": traceback.format_exc(limit=TRACEBACK_LIMIT)}

            return render_template("feature_transform.html",
                                   form=form,
                                   selected_dataset=loaded_data_name,
                                   datasets=DATASETS,
                                   raw_data=raw_data,
                                   transform_error=transform_error)
        except DataLoadingError as data_error:
            data_loading_error = {"error": data_error,
                                  "traceback": traceback.format_exc(limit=TRACEBACK_LIMIT)}

            return render_template("feature_transform.html",
                                   form=form,
                                   datasets=DATASETS,
                                   selected_dataset=loaded_data_name,
                                   raw_data=raw_data,
                                   data_loading_error=data_loading_error)
    else:
        form.code.data = default_feature_code

        return render_template("feature_transform.html",
                               form=form,
                               datasets=DATASETS)


@app.route(rule="/training", methods=["GET", "POST"])
def training():

    global loaded_data
    global loaded_data_name

    form = CodeInputForm()

    if request.method == "POST":

        code_raw_string = form.code.data

        model_class = request.form.get("select_model")
        loaded_data_name = request.form.get("select_data")
        loaded_data = get_data(url=DATASETS[loaded_data_name]["URL"])

        try:
            parser = parse_feature_code(code_raw_string=code_raw_string)
            parsed_data_df = parser.parse_to_df(loaded_data)

            # ATTACH TARGET COLUMN TO TRANSFORMED FEATURES
            target_name = DATASETS[loaded_data_name]["target"]

            features_with_target = parsed_data_df

            prelim_target = loaded_data[target_name].astype(int)
            unique_target_values = prelim_target.unique()
            n_classes = len(unique_target_values)

            target_value_lookup = dict()

            for i, value in enumerate(sorted(unique_target_values)):
                target_value_lookup[value] = i

            features_with_target[target_name] = prelim_target

            for i, value in enumerate(prelim_target):
                prelim_target[i] = target_value_lookup[value]

            features_with_target[target_name] = prelim_target

            model_params = {"hidden_units": [3, 5, 3],
                            "n_classes": n_classes}

            model = TFModel(tf_estimator_class=model_class,
                            model_parameters=model_params,
                            feature_parser=parser,
                            model_export_directory="./model_export")

            model.train(features=features_with_target,
                        target=target_name,
                        num_steps=100)

            return render_template("model_training.html",
                                   form=form,
                                   datasets=DATASETS,
                                   success=True)
        except ValueError as error:
            print(error)
            return render_template("model_training.html",
                                   form=form,
                                   datasets=DATASETS,
                                   success=False)

    else:
        form.code.data = default_feature_code

        return render_template("model_training.html",
                               form=form,
                               datasets=DATASETS)


if __name__ == "__main__":

    app.run(host="0.0.0.0",
            port=5000,
            debug=True)
