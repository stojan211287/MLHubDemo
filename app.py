import traceback

from flask import Flask, Response, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from wtforms import TextAreaField

from utils import default_feature_code, model_param_lookup, get_data, construct_parser, encode_target
from constants import NO_OF_ROWS_TO_SHOW, TRACEBACK_LIMIT, DATASETS, DATA_ERRORS, TRAINING_ERRORS

from data import DataLoadingError
from models import TFModel


def create_app():
    application = Flask(__name__)
    Bootstrap(application)
    return application


# INPUT FORM FOR FEATURE TRANSFORMATION CODE
class CodeInputForm(FlaskForm):
    code = TextAreaField("Put your feature transformation code in the text box below:")


# LOADED DATA CONTAINER
loaded_data = None
loaded_data_name = None
data_with_model_predictions = None


# INIT APP
app = create_app()
app.config['SECRET_KEY'] = "ThisIsSuperSecret"


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

            parser = construct_parser(code_raw_string=code_raw_string)
            parsed_data_df = parser.parse_to_df(data=loaded_data)

            feature_rows = parsed_data_df.head(NO_OF_ROWS_TO_SHOW).to_json()

            return render_template("feature_transform.html",
                                   selected_dataset=loaded_data_name,
                                   form=form,
                                   datasets=DATASETS,
                                   raw_data=raw_data,
                                   features=feature_rows)

        except DATA_ERRORS as error:
            transform_error = {"error": error,
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
    global data_with_model_predictions

    form = CodeInputForm()

    if request.method == "POST":

        code_raw_string = form.code.data

        model_class = request.form.get("select_model")
        loaded_data_name = request.form.get("select_data")
        loaded_data = get_data(url=DATASETS[loaded_data_name]["URL"])

        try:
            parser = construct_parser(code_raw_string=code_raw_string)
            parsed_data_df = parser.parse_to_df(loaded_data)

            # ATTACH TARGET COLUMN TO TRANSFORMED FEATURES
            target_name = DATASETS[loaded_data_name]["target"]

            encoded_target, n_classes = encode_target(raw_target=loaded_data[target_name])
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

            return render_template("model_training.html",
                                   form=form,
                                   datasets=DATASETS,
                                   model_eval=model_eval,
                                   model_class=model_class,
                                   loaded_data_name=loaded_data_name)

        except TRAINING_ERRORS as error:
            training_error = {"error": error,
                              "traceback": traceback.format_exc(limit=TRACEBACK_LIMIT)}
            return render_template("model_training.html",
                                   form=form,
                                   datasets=DATASETS,
                                   training_error=training_error)

    else:
        form.code.data = default_feature_code

        return render_template("model_training.html",
                               form=form,
                               datasets=DATASETS)


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


if __name__ == "__main__":

    app.run(host="0.0.0.0",
            port=5000,
            debug=True)
