import os

from datetime import datetime

from flask import (
    Flask,
    Response,
    render_template,
    request,
    flash,
    url_for,
    redirect,
    jsonify,
)
from flask_bootstrap import Bootstrap

from mlhub.utils import construct_parser, error_with_traceback

from mlhub.constants import (
    NO_OF_ROWS_TO_SHOW,
    DATASETS,
    DATA_ERRORS,
    ADMIN,
    SUPER_SAFE_ADMIN_PASSWORD,
    default_feature_code,
    default_model_code,
)

from mlhub.session import UserSession

from mlhub.data import DataLoader
from mlhub.data_errors import DataLoadingError, MalformedDataUrl, DataNotFoundRemotly

from mlhub.models import KerasModel
from mlhub.models_errors import UserCodeExecutionError, InputDataError

from mlhub.deployed_model import DeploymentError, DeployedModel


def create_app():
    application = Flask(__name__)
    Bootstrap(application)
    return application


# MAKE USER
user_session = UserSession(datasets=DATASETS)

# MAKE DATA LOADER
data_loader = DataLoader(local_data_dir="./data")

# INIT APP
app = create_app()

# RANDOMIZE SECRET KEY
app.config["SECRET_KEY"] = os.urandom(12)

deployed_model = DeployedModel()


@app.route("/login", methods=["GET", "POST"])
def do_admin_login():
    if (
        request.form["password"] == SUPER_SAFE_ADMIN_PASSWORD
        and request.form["username"] == ADMIN
    ):
        user_session.logged_in = True
    else:
        flash("Wrong password!")
    return index()


@app.route(rule="/", methods=["GET", "POST"])
def index():
    if not user_session.logged_in:
        return render_template("login.html")
    else:
        return render_template("index.html")


@app.route(rule="/datasets", methods=["GET", "POST"])
def datasets():

    backend_response = {
        "data_loading_error": None,
        "available_data_features": None,
        "loaded_data_features": None,
    }

    if request.method == "POST":
        # HANDLES POST REQUEST FROM Download BUTTON
        try:
            data_url = request.form.get("data_url")

            new_dataset_name = data_url.split("/")[-1].split(".")[0]

            user_session.loaded_data = data_loader.load_data(data_path=data_url)
            user_session.loaded_data_name = new_dataset_name

            user_session.add_dataset(url=data_url, dataset_name=new_dataset_name)

            backend_response["loaded_data_features"] = user_session.loaded_data.columns

        except (MalformedDataUrl, DataNotFoundRemotly) as error:
            backend_response["data_loading_error"] = {"error_message": error}
    else:
        dataset_name = request.args.get("dataset")

        if dataset_name:
            try:
                user_session.loaded_data_name = dataset_name
                data_url = user_session.available_datasets[dataset_name]
                user_session.loaded_data = data_loader.load_data(data_path=data_url)

                backend_response["available_data_features"] = (
                    user_session.loaded_data.describe()
                    .head(NO_OF_ROWS_TO_SHOW)
                    .to_json()
                )

            except DataNotFoundRemotly as error:
                backend_response["data_loading_error"] = {"error_message": error}

    return render_template(
        "datasets.html", user=user_session, backend_response=backend_response
    )


@app.route(rule="/features", methods=["GET", "POST"])
def features():

    backend_response = {
        "raw_data_preview": None,
        "feature_preview": None,
        "transform_error": None,
        "committed_feature_hash": None,
        "data_loading_error": None,
    }

    if request.method == "POST":

        user_session.feature_code = request.form.get("code_box")
        user_session.loaded_data_name = request.form.get("select_data")

        commit_was_pressed = request.form.get("commit_button")

        try:
            user_session.loaded_data = data_loader.load_data(
                data_path=user_session.available_datasets[user_session.loaded_data_name]
            )
            parser, completed_code = construct_parser(
                code_raw_string=user_session.feature_code
            )

            backend_response["raw_data_preview"] = user_session.loaded_data.head(
                NO_OF_ROWS_TO_SHOW
            ).to_json()

            all_features = parser.add_parsed_to_data(data=user_session.loaded_data)
            feature_preview = all_features.head(NO_OF_ROWS_TO_SHOW).to_json()

            if commit_was_pressed:
                user_session.commit_features(
                    feature_preview=feature_preview,
                    all_features=all_features.columns.values,
                )
                backend_response[
                    "committed_feature_hash"
                ] = user_session.latest_feature_commit
            else:
                backend_response["feature_preview"] = feature_preview

        except DATA_ERRORS as error:
            backend_response["transform_error"] = error_with_traceback(error=error)

        except DataLoadingError as data_error:
            backend_response["data_loading_error"] = error_with_traceback(
                error=data_error
            )
    else:
        user_session.feature_code = default_feature_code

        if request.args.get("dataset"):
            user_session.loaded_data_name = request.args.get("dataset")

    return render_template(
        "feature_transform.html", user=user_session, backend_response=backend_response
    )


@app.route(rule="/training", methods=["GET", "POST"])
def training():

    backend_response = {"model_eval": None, "training_error": None}

    if request.method == "POST":

        user_session.model_code = request.form.get("model_box")
        feature_commit = request.form.get("select_commit")
        target_name = request.form.get("select_target")

        try:
            user_session.loaded_data_name = user_session.get_commit_data_name(
                commit_hash=feature_commit
            )
            user_session.loaded_data = data_loader.load_data(
                data_path=user_session.available_datasets[user_session.loaded_data_name]
            )

            feature_def_list = user_session.get_feature_def_list(
                commit_hash=feature_commit
            )
            feature_code = ";".join(feature_def_list)

            parser, completed_code = construct_parser(code_raw_string=feature_code)

            all_features = parser.add_parsed_to_data(data=user_session.loaded_data)

            model = KerasModel(model_code=user_session.model_code)

            model_eval, data_with_preds = model.train_and_eval(
                features=all_features, target_name=target_name
            )

            user_session.latest_trained_model = model
            user_session.latest_predictions = data_with_preds

            backend_response["model_eval"] = model_eval

        except TypeError:
            no_commit_error_message = "You must select a commit to train a model!"
            backend_response["training_error"] = error_with_traceback(
                error=Exception(no_commit_error_message)
            )

        except (UserCodeExecutionError, InputDataError) as error:
            code_exec_error_message = (
                "An error occurred while executing your code: %s" % (str(error),)
            )
            backend_response["training_error"] = error_with_traceback(
                error=Exception(code_exec_error_message)
            )
    else:
        user_session.model_code = default_model_code

    return render_template(
        "model_training.html", user=user_session, backend_response=backend_response
    )


@app.route(rule="/mlhub-model-predictions", methods=["GET"])
def get_predictions():

    if user_session.latest_predictions is not None:

        data_with_model_predictions = user_session.latest_predictions

        def prediction_generator():

            row_number = 0

            for _, row_entries in data_with_model_predictions.iterrows():
                if row_number == 0:
                    yield ",".join(data_with_model_predictions.columns.values) + "\n"
                else:
                    yield ",".join(row_entries.values.astype(str)) + "\n"

                row_number += 1

        return Response(prediction_generator(), mimetype="text/csv")
    else:
        raise ValueError("No model prediction data available. Train a model first.")


@app.route(rule="/docs", methods=["GET", "POST"])
def docs():
    return render_template("documentation.html")


@app.route(rule="/commits", methods=["GET"])
def commits():
    return render_template("commits.html", user=user_session)


@app.route(rule="/deployment", methods=["GET"])
def deployment():
    return render_template("deployment.html", user=user_session)


@app.route(rule="/deploy_model", methods=["GET"])
def deploy_model():

    model_code = (
        user_session.loaded_data_name + "_" + datetime.now().strftime("%Y-%m-%d")
    )

    latest_model = user_session.latest_trained_model
    latest_model.deploy(model_id=model_code)

    try:
        deployed_model.load_model(model_code=model_code)

        user_session.available_models.update(
            {
                model_code: {
                    "model_instance": latest_model,
                    "model_summary": latest_model.summarise(),
                }
            }
        )
        return redirect(url_for("commits"))

    except DeploymentError as error:
        return jsonify(error="Deployment error: %s" % (str(error),)), 500


@app.route(rule="/logout", methods=["GET"])
def logout():
    if user_session.logged_in:
        user_session.logged_in = False
        return redirect(url_for("index"))
    else:
        pass


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)
