from flask import Flask, jsonify, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm

from wtforms import TextAreaField

from utils import exec_user_code, complete_user_code, default_feature_code

from data import DataLoader
from features import FeatureParser


def create_app():
    application = Flask(__name__)
    Bootstrap(application)

    return application


# INPUT FORM FOR FEATURE TRANSFORMATION CODE
class CodeInputForm(FlaskForm):
    code = TextAreaField("Put your feature transformation code in the text box below:")


# INIT APP
app = create_app()
app.config['SECRET_KEY'] = "any secret string"

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
NO_OF_ROWS_TO_SHOW = 8


def get_data(url):

    data_loader = DataLoader(local_data_dir="./data")
    return data_loader.load_data(data_path=url)


# GET DATA FROM FILE OR URL
data = get_data(url=URL)


@app.route(rule="/", methods=["GET", "POST"])
def index():

    form = CodeInputForm()

    if request.method == "POST":

        code_raw_string = form.code.data
        complete_code = complete_user_code(user_code=code_raw_string)

        try:
            module_with_user_code = exec_user_code(code=complete_code)
            features = getattr(module_with_user_code, "features")

            parser = FeatureParser(features=features)
            parsed_data_df = parser.parse_to_df(data)

            transformed_features = parsed_data_df.head(NO_OF_ROWS_TO_SHOW).to_json()

            return render_template("index.html",
                                   form=form,
                                   raw_data=data.head(NO_OF_ROWS_TO_SHOW).to_json(),
                                   features=transformed_features,
                                   has_transformed=True)

        except (NameError, SyntaxError, AttributeError, KeyError, ValueError) as execution_error:
            return render_template("index.html",
                                   form=form,
                                   raw_data=data.head(NO_OF_ROWS_TO_SHOW).to_json(),
                                   error=str(execution_error))
            #jsonify(execution_error=str(execution_error))
    else:
        form.code.data = default_feature_code
        return render_template("index.html",
                               form=form,
                               raw_data=data.head(NO_OF_ROWS_TO_SHOW).to_json(),
                               has_transformed=False)


@app.route(rule="/transform/<string:features>", methods=["GET"])
def transform_features(features):
    return render_template("feature_preview.html", transformed_features=features)


@app.route(rule="/train", methods=["POST"])
def train_model():
    pass


if __name__ == "__main__":

    app.run(host="0.0.0.0",
            port=5000,
            debug=True)
