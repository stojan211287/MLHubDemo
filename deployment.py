from flask import Flask, request, jsonify

from deployed_model import DeployedModel, DeploymentError

deployment = Flask(__name__)

deployed_model = DeployedModel()


@deployment.route(rule="/", methods=["GET"])
def get_models():
    pass


@deployment.route(rule="/deploy", methods=["GET"])
def deploy():

    model_code = request.args.get("model_code")

    if model_code:
        try:
            deployed_model.load_model(model_code=model_code)

            return jsonify(status="Model loaded successfully!"), 200

        except DeploymentError as error:
            return jsonify(error="Deployment error: %s" % (str(error), )), 500
    else:
        return jsonify(error="Model code query parameter missing."), 500


@deployment.route(rule="/predict", methods=["POST"])
def predict():

    data = request.json()

    model_predictions = deployed_model.get_predictions(data=data)

    return jsonify(predictions=model_predictions)


if __name__ == "__main__":

    deployment.run(host="0.0.0.0",
                   port=8000,
                   debug=True)
