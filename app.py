import numpy
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)
model_hl = pickle.load(open('models/cat_hl.pkl', 'rb'))
model_cl = pickle.load(open('models/cat_cl.pkl', 'rb'))

@app.route('/')
def home():
    return {'message': 'Hello World'}


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    Accepts a JSON payload containing input features for heating and cooling load predictions,
    and returns a JSON response with the predicted heating and cooling loads.

    Example JSON input:
    {
        "relative_compactness": 0.74,
        "surface_area": 686.0,
        "wall_area": 245.0,
        "roof_area": 220.5,
        "overall_height": 3.5,
        "orientation": 2,
        "glazing_area": 0.1,
        "glazing_area_distribution": 3
    }

    Example JSON output:
    {
        "hl_prediction": 21.5,
        "cl_prediction": 28.8,
        "message": "success"
    }

    Returns:
        A JSON object with keys for heating load prediction, cooling load prediction,
        and a success message. The predicted heating and cooling loads are exponential
        transformations of the neural network outputs.
    '''
    data = request.get_json(force=True)

    hl_prediction = model_hl.predict([numpy.array(list(data.values()))])
    cl_prediction = model_cl.predict([numpy.array(list(data.values()))])

    hl_output = hl_prediction[0]
    hl_pred = numpy.exp(hl_output)

    cl_output = cl_prediction[0]
    cl_pred = numpy.exp(cl_output)

    return {'hl_prediction': hl_pred, 'cl_prediction': cl_pred, 'message': 'success'}
