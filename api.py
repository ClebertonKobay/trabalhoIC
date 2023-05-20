from flask import Flask, request, jsonify
from flask_cors import CORS
import wrapper_sklearn  

app = Flask(__name__)
CORS(app)
accuracy = wrapper_sklearn.api()

@app.route('/predict', methods=['GET'])
def predict():
    response = {'accuracy': accuracy}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)