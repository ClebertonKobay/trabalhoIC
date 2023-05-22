from flask import Flask, request, jsonify
from flask_cors import CORS
import wrapper_sklearn  

app = Flask(__name__)
CORS(app)
wrapper_AG_solution , wrapper_AG_acurracy, elapsed_time_AG, wrapper_hillClimbing_solution,wrapper_hillClimbing_acurracy, elapsed_time_hill,All_accuracy_AG, All_accuracy_Hill = wrapper_sklearn.api()

@app.route('/predict', methods=['GET'])
def predict():
    response = {
        'wrapper_AG_solution': wrapper_AG_solution,
        'wrapper_AG_acurracy': wrapper_AG_acurracy,
        'elapsed_time_AG':elapsed_time_AG ,
        'wrapper_hillClimbing_solution': wrapper_hillClimbing_solution,
        'wrapper_hillClimbing_acurracy': wrapper_hillClimbing_acurracy,
        'elapsed_time_hill': elapsed_time_hill,
        'All_accuracy_AG' : All_accuracy_AG,
        'All_accuracy_Hill' : All_accuracy_Hill
        }
    return jsonify(response)
# Função que retorna a acuracia de cada geração, para construir o gráfico
if __name__ == '__main__':
    app.run(debug=True)