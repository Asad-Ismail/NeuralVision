from flask import Flask, jsonify
import subprocess
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/api/start_training', methods=['GET'])
def start_training():
    #process = subprocess.Popen(['python', 'train.py'])
    return jsonify({'message': 'Training started'}), 200

if __name__ == '__main__':
    app.run(debug=True)
