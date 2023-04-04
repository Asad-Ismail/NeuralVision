from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import subprocess
import threading
import json

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

def start_training_thread():
    process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)

    for line in process.stdout:
        log = line.rstrip()
        try:
            metric = json.loads(log)
            socketio.emit('metric', metric)
        except ValueError:
            pass
        socketio.emit('log', log)

    process.communicate()

@app.route('/api/start_training', methods=['GET'])
def start_training():
    training_thread = threading.Thread(target=start_training_thread)
    training_thread.start()

    return jsonify({'message': 'Training started'}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True)
