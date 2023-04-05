from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask import request
import subprocess
import threading
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

status = None
training_thread = None
observer = None

class TrainingDataHandler(FileSystemEventHandler):
    def __init__(self, socketio):
        self.socketio = socketio
        self.stop_word = 'Training completed'
        self.metrics = []

    def on_modified(self, event):
        global status
        if not event.is_directory and event.src_path.endswith('.json'):
            with open(event.src_path, 'r') as f:
                for line in f:
                    if self.stop_word in line.strip():
                        status = "Done"
                        break
                    try:
                        metric = json.loads(line.strip())
                        if metric not in self.metrics:
                            self.metrics.append(metric)
                            self.socketio.emit('metric', metric)
                        status = "Training"
                    except ValueError:
                        pass

def read_metrics_from_file(socketio):
    global observer
    event_handler = TrainingDataHandler(socketio)
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def start_training_thread():
    process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    process.communicate()
    

@app.route('/api/set_hyperparameters', methods=['POST'])
def set_hyperparameters():
    hyperparameters = request.get_json()
    logging.debug(f"Received hyperparameters: {hyperparameters}")
    # Save hyperparameters to a JSON file
    with open('hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    return jsonify({'message': 'Hyperparameters submitted'}), 200

@app.route('/api/start_training', methods=['GET'])
def start_training():
    global status, training_thread
    if training_thread and training_thread.is_alive():
        return jsonify({'message': 'Training is already running'}), 200

    training_thread = threading.Thread(target=start_training_thread)
    training_thread.start()
    read_metrics_from_file(socketio)
    status = "Training"
    return jsonify({'message': 'Training started'}), 200

@app.route('/api/stop_training', methods=['GET'])
def stop_training():
    global status, training_thread, observer
    if training_thread and training_thread.is_alive():
        training_thread.terminate()
        training_thread = None

    if observer:
        observer.stop()
        observer = None
        
    status = None

    return jsonify({'message': 'Training stopped and variables reset'}), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    if status == None:
        log = "Training Not Started/Stopped"
    elif status == 'Training':
        log = "Training!!"
    elif status == "Done":
        log = "Done!!"
    return jsonify({'status': log}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True)
