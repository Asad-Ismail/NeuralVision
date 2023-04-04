from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import subprocess
import threading
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

""""This will update training data"""
class TrainingDataHandler(FileSystemEventHandler):
    def __init__(self, socketio):
        self.socketio = socketio
        self.stop_word = 'Training completed'
        self.metrics = []

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            with open(event.src_path, 'r') as f:
                for line in f:
                    if line.strip() == self.stop_word:
                        # Training is completed, stop reading the file
                        break
                    try:
                        metric = json.loads(line.strip())
                        if metric not in self.metrics:
                            self.metrics.append(metric)
                            self.socketio.emit('metric', metric)
                    except ValueError:
                        pass

def read_metrics_from_file(socketio):
    event_handler = TrainingDataHandler(socketio)
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(.1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def start_training_thread():
    process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    process.communicate()

@app.route('/api/start_training', methods=['GET'])
def start_training():
    training_thread = threading.Thread(target=start_training_thread)
    training_thread.start()
    # read logging
    read_metrics_from_file()

    return jsonify({'message': 'Training started'}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True)
