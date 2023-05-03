#import eventlet
#eventlet.monkey_patch()
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
import signal
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['UPLOAD_FOLDER'] = 'static/uploads'  
# Create the directory and its parent directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



class NoAccessLogFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.DEBUG

class CustomLogFilter(logging.Filter):
    def filter(self, record):
        # Only allow messages that do NOT contain the specific message format
        return not (record.levelname == 'INFO' and 'GET /api/status HTTP' in record.msg)

log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)
log.addFilter(NoAccessLogFilter())
log.addFilter(CustomLogFilter())

status = None
process = None
training_thread =None
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
    global process
    process = subprocess.Popen(['python', 'train.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    process.communicate()


@app.route('/api/newproject', methods=['POST'])
def new_project():
    project_name = request.json['name']
    logging.debug(f"Received Project Name"*20)
    logging.debug(project_name)
    return jsonify({'status': 'success'})

def count_images_in_directory(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    return sum(1 for file in os.listdir(directory) if file.lower().endswith(image_extensions))

@app.route('/api/ssl_uploaddata', methods=['POST'])
def ssl_upload():
    data = request.get_json()

    if 'dataPath' not in data:
        return jsonify({'error': 'dataPath is missing'}), 400

    data_path = data['dataPath']
    
    logging.debug(f"Received Data path: {data_path}")

    if not os.path.isdir(data_path):
        return jsonify({'error': 'Invalid dataPath'}), 400

    image_count = count_images_in_directory(data_path)
    
    return jsonify({'trainImagesCount': image_count})
    
    
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
        logging.info("Training is already started!!")
        return jsonify({'message': 'Training is already running'}), 200

    logging.info("Trying to start training!!")
    training_thread = threading.Thread(target=start_training_thread)
    training_thread.start()
    read_metrics_from_file(socketio)
    status = "Training"
    logging.info("Training started!!")
    return jsonify({'message': 'Training started'}), 200


def stop_training_thread():
    global process
    if process:
        os.kill(process.pid, signal.SIGTERM)
        process = None

@app.route('/api/stop_training', methods=['GET'])
def stop_training():
    logging.debug(f"Trying to end the training ")
    global status, training_thread, observer
    if training_thread and training_thread.is_alive():
        logging.debug(f"Killing Training Process!! ")
        stop_training_thread()  # Called stop_training_thread() here
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
    app.debug = True
    socketio.run(app, debug=True)


