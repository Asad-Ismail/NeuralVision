from flask import Flask, request

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    files = request.files.getlist('files')
    # Process the selected images and train the model using PyTorch Lightning
    return 'Training successful'
