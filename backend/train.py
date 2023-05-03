import json
import time
import os
import numpy as np

#import json

#with open('hyperparameters.json', 'r') as f:
#    hyperparameters = json.load(f)

#learning_rate = hyperparameters['learningRate']

def train():
    
    if os.path.exists("training_data.json"):
        os.remove("training_data.json")

    # Simulate training by generating random loss values
    for epoch in range(1, 30):
        loss = 1- epoch / 10
        metric = {'epoch': epoch, 'train_loss': loss, 'val_loss':loss+np.random.randn(1)}
        with open('training_data.json', 'a') as f:
            f.write(json.dumps(metric) + '\n')
        time.sleep(1)

    # Write the stop word at the end of the file
    with open('training_data.json', 'a') as f:
        f.write('{"Training completed": true}\n')

if __name__ == '__main__':
    train()
