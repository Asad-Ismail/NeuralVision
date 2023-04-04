import matplotlib.pyplot as plt
import numpy as np
import random
import json


if __name__=="__main__":
    for epoch in range(1000):
        # Create a JSON-formatted log
        loss = round(random.uniform(0.1, 1.0), 4)
        log = {
            "epoch": epoch,
            "loss": loss
        }

        # Print the JSON-formatted log
        print(json.dumps(log))