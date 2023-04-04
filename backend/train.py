import matplotlib.pyplot as plt
import numpy as np
import random
import json
from time import sleep


if __name__=="__main__":
    for epoch in range(1000):
        # Create a JSON-formatted log
        loss = round(random.uniform(0.1, 1.0), 4)
        log = {
            "epoch": epoch,
            "loss": loss
        }

        # Print the JSON-formatted log
        sleep(1)
        print(json.dumps(log))