import json
import sys
import math
import requests
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    # Parse provided spec into a dict
    spec = json.loads(sys.stdin.read())
    evaluate(spec)

def evaluate(spec):
    
    data = []
    for x in range(8):
        offset = 1 + (x * 60)
        params = {'query': f'sum(rate(container_cpu_usage_seconds_total{{image="docker.io/descartesresearch/teastore-webui:latest"}}[150s]offset {offset}s ))*100'}
        response = requests.get('http://34.73.51.51:32662/api/v1/query', params=params)
        cpu_util = response.json()
        cpu_util = cpu_util.get('data').get('result')[0].get('value')[1]
        cpu_util = float(cpu_util)
        data.append(cpu_util)
    data = [data]
    scaler = joblib.load('sc.joblib')
    data = scaler.transform(data)
    data = [data[0][1:]]
    data = pd.DataFrame(data)
    from copy import deepcopy as dc
   



    data = data.to_numpy()

    data = dc(np.flip(data,axis=1))
    data = data.reshape((-1, 7, 1))
    data = torch.tensor(data).float()

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    test_predictions = loaded_model(data.to(device)).detach().cpu().numpy().flatten()


    dummies = np.zeros((data.shape[0], 8))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])
    
    evaluation = {}
    evaluation["targetReplicas"] = 1 + (math.floor(test_predictions/12.5))

    # Output JSON to stdout
    sys.stdout.write(json.dumps(evaluation))

if __name__ == "__main__":
    main()