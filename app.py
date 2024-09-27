
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# Define the Flask app
app = Flask(__name__)

# Load the trained GNN model
class FraudGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FraudGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = FraudGCN(input_dim=2, hidden_dim=16, output_dim=2)
model.load_state_dict(torch.load('gnn_fraud_detection_model.pth'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()
    # Convert to tensor
    features = torch.tensor(data['features']).float()
    # Simulate inference (you should modify according to your actual feature inputs)
    prediction = model(features).argmax().item()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
