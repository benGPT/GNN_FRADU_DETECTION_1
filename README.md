
# Fraud Detection using Graph Neural Networks (GNN)

## Overview
This project utilizes Graph Neural Networks (GNN) to detect fraudulent transactions in a credit card dataset. Each transaction is represented as a node, and relationships between users form edges in the network.

### Features:
- Data preprocessing from CSV to graph format.
- Implementation of Graph Convolutional Network (GCN) for fraud detection.
- Model saving and evaluation included.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NetworkX
- Pandas

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/fraud-detection-gnn.git
   cd fraud-detection-gnn
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to train and save the model:
   ```
   jupyter notebook rephrased_fraud_detection_with_gnn.ipynb
   ```

4. For deployment, use Flask API or Streamlit UI.

## Deployment Instructions (Flask)

To deploy this project using Docker and Flask:

1. Build the Docker image:
   ```
   docker build -t fraud-detection-gnn .
   ```

2. Run the Docker container:
   ```
   docker run -p 5000:5000 fraud-detection-gnn
   ```

