import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.
    """
    # Convert inputs to PyTorch tensors
    X = torch.tensor(features, dtype=torch.float32)
    y_true = torch.tensor(labels, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    b = torch.tensor(bias, dtype=torch.float32)
    
    # Compute the linear combination: z = Xw + b
    # X is shape (num_samples, num_features), w is shape (num_features,)
    z = torch.matmul(X, w) + b
    
    # Apply sigmoid activation to get probabilities
    probabilities = torch.sigmoid(z)
    
    # Compute Mean Squared Error (MSE)
    mse = torch.mean((probabilities - y_true) ** 2)
    
    # Round the outputs to 4 decimal places and convert to lists/floats
    rounded_probabilities = [round(p, 4) for p in probabilities.tolist()]
    rounded_mse = round(mse.item(), 4)
    
    return rounded_probabilities, rounded_mse