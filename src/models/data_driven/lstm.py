"""
LSTM (Long Short-Term Memory) - Recurrent neural network for hydrological time series prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict


class LSTMModel(nn.Module):
    """PyTorch LSTM model"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        output = self.fc(lstm_out[:, -1, :])
        
        return output


class LSTM:
    """
    LSTM model for discharge prediction
    """
    
    def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                 seq_length: int = 7, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32,
                 dropout: float = 0.2, random_state: int = 42):
        """
        Initialize LSTM model
        
        Args:
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            seq_length: Length of input sequences
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
            dropout: Dropout rate
            random_state: Random seed
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_fitted = False
        
        # Store normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """
        Create sequences for LSTM training
        
        Args:
            X: Input features [precipitation, temperature, pet]
            y: Target values (optional)
            
        Returns:
            X_seq, y_seq (if y provided) or X_seq (if y not provided)
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i + self.seq_length])
            if y is not None:
                y_seq.append(y[i + self.seq_length])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        else:
            return X_seq
    
    def normalize(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True):
        """
        Normalize features and target
        
        Args:
            X: Input features
            y: Target values (optional)
            fit: Whether to fit normalization parameters
            
        Returns:
            Normalized X and y
        """
        if fit:
            # Reshape for normalization (handle sequences)
            if len(X.shape) == 3:  # Sequences
                X_flat = X.reshape(-1, X.shape[-1])
                self.X_mean = np.mean(X_flat, axis=0)
                self.X_std = np.std(X_flat, axis=0)
            else:
                self.X_mean = np.mean(X, axis=0)
                self.X_std = np.std(X, axis=0)
            
            self.X_std[self.X_std == 0] = 1.0
            
            if y is not None:
                self.y_mean = np.mean(y)
                self.y_std = np.std(y)
                if self.y_std == 0:
                    self.y_std = 1.0
        
        X_norm = (X - self.X_mean) / self.X_std
        
        if y is not None:
            y_norm = (y - self.y_mean) / self.y_std
            return X_norm, y_norm
        else:
            return X_norm
    
    def denormalize(self, y_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize predictions
        
        Args:
            y_norm: Normalized predictions
            
        Returns:
            Original scale predictions
        """
        return y_norm * self.y_std + self.y_mean
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit LSTM model
        
        Args:
            X_train: Training features [precipitation, temperature, pet]
            y_train: Training target (discharge)
        """
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_train, y_train)
        
        # Normalize
        X_seq_norm, y_seq_norm = self.normalize(X_seq, y_seq, fit=True)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq_norm).to(self.device)
        y_tensor = torch.FloatTensor(y_seq_norm).reshape(-1, 1).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Optional: print progress
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                # print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict discharge
        
        Args:
            X: Features [precipitation, temperature, pet]
            
        Returns:
            Predicted discharge
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Create sequences
        X_seq = self.create_sequences(X)
        
        # Normalize
        X_seq_norm = self.normalize(X_seq, fit=False)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq_norm).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            y_pred_norm = self.model(X_tensor).cpu().numpy().flatten()
        
        # Denormalize
        y_pred = self.denormalize(y_pred_norm)
        
        # Pad predictions to match original length
        y_pred_full = np.zeros(len(X))
        y_pred_full[self.seq_length:] = y_pred
        
        # Ensure non-negative discharge
        y_pred_full = np.maximum(0, y_pred_full)
        
        return y_pred_full
    
    def calibrate(self, X_train: np.ndarray, y_train: np.ndarray,
                  n_iterations: int = 5) -> Dict[str, float]:
        """
        Calibrate LSTM model by trying different hyperparameters
        
        Args:
            X_train: Training features
            y_train: Training target
            n_iterations: Number of calibration iterations (not used, grid search)
            
        Returns:
            Best parameters
        """
        from ..utils.metrics import nash_sutcliffe_efficiency
        
        best_nse = -np.inf
        best_params = None
        
        # Grid search over hyperparameters (limited due to training time)
        for hidden in [32, 64]:
            for layers in [1, 2]:
                for seq_len in [5, 7]:
                    self.hidden_size = hidden
                    self.num_layers = layers
                    self.seq_length = seq_len
                    
                    # Fit
                    self.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = self.predict(X_train)
                    
                    # Evaluate (skip first seq_length values)
                    nse = nash_sutcliffe_efficiency(
                        y_train[self.seq_length:], 
                        y_pred[self.seq_length:]
                    )
                    
                    if nse > best_nse:
                        best_nse = nse
                        best_params = {
                            'hidden_size': hidden,
                            'num_layers': layers,
                            'seq_length': seq_len,
                            'NSE': nse
                        }
        
        # Set best parameters and refit
        if best_params:
            self.hidden_size = best_params['hidden_size']
            self.num_layers = best_params['num_layers']
            self.seq_length = best_params['seq_length']
            self.fit(X_train, y_train)
        
        return best_params
