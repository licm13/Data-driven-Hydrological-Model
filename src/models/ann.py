"""
ANN (Artificial Neural Network) - 人工神经网络
前馈神经网络用于径流预测
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
from .base_model import BaseHydrologicalModel

class ANNModel(nn.Module):
    """PyTorch神经网络模型"""
    
    def __init__(self, 
                 input_size: int, 
                 hidden_sizes: list = [64, 64, 64],
                 dropout_rate: float = 0.4):
        super(ANNModel, self).__init__()
        
        layers = []
        in_features = input_size
        
        # 隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_size
        
        # 输出层
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ANN(BaseHydrologicalModel):
    """
    人工神经网络径流预测模型
    
    使用时间窗口lag创建输入特征
    """
    
    def __init__(self, 
                 hidden_sizes: list = [64, 64, 64],
                 time_lag: int = 7,
                 dropout_rate: float = 0.4,
                 learning_rate: float = 0.001,
                 n_epochs: int = 30,
                 batch_size: int = 32,
                 device: str = 'cpu'):
        """
        Parameters:
        -----------
        hidden_sizes : list, 每层隐藏单元数
        time_lag : int, 时间滞后窗口
        dropout_rate : float, Dropout比例
        learning_rate : float, 学习率
        n_epochs : int, 训练轮数
        batch_size : int, 批次大小
        device : str, 计算设备
        """
        super().__init__("ANN")
        self.hidden_sizes = hidden_sizes
        self.time_lag = time_lag
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def initialize(self, params: Dict = None) -> None:
        """初始化ANN"""
        if params:
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.n_epochs = params.get('n_epochs', self.n_epochs)
        
        self.parameters = {
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'hidden_sizes': self.hidden_sizes,
        }
        self.states = {}
    
    def _create_lagged_features(self, 
                               precip: np.ndarray, 
                               temp: np.ndarray, 
                               pet: np.ndarray) -> np.ndarray:
        """
        创建时间滞后特征
        
        对于每个时间步t，创建特征：
        [precip[t-lag:t+1], temp[t-lag:t+1], pet[t-lag:t+1]]
        
        Returns:
        --------
        features : array, shape (n - lag, (lag+1) * 3)
        """
        n = len(precip)
        n_features = (self.time_lag + 1) * 3
        
        # 有效样本数
        n_valid = n - self.time_lag
        features = np.zeros((n_valid, n_features))
        
        for i in range(n_valid):
            idx = i + self.time_lag
            # 降水滞后
            features[i, :self.time_lag+1] = precip[idx-self.time_lag:idx+1]
            # 温度滞后
            features[i, self.time_lag+1:2*(self.time_lag+1)] = temp[idx-self.time_lag:idx+1]
            # 蒸发滞后
            features[i, 2*(self.time_lag+1):] = pet[idx-self.time_lag:idx+1]
        
        return features
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray, 
                       fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """标准化数据"""
        from sklearn.preprocessing import StandardScaler
        
        if fit:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            X_norm = self.scaler_X.fit_transform(X)
            y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            X_norm = self.scaler_X.transform(X)
            y_norm = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        return X_norm, y_norm
    
    def train(self, 
              precip: np.ndarray, 
              temp: np.ndarray, 
              pet: np.ndarray,
              discharge: np.ndarray,
              val_split: float = 0.1) -> Dict[str, list]:
        """
        训练ANN
        
        Parameters:
        -----------
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        discharge : array, 径流
        val_split : float, 验证集比例
        
        Returns:
        --------
        history : dict, 训练历史
        """
        # 创建特征
        X = self._create_lagged_features(precip, temp, pet)
        y = discharge[self.time_lag:]  # 对齐
        
        # 标准化
        X_norm, y_norm = self._normalize_data(X, y, fit=True)
        
        # 划分训练/验证集
        n_train = int(len(X) * (1 - val_split))
        X_train, y_train = X_norm[:n_train], y_norm[:n_train]
        X_val, y_val = X_norm[n_train:], y_norm[n_train:]
        
        # 转换为PyTorch张量
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        # 创建模型
        input_size = X_train.shape[1]
        self.model = ANNModel(
            input_size, 
            self.hidden_sizes, 
            self.dropout_rate
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        # 学习率衰减
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # 训练
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"Training ANN for {self.n_epochs} epochs...")
        for epoch in range(self.n_epochs):
            self.model.train()
            
            # Mini-batch训练
            n_batches = len(X_train) // self.batch_size
            train_losses = []
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                
                X_batch = X_train_t[start_idx:end_idx]
                y_batch = y_train_t[start_idx:end_idx]
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t)
            
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss.item())
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, "
                      f"Train Loss: {history['train_loss'][-1]:.4f}, "
                      f"Val Loss: {history['val_loss'][-1]:.4f}")
            
            scheduler.step()
        
        return history
    
    def predict(self, 
                precip: np.ndarray, 
                temp: np.ndarray, 
                pet: np.ndarray) -> np.ndarray:
        """
        预测径流
        
        Returns:
        --------
        discharge : array, 预测径流
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # 创建特征
        X = self._create_lagged_features(precip, temp, pet)
        
        # 标准化
        X_norm, _ = self._normalize_data(X, np.zeros(len(X)), fit=False)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X_norm).to(self.device)
            y_pred_norm = self.model(X_t).squeeze().cpu().numpy()
        
        # 反标准化
        y_pred = self.scaler_y.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        
        # 补齐前面的时间步（用0或平均值填充）
        discharge = np.zeros(len(precip))
        discharge[self.time_lag:] = y_pred
        
        # 确保非负
        discharge = np.maximum(discharge, 0)
        
        return discharge
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """单步预测不适用于ANN"""
        raise NotImplementedError(
            "ANN requires full time series. Use predict() method."
        )
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回超参数范围"""
        return {
            'learning_rate': (0.0001, 0.01),
            'n_epochs': (10, 100),
        }
    
    def save_model(self, path: str) -> None:
        """保存模型"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': {
                    'hidden_sizes': self.hidden_sizes,
                    'time_lag': self.time_lag,
                    'dropout_rate': self.dropout_rate,
                }
            }, path)
    
    def load_model(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        config = checkpoint['config']
        self.hidden_sizes = config['hidden_sizes']
        self.time_lag = config['time_lag']
        self.dropout_rate = config['dropout_rate']
        
        # 重建模型
        input_size = (self.time_lag + 1) * 3
        self.model = ANNModel(
            input_size, 
            self.hidden_sizes, 
            self.dropout_rate
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']