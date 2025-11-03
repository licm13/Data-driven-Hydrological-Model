"""
LSTM (Long Short-Term Memory) - 长短期记忆网络
最先进的序列建模方法，用于径流预测
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional, List
from .base_model import BaseHydrologicalModel

class LSTMModel(nn.Module):
    """PyTorch LSTM模型"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout_rate: float = 0.4):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_size, 1)
        
        # 初始化forget gate bias为1（有助于学习长期依赖）
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.0)
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq_len, input_size)
        """
        lstm_out, hidden = self.lstm(x, hidden)
        
        # 取最后一个时间步
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout(lstm_out)
        
        # 输出
        out = self.fc(lstm_out)
        
        return out.squeeze(), hidden


class LSTM(BaseHydrologicalModel):
    """
    LSTM径流预测模型
    
    特点：
    - 自动学习时间依赖关系
    - 无需手动创建lag特征
    - 可以处理长期依赖
    """
    
    def __init__(self, 
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 sequence_length: int = 365,
                 dropout_rate: float = 0.4,
                 learning_rate: float = 0.01,
                 n_epochs: int = 20,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 n_init: int = 3):
        """
        Parameters:
        -----------
        hidden_size : int, LSTM隐藏单元数
        num_layers : int, LSTM层数
        sequence_length : int, 输入序列长度
        dropout_rate : float, Dropout比例
        learning_rate : float, 学习率
        n_epochs : int, 训练轮数
        batch_size : int, 批次大小
        device : str, 计算设备
        n_init : int, 随机初始化次数（集成）
        """
        super().__init__("LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.n_init = n_init
        self.models = []  # 存储多个初始化的模型
        self.scaler_X = None
        self.scaler_y = None
        
    def initialize(self, params: Dict = None) -> None:
        """初始化LSTM"""
        if params:
            self.learning_rate = params.get('learning_rate', self.learning_rate)
            self.n_epochs = params.get('n_epochs', self.n_epochs)
            self.hidden_size = params.get('hidden_size', self.hidden_size)
        
        self.parameters = {
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'hidden_size': self.hidden_size,
            'sequence_length': self.sequence_length,
        }
        self.states = {}
    
    def _create_sequences(self, 
                         precip: np.ndarray, 
                         temp: np.ndarray, 
                         pet: np.ndarray,
                         discharge: Optional[np.ndarray] = None) -> Tuple:
        """
        创建LSTM序列
        
        对于每个时间步t，创建序列：
        X[i] = [[P[t-seq:t], T[t-seq:t], PET[t-seq:t]]]
        y[i] = Q[t]
        
        Returns:
        --------
        X_seq : array, shape (n_samples, seq_len, 3)
        y_seq : array, shape (n_samples,) or None
        """
        n = len(precip)
        n_samples = n - self.sequence_length
        
        # 输入特征：3个变量
        X_seq = np.zeros((n_samples, self.sequence_length, 3))
        
        for i in range(n_samples):
            start = i
            end = i + self.sequence_length
            
            X_seq[i, :, 0] = precip[start:end]
            X_seq[i, :, 1] = temp[start:end]
            X_seq[i, :, 2] = pet[start:end]
        
        if discharge is not None:
            # 目标：序列后的径流值
            y_seq = discharge[self.sequence_length:]
            return X_seq, y_seq
        else:
            return X_seq, None
    
    def _normalize_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                       fit: bool = True) -> Tuple:
        """标准化数据"""
        from sklearn.preprocessing import StandardScaler
        
        # X是3D: (n_samples, seq_len, n_features)
        n_samples, seq_len, n_features = X.shape
        
        if fit:
            self.scaler_X = StandardScaler()
            # 重塑为2D进行标准化
            X_2d = X.reshape(-1, n_features)
            X_norm_2d = self.scaler_X.fit_transform(X_2d)
            X_norm = X_norm_2d.reshape(n_samples, seq_len, n_features)
            
            if y is not None:
                self.scaler_y = StandardScaler()
                y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = None
        else:
            X_2d = X.reshape(-1, n_features)
            X_norm_2d = self.scaler_X.transform(X_2d)
            X_norm = X_norm_2d.reshape(n_samples, seq_len, n_features)
            
            if y is not None:
                y_norm = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            else:
                y_norm = None
        
        return X_norm, y_norm
    
    def train(self, 
              precip: np.ndarray, 
              temp: np.ndarray, 
              pet: np.ndarray,
              discharge: np.ndarray,
              val_split: float = 0.1) -> List[Dict]:
        """
        训练LSTM（多次初始化集成）
        
        Parameters:
        -----------
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        discharge : array, 径流
        val_split : float, 验证集比例
        
        Returns:
        --------
        histories : list of dict, 每次初始化的训练历史
        """
        # 创建序列
        X, y = self._create_sequences(precip, temp, pet, discharge)
        
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
        
        # 多次初始化训练
        histories = []
        self.models = []
        
        for init_idx in range(self.n_init):
            print(f"\nTraining LSTM initialization {init_idx + 1}/{self.n_init}")
            
            # 创建新模型
            model = LSTMModel(
                input_size=3,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            # 优化器
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()
            
            # 学习率衰减
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            
            # 训练
            history = {'train_loss': [], 'val_loss': []}
            
            for epoch in range(self.n_epochs):
                model.train()
                
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
                    outputs, _ = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                
                # 验证
                model.eval()
                with torch.no_grad():
                    val_outputs, _ = model(X_val_t)
                    val_loss = criterion(val_outputs, y_val_t)
                
                history['train_loss'].append(np.mean(train_losses))
                history['val_loss'].append(val_loss.item())
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, "
                          f"Train Loss: {history['train_loss'][-1]:.4f}, "
                          f"Val Loss: {history['val_loss'][-1]:.4f}")
                
                scheduler.step()
            
            # 保存模型
            self.models.append(model)
            histories.append(history)
        
        return histories
    
    def predict(self, 
                precip: np.ndarray, 
                temp: np.ndarray, 
                pet: np.ndarray,
                return_std: bool = False) -> np.ndarray:
        """
        预测径流（集成多个模型的平均）
        
        Parameters:
        -----------
        precip : array, 降水
        temp : array, 温度
        pet : array, 蒸发
        return_std : bool, 是否返回标准差
        
        Returns:
        --------
        discharge : array, 预测径流
        std : array (可选), 预测标准差
        """
        if len(self.models) == 0:
            raise ValueError("Model not trained. Call train() first.")
        
        # 创建序列
        X, _ = self._create_sequences(precip, temp, pet)
        
        # 标准化
        X_norm, _ = self._normalize_data(X, fit=False)
        
        # 转换为张量
        X_t = torch.FloatTensor(X_norm).to(self.device)
        
        # 集成预测
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                y_pred_norm, _ = model(X_t)
                y_pred_norm = y_pred_norm.cpu().numpy()
                
                # 反标准化
                y_pred = self.scaler_y.inverse_transform(
                    y_pred_norm.reshape(-1, 1)
                ).flatten()
                
                predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # 平均
        discharge_pred = np.mean(predictions, axis=0)
        
        # 补齐前面的序列长度
        discharge = np.zeros(len(precip))
        discharge[self.sequence_length:] = discharge_pred
        
        # 确保非负
        discharge = np.maximum(discharge, 0)
        
        if return_std:
            std_pred = np.std(predictions, axis=0)
            std = np.zeros(len(precip))
            std[self.sequence_length:] = std_pred
            return discharge, std
        else:
            return discharge
    
    def run_timestep(self, 
                     precip: float, 
                     temp: float, 
                     pet: float,
                     timestep: int) -> float:
        """单步预测不适用于LSTM"""
        raise NotImplementedError(
            "LSTM requires full time series. Use predict() method."
        )
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """返回超参数范围"""
        return {
            'learning_rate': (0.001, 0.1),
            'n_epochs': (10, 50),
            'hidden_size': (32, 256),
        }
    
    def save_models(self, path_prefix: str) -> None:
        """保存所有模型"""
        for i, model in enumerate(self.models):
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X': self.scaler_X,
                'scaler_y': self.scaler_y,
                'config': {
                    'hidden_size': self.hidden_size,
                    'num_layers': self.num_layers,
                    'sequence_length': self.sequence_length,
                    'dropout_rate': self.dropout_rate,
                }
            }, f"{path_prefix}_init{i}.pt")
    
    def load_models(self, path_prefix: str, n_models: int = None) -> None:
        """加载模型"""
        if n_models is None:
            n_models = self.n_init
        
        self.models = []
        
        for i in range(n_models):
            path = f"{path_prefix}_init{i}.pt"
            checkpoint = torch.load(path, map_location=self.device)
            
            if i == 0:
                # 从第一个模型加载配置
                config = checkpoint['config']
                self.hidden_size = config['hidden_size']
                self.num_layers = config['num_layers']
                self.sequence_length = config['sequence_length']
                self.dropout_rate = config['dropout_rate']
                self.scaler_X = checkpoint['scaler_X']
                self.scaler_y = checkpoint['scaler_y']
            
            # 重建模型
            model = LSTMModel(
                input_size=3,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            self.models.append(model)