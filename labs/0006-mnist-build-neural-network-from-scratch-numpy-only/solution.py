# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr
        # 1. 初始化权重 (加入了 He 初始化，防止梯度消失/爆炸)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
    
    def forward(self, X):
        # 1. 第一层：线性 + ReLU
        self.z1 = X @ self.W1 + self.b1
        self.hidden = np.maximum(0, self.z1)
        
        # 2. 第二层：线性 + Softmax
        self.scores = self.hidden @ self.W2 + self.b2
        scores_max = np.max(self.scores, axis=1, keepdims=True)
        exp_scores = np.exp(self.scores - scores_max)
        self.probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        return self.probs
    
    def backward(self, X, y, probs):
        N = X.shape[0]
        
        # 1. 计算交叉熵损失
        loss = -np.sum(np.log(probs[np.arange(N), y] + 1e-8)) / N
        
        # 2. 计算 loss 对 scores 的梯度 (Softmax + Cross-Entropy)
        dL_dscores = probs.copy()
        dL_dscores[np.arange(N), y] -= 1  # 修复了语法错误
        dL_dscores = dL_dscores / N
        
        # 3. 反向传播到第二层 (W2, b2)
        dL_dW2 = self.hidden.T @ dL_dscores
        dL_db2 = np.sum(dL_dscores, axis=0)  # 去掉 keepdims，保证形状为 (10,)
        
        # 4. 反向传播到隐藏层并经过 ReLU
        dL_dhidden = dL_dscores @ self.W2.T
        dL_dz1 = dL_dhidden * (self.z1 > 0)
        
        # 5. 反向传播到第一层 (W1, b1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0)  # 修复了变量名 dz1 -> dL_dz1，去掉 keepdims
        
        # 6. 更新所有参数
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        
        return loss
    
    def train_step(self, X, y):
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)