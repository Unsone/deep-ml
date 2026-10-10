import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        T = len(x)
        self.last_inputs = [x[t].reshape(-1, 1) for t in range(T)]
        self.last_hiddens = [np.zeros((self.hidden_size, 1))]   # h_0 = 0
        outputs = []
        for t in range(T):
            h = np.tanh(self.W_xh @ self.last_inputs[t]
                        + self.W_hh @ self.last_hiddens[-1] + self.b_h)
            y = self.W_hy @ h + self.b_y
            self.last_hiddens.append(h)
            outputs.append(y)
        self.last_outputs = outputs
        return np.array(outputs).reshape(T, -1)

    def backward(self, x, y, learning_rate):
        T = len(x)
        dW_xh = np.zeros_like(self.W_xh); dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h);   db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros((self.hidden_size, 1))
        for t in reversed(range(T)):
            dy = self.last_outputs[t] - y[t].reshape(-1, 1)       # dL/dy_t
            h, h_prev = self.last_hiddens[t + 1], self.last_hiddens[t]
            dW_hy += dy @ h.T
            db_y += dy
            dh = self.W_hy.T @ dy + dh_next                        # 当前输出 + 未来传回
            dz = dh * (1 - h ** 2)                                 # 过 tanh
            dW_xh += dz @ self.last_inputs[t].T
            dW_hh += dz @ h_prev.T
            db_h += dz
            dh_next = self.W_hh.T @ dz                             # 传给 t-1
        for p, g in ((self.W_xh, dW_xh), (self.W_hh, dW_hh), (self.W_hy, dW_hy),
                     (self.b_h, db_h), (self.b_y, db_y)):
            p -= learning_rate * g