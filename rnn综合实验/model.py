import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0  # 单层不dropout
        )
        # 增强特征融合：拼接最后一个时间步和全局均值
        self.fc1 = nn.Linear(2 * hidden_size * 2, hidden_size * 2)  # 2*hidden_size（双向） * 2（最后一步+均值）
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, max_len, input_size)
        outputs, h_n = self.rnn(x)  # outputs: (batch_size, max_len, 2*hidden_size)

        # 特征融合：最后一个时间步 + 全局均值
        last_step = outputs[:, -1, :]  # (batch_size, 2*hidden_size)
        global_avg = outputs.mean(dim=1)  # (batch_size, 2*hidden_size)
        combined = torch.cat([last_step, global_avg], dim=1)  # (batch_size, 4*hidden_size)

        # 全连接层
        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.fc2(out)
        return torch.log_softmax(out, dim=-1)


class CharGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(2 * hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs, h_n = self.gru(x)

        last_step = outputs[:, -1, :]
        global_avg = outputs.mean(dim=1)
        combined = torch.cat([last_step, global_avg], dim=1)

        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.fc2(out)
        return torch.log_softmax(out, dim=-1)


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.3):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(2 * hidden_size * 2, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        outputs, (h_n, c_n) = self.lstm(x)

        last_step = outputs[:, -1, :]
        global_avg = outputs.mean(dim=1)
        combined = torch.cat([last_step, global_avg], dim=1)

        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.fc2(out)
        return torch.log_softmax(out, dim=-1)