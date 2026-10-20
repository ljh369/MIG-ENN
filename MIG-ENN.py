import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MINELayer(nn.Module):
    def __init__(self, h_mine):
        super(MINELayer, self).__init__()
        self.fc1 = nn.Linear(1, h_mine)
        self.fc2 = nn.Linear(1, h_mine)
        self.fc3 = nn.Linear(h_mine, 1)

    def forward(self, x, y):
        h1 = F.relu(self.fc1(x) + self.fc2(y))
        return self.fc3(h1)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.att = nn.Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.att)

    def forward(self, h, adj):
        Wh = self.fc(h)
        N = Wh.size(0)

        Wh_rep = Wh.repeat(1, N).view(N * N, -1)
        Wh_rep_T = Wh.repeat(N, 1)
        a_input = torch.cat([Wh_rep, Wh_rep_T], dim=1)

        e = F.leaky_relu(torch.matmul(a_input, self.att).view(N, N))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        return torch.matmul(attention, Wh)


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim

        self.Wq = nn.Parameter(torch.Tensor(mem_dim, input_size))
        self.Wk = nn.Parameter(torch.Tensor(mem_dim, input_size))
        self.Wv = nn.Parameter(torch.Tensor(mem_dim, input_size))
        self.Wo = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.Wc = nn.Parameter(torch.Tensor(hidden_size, mem_dim))

        self.wi = nn.Parameter(torch.Tensor(1, input_size))
        self.wf = nn.Parameter(torch.Tensor(1, input_size))

        self.bq = nn.Parameter(torch.Tensor(mem_dim))
        self.bk = nn.Parameter(torch.Tensor(mem_dim))
        self.bv = nn.Parameter(torch.Tensor(mem_dim))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))
        self.bi = nn.Parameter(torch.Tensor(1))
        self.bf = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, states):
        B, T, F = x.shape
        C_prev, n_prev = states
        outputs = []

        for t in range(T):
            xt = x[:, t, :]

            qt = torch.matmul(xt, self.Wq.T) + self.bq
            kt = torch.matmul(xt, self.Wk.T) + self.bk
            vt = torch.matmul(xt, self.Wv.T) + self.bv

            it = torch.sigmoid(torch.matmul(xt, self.wi.T) + self.bi)
            ft = torch.sigmoid(torch.matmul(xt, self.wf.T) + self.bf)

            outer_product = vt.unsqueeze(2) * kt.unsqueeze(1)
            C = ft.unsqueeze(2) * C_prev + it.unsqueeze(2) * outer_product
            n = ft * n_prev + it * kt.unsqueeze(2)

            h_tilde = torch.matmul(C, qt.unsqueeze(2)).squeeze(2)
            h_proj = torch.matmul(h_tilde, self.Wc.T)

            ot = torch.sigmoid(torch.matmul(xt, self.Wo.T) + self.bo)
            ht = ot * h_proj
            outputs.append(ht.unsqueeze(1))

            C_prev = C
            n_prev = n

        outputs = torch.cat(outputs, dim=1)
        return outputs, (C, n)

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.mem_dim, self.mem_dim),
                torch.zeros(batch_size, self.mem_dim, 1))


class EEMIPF(nn.Module):
    def __init__(self, h_mine, gat_in_features, gat_out_features, lstm_hidden_size, mem_dim=32):
        super(EEMIPF, self).__init__()
        self.MINELayer = MINELayer(h_mine)
        self.GATLayer = GATLayer(gat_in_features, gat_out_features)
        self.MLSTMLayer = mLSTM(gat_out_features, lstm_hidden_size, mem_dim)
        self.FCLayer = nn.Linear(lstm_hidden_size, 1)

    def forward(self, X):
        N, T = X.shape

        X_i = X.unsqueeze(1).expand(N, N, T).reshape(-1, 1)
        X_j = X.unsqueeze(0).expand(N, N, T).reshape(-1, 1)

        pos_scores = self.MINELayer(X_i, X_j).view(N * N, T).mean(dim=1).view(N, N)

        perm = torch.randperm(N * N * T, device=X.device)
        X_j_shuffle = X_j[perm]
        neg_scores = self.MINELayer(X_i, X_j_shuffle).view(N * N, T)

        neg_mean_exp = torch.exp(neg_scores).mean(dim=1).view(N, N)
        adj = pos_scores - torch.log(neg_mean_exp + 1e-8)

        GAT_out_list = []
        for t in range(T):
            x_gat = X[:, t].unsqueeze(1)
            gat_output = self.GATLayer(x_gat, adj)
            GAT_out_list.append(gat_output.unsqueeze(1))

        gat_outputs = torch.cat(GAT_out_list, dim=1)

        states = self.MLSTMLayer.init_hidden(N)
        states = tuple(tensor.to(X.device) for tensor in states)
        mlstm_output, _ = self.MLSTMLayer(gat_outputs, states)

        mlstm_out_last = mlstm_output[:, -1, :]
        out = self.FCLayer(mlstm_out_last)

        return out, adj


def sliding_window(matrix, window_size):
    n_rows, n_cols = matrix.shape
    num_windows = n_cols - window_size
    sub_matrices, labels = [], []
    for i in range(num_windows):
        sub_matrices.append(matrix[:, i:i + window_size])
        labels.append(matrix[:, i + window_size:i + window_size + 1])
    return np.array(sub_matrices), np.array(labels)


def split_data(data_Xs, data_Ys, train_ratio=0.8, val_ratio=0.1):
    total_samples = data_Xs.shape[0]
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    X_train, Y_train = data_Xs[:train_size], data_Ys[:train_size]
    X_val, Y_val = data_Xs[train_size:train_size + val_size], data_Ys[train_size:train_size + val_size]
    X_test, Y_test = data_Xs[train_size + val_size:], data_Ys[train_size + val_size:]
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


if __name__ == "__main__":
    file_path = '../Data/PV.csv'
    try:
        data = pd.read_csv(file_path, header=None)
        data_matrix = data.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized_matrix = scaler.fit_transform(data_matrix)
        noise = np.random.normal(0, 0.1, normalized_matrix.shape)
        normalized_matrix = normalized_matrix + noise
        window_size = 240
        data_Xs, data_Ys = sliding_window(normalized_matrix, window_size)
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(data_Xs, data_Ys)
    except Exception as e:
        print(f"数据加载失败: {e}, 使用随机数据演示")
        N_nodes, T_total = 137, 500
        normalized_matrix = np.random.randn(N_nodes, T_total)
        data_Xs, data_Ys = sliding_window(normalized_matrix, 240)
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = split_data(data_Xs, data_Ys)

    h_mine = 64
    gat_in_features = 1
    gat_out_features = 8
    lstm_hidden_size = 32
    mem_dim = 32
    lambda_l1 = 0.001
    lambda_mine = 0.1
    lambda_sparse = 0.01
    num_epochs = 50
    patience = 5

    EEMIPF_model = EEMIPF(h_mine, gat_in_features, gat_out_features, lstm_hidden_size, mem_dim).to(device)
    optimizer = torch.optim.Adam(EEMIPF_model.parameters(), lr=0.001, weight_decay=0.001)

    best_val_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        EEMIPF_model.train()
        total_train_loss = 0.0

        for x_train, y_train in zip(X_train, Y_train):
            optimizer.zero_grad()

            x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)

            output_pred, output_adj = EEMIPF_model(x_tensor)

            mse_loss = F.mse_loss(output_pred, y_tensor)
            mine_loss = - output_adj.mean()
            adj_sparse_loss = output_adj.abs().sum()
            l1_reg = sum(p.norm(1) for p in EEMIPF_model.parameters())

            loss = mse_loss + lambda_mine * mine_loss + lambda_sparse * adj_sparse_loss + lambda_l1 * l1_reg

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(X_train)

        EEMIPF_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in zip(X_val, Y_val):
                x_tensor = torch.tensor(x_val, dtype=torch.float32, device=device)
                y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).view(-1, 1)

                output_pred, output_adj = EEMIPF_model(x_tensor)

                mse_loss = F.mse_loss(output_pred, y_tensor)
                mine_loss = - output_adj.mean()
                adj_sparse_loss = output_adj.abs().sum()
                l1_reg = sum(p.norm(1) for p in EEMIPF_model.parameters())

                loss = mse_loss + lambda_mine * mine_loss + lambda_sparse * adj_sparse_loss + lambda_l1 * l1_reg
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(X_val)
        print(f'Epoch: {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(EEMIPF_model.state_dict(), 'best_EEMIPF_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

    print("\n--- 开始测试 ---")
    EEMIPF_model.load_state_dict(torch.load('best_EEMIPF_model.pth'))
    EEMIPF_model.eval()

    mean_mse, mean_mae = 0.0, 0.0
    with torch.no_grad():
        for x_test, y_test in zip(X_test, Y_test):
            x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
            y_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)

            output_pred, _ = EEMIPF_model(x_tensor)

            mse = F.mse_loss(output_pred, y_tensor).item()
            mae = F.l1_loss(output_pred, y_tensor).item()
            mean_mse += mse
            mean_mae += mae

    print(f"Test MSE: {mean_mse / len(X_test):.6f}")
    print(f"Test MAE: {mean_mae / len(X_test):.6f}")
    print("训练与测试完成！")