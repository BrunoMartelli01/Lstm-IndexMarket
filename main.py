import numpy as np
import pandas as pd
from pylab import plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
def load_data(stock, look_back, look_forward, batchsize =1):
    data_raw = stock.values  # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)

    test_set_size = int(np.round(0.25 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train_temp = data[:train_set_size, :-look_forward, :]
    y_train_temp = data[:train_set_size, -look_forward:, :]

    x_validation = data[train_set_size+2*look_back :-1, :-look_forward, :]
    y_validation = data[train_set_size+2*look_back :-1, -look_forward:, :]

    x_test = data[-1, :-look_forward, :]
    y_test = data[-1, -look_forward:, :]

    x_train = []
    y_train = []

    for i in range(0,  len(x_train_temp), batchsize):
        x_train.append(x_train_temp[i:i+batchsize,:,:])
        y_train.append(y_train_temp[i:i+batchsize,:,:])

    x_train =x_train[:-1]
    y_train =y_train[:-1]
    return [np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_validation), np.array(y_validation)]


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).cuda()

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim).cuda()

    def forward(self, x):
        cuda0 = torch.device(0)
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(cuda0).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(cuda0).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()),)

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


if __name__ == "__main__":

    dates = pd.date_range('2018-01-04', '2023-10-01')
    df1 = pd.DataFrame(index=dates)
    df_stockIndex = pd.read_csv("./kaggle/Data/Stocks/us100.us.txt", parse_dates=True, index_col=0)
    df_stockIndex = df1.join(df_stockIndex)
    df_stockIndex[['Close']].plot(figsize=(15, 6))
    plt.ylabel("stock_price")
    plt.title("NASDAQ Stock")
    plt.show()
    df_stockIndex = df_stockIndex[['Close']]
    df_stockIndex = df_stockIndex.ffill()
    scaler = MinMaxScaler(feature_range=(-1, 1))

    df_stockIndex['Close'] = scaler.fit_transform(df_stockIndex['Close'].values.reshape(-1, 1))

    look_back = 60  # choose sequence length
    look_forward = 10
    x_train, y_train , x_test, y_test , x_validation, y_validation= load_data(df_stockIndex, look_back, look_forward)

    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    print('x_validation.shape = ', x_validation.shape)
    print('y_validation.shape = ', y_validation.shape)

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    x_validation = torch.from_numpy(x_validation).type(torch.Tensor)
    y_validation = torch.from_numpy(y_validation).type(torch.Tensor)
    input_dim = 1
    hidden_dim = 128
    num_layers = 2
    output_dim = look_forward
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    print(x_train.shape, x_test.shape)
    num_epochs =10

    hist = np.zeros(num_epochs*len(x_train))

    validation = np.zeros(num_epochs)

    # Number of steps to unroll
    cuda0 = torch.device(0)
    for t in range(num_epochs):
        for i in range(len(x_train)):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            y_train_pred = model(x_train[i].to(cuda0))

            loss = loss_fn(torch.unsqueeze(y_train_pred.cpu(), -1),y_train[i])

            hist[t*len(x_train)+i] = loss.item()

            optimiser.zero_grad()

                    # Backward pass
            loss.backward()

                    # Update parameters
            optimiser.step()

            y_train_pred = model(x_validation.to(cuda0))
            validation[t] = loss_fn(torch.unsqueeze(y_train_pred.cpu(), -1), y_validation).item()

        print("Epoch ", t, "MSE: ", hist[t*len(x_train)], "Validation: ", validation[t].item())

        # Zero out gradient, else they will accumulate between epochs

    plt.plot(validation, label="Validation loss", color='red')
    plt.plot(hist, label="Training loss")
    plt.legend()
    plt.show()


    # make predictions

    print(x_test.shape)
    y_test_pred = model(torch.unsqueeze(x_test.to(cuda0),0))




    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy().reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.detach().numpy().reshape(-1, 1))


    # Visualising the results
    figure, axes = plt.subplots(figsize=(15, 6))
    axes.xaxis_date()
    print(y_test_pred.shape)
    print(y_test.shape)
    print(df_stockIndex[:].index.shape)

    axes.plot(df_stockIndex[-y_test.shape[0]:].index, y_test, color='red', label='Real NASDQ Index Price')
    axes.plot(df_stockIndex[-y_test.shape[0]:].index, y_test_pred, color='blue',
              label='Predicted IBM Stock Price')
    # axes.xticks(np.arange(0,394,50))
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.savefig('ibm_pred.png')
    plt.show()
