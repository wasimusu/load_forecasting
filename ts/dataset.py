import numpy as np


def create_dataset(dataset, look_back=1, forecast_horizon=1, batch_size=1):
    batch_x, batch_y, batch_z = [], [], []
    for i in range(0, len(dataset) - look_back - forecast_horizon - batch_size + 1, batch_size):
        for n in range(batch_size):
            x = dataset[['log_counts', 'next_is_holiday', 'next_bad_weather']].values[i + n:(i + n + look_back), :]
            offset = x[0, 0]
            y = dataset['log_counts'].values[i + n + look_back:i + n + look_back + forecast_horizon]
            batch_x.append(np.array(x).reshape(look_back, -1))
            batch_y.append(np.array(y))
            batch_z.append(np.array(offset))
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_z = np.array(batch_z)
        batch_x[:, :, 0] -= batch_z.reshape(-1, 1)
        batch_y -= batch_z.reshape(-1, 1)
        yield batch_x, batch_y, batch_z
        batch_x, batch_y, batch_z = [], [], []
