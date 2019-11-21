import torch

n_repeats = 10
test_true_y = []
test_pred_y = []
for repeats in range(n_repeats):
    ep_loss = []
    preds = []
    for batch in create_dataset(df[df.index>="2018"], look_back=look_back, forecast_horizon=1, batch_size=1):
        try:
            batch = [torch.Tensor(x) for x in batch]
        except:
            break
        out = model.forward(batch[0].float(), batch_size=1)
        loss = model.loss(out, batch[1].float())
        ep_loss.append(loss.item())
        if repeats == 0:
            test_true_y.append((batch[1] + batch[2]).detach().numpy().reshape(-1))
        preds.append((out + batch[2]).detach().numpy().reshape(-1))
    print("{:0.4f}".format(100*np.mean(ep_loss)), end=", ")
    test_pred_y.append(preds)
test_true_y = np.array(test_true_y)
test_pred_y = np.array(test_pred_y)

mean = np.mean(test_pred_y, axis=0).reshape(-1)
std = np.std(test_pred_y, axis=0).reshape(-1)
lower = np.percentile(test_pred_y, 5, axis=0).reshape(-1)
upper = np.percentile(test_pred_y, 95, axis=0).reshape(-1)

fig, ax = plt.subplots(figsize=(15,6))
ax.plot(np.array(test_true_y).reshape(-1), label='truth')
ax.plot(mean, label='pred', c='brown', linestyle='--', alpha=0.5)
ax.fill_between([*range(len(test_true_y.reshape(-1)))], mean-2*std, mean+2*std, label='95%', color='brown', alpha=.3)
ax.fill_between([*range(len(test_true_y.reshape(-1)))], mean-3*std, mean+3*std, label='99%', color='orange', alpha=.4)
ax.set(title="Test set - 1 step ahead forecast", ylabel="Number of trips (scaled)", xlabel="Days", ylim=(12.7, 13.8))
ax.legend();