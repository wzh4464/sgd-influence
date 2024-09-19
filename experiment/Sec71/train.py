import os, sys
import argparse
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import joblib
import torch
import torch.nn as nn
from DataModule import MnistModule, NewsModule, AdultModule
from MyNet import LogReg, DNN, NetList

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def settings_logreg(key):
    assert key in ["mnist", "20news", "adult"]
    if key == "mnist":
        module = MnistModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.1, True, 5, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)
    elif key == "20news":
        module = NewsModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.01, True, 10, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)
    elif key == "adult":
        module = AdultModule(csv_path="./data")
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        lr, decay, num_epoch, batch_size = 0.1, True, 20, 5
        return module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size)


def settings_dnn(key):
    assert key in ["mnist", "20news", "adult"]
    if key == "mnist":
        module = MnistModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 12, 20
        return (
            module,
            (n_tr, n_val, n_test),
            m,
            alpha,
            (lr, decay, num_epoch, batch_size),
        )
    elif key == "20news":
        module = NewsModule()
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 10, 20
        return (
            module,
            (n_tr, n_val, n_test),
            m,
            alpha,
            (lr, decay, num_epoch, batch_size),
        )
    elif key == "adult":
        module = AdultModule(csv_path="./data")
        module.append_one = False
        n_tr, n_val, n_test = 200, 200, 200
        m = [8, 8]
        alpha = 0.001
        lr, decay, num_epoch, batch_size = 0.1, False, 12, 20
        return (
            module,
            (n_tr, n_val, n_test),
            m,
            alpha,
            (lr, decay, num_epoch, batch_size),
        )


def test(key, model_type, seed=0, gpu=0):
    dn = f"./{key}_{model_type}"
    fn = "%s/sgd%03d.dat" % (dn, seed)
    os.makedirs(dn, exist_ok=True)
    # device = "cuda:%d" % (gpu,)

    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS")
    else:
        device = "cpu"

    # fetch data
    if model_type == "logreg":
        module, (n_tr, n_val, n_test), (lr, decay, num_epoch, batch_size) = (
            settings_logreg(key)
        )
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

        # selection of alpha
        model = LogisticRegressionCV(random_state=seed, fit_intercept=False, cv=5)
        model.fit(x_tr, y_tr)
        alpha = 1 / (model.C_[0] * n_tr)

        # model
        net_func = lambda: LogReg(x_tr.shape[1]).to(device)
    elif model_type == "dnn":
        module, (n_tr, n_val, n_test), m, alpha, (lr, decay, num_epoch, batch_size) = (
            settings_dnn(key)
        )
        z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
        (x_tr, y_tr), (x_val, y_val) = z_tr, z_val
        net_func = lambda: DNN(x_tr.shape[1]).to(device)

    # to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)

    # fit
    num_steps = int(np.ceil(n_tr / batch_size))
    list_of_sgd_models = []
    list_of_counterfactual_models = [NetList([]) for _ in range(n_tr)]  # 每个样本有一个 NetList，保存该样本的训练过程
    main_losses = []
    # counterfactual_losses = np.zeros((n_tr, num_epoch * num_steps + 1))
    train_losses = np.zeros(num_epoch * num_steps + 1)  # 用于保存每步的训练损失
    train_losses[0] = nn.BCEWithLogitsLoss()(net_func()(x_tr), y_tr).item() # 记录初始损失
    # main_losses.append(train_losses[0])

    val_interval = 10  # 设置验证损失的计算间隔

    for n in range(-1, n_tr):
        torch.manual_seed(seed)
        model = net_func()
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.0)
        lr_n = lr
        skip = [n]
        info = []
        c = 0
        for epoch in range(num_epoch):
            np.random.seed(epoch)
            idx_list = np.array_split(np.random.permutation(n_tr), num_steps)
            for i in range(num_steps):
                info.append({"idx": idx_list[i], "lr": lr_n})
                c += 1

                # store model
                if n < 0:
                    # 保存 SGD 模型和验证损失（如果符合间隔要求）
                    m = net_func()
                    m.load_state_dict(copy.deepcopy(model.state_dict()))
                    list_of_sgd_models.append(m)
                    if c % val_interval == 0 or c == num_steps * num_epoch:
                        with torch.no_grad():
                            main_losses.append(loss_fn(model(x_val), y_val).item())
                else:
                    # 保存反事实模型和验证损失（如果符合间隔要求）
                    m = net_func()
                    m.load_state_dict(copy.deepcopy(model.state_dict()))
                    list_of_counterfactual_models[n].models.append(m)
                    # if c % val_interval == 0 or c == num_steps * num_epoch:
                    #     with torch.no_grad():
                    #         counterfactual_losses[n, c - 1] = loss_fn(
                    #             model(x_val), y_val
                    #         ).item()

                # SGD 优化
                idx = idx_list[i]
                b = idx.size
                idx = np.setdiff1d(idx, skip)
                z = model(x_tr[idx])
                loss = loss_fn(z, y_tr[idx])

                # 记录训练损失
                train_losses[c] = loss.item()

                # 添加正则化项
                for p in model.parameters():
                    loss += 0.5 * alpha * (p * p).sum()
                optimizer.zero_grad()
                loss.backward()
                for p in model.parameters():
                    p.grad.data *= idx.size / b
                optimizer.step()

                # 学习率衰减
                if decay:
                    lr_n *= np.sqrt(c / (c + 1))
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_n

        # 最后一步保存模型
        if n < 0:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            list_of_sgd_models.append(m)
            main_losses.append(loss_fn(model(x_val), y_val).item())
        else:
            m = net_func()
            m.load_state_dict(copy.deepcopy(model.state_dict()))
            list_of_counterfactual_models[n].models.append(m)

    # 保存所有数据
    joblib.dump(
        {
            "models": NetList(list_of_sgd_models),
            "info": info,
            "counterfactual": list_of_counterfactual_models,
            "alpha": alpha,
            "main_losses": main_losses,
            # "counterfactual_losses": counterfactual_losses,
            "train_losses": train_losses,  # 保存训练损失
        },
        fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Models & Save")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    args = parser.parse_args()
    assert args.target in ["mnist", "20news", "adult"]
    assert args.model in ["logreg", "dnn"]
    if args.seed >= 0:
        test(args.target, args.model, args.seed, args.gpu)
    else:
        for seed in range(100):
            test(args.target, args.model, seed, args.gpu)
