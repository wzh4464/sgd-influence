import os
import argparse
import numpy as np
import pandas as pd
import torch
from DataModule import MnistModule, NewsModule, AdultModule, EMNISTModule
from MyNet import LogReg, DNN, NetList
import warnings
from logging_utils import setup_logging

warnings.simplefilter(action="ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Parameters for ICML training
BATCH_SIZE = 200
LR = 0.01
MOMENTUM = 0.9
NUM_EPOCHS = 100


def get_data_module(key):
    if key == "mnist":
        return MnistModule()
    elif key == "20news":
        return NewsModule()
    elif key == "adult":
        return AdultModule(csv_path=os.path.join(SCRIPT_DIR, "data"))
    elif key == "emnist":
        return EMNISTModule()
    else:
        raise ValueError(f"Unsupported dataset: {key}")


def load_data(key, n_tr, n_val, n_test, seed, device):
    module = get_data_module(key)
    z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
    (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(np.expand_dims(y_tr, axis=1)).to(torch.float32).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(np.expand_dims(y_val, axis=1)).to(torch.float32).to(device)

    return x_tr, y_tr, x_val, y_val


def compute_gradient(x, y, model, loss_fn):
    z = model(x)
    loss = loss_fn(z, y)
    model.zero_grad()
    loss.backward()
    u = [param.grad.data.clone() for param in model.parameters()]
    for uu in u:
        uu.requires_grad = False
    return u


def get_file_paths(key, model_type, seed, infl_type=None):
    dn = os.path.join(SCRIPT_DIR, f"{key}_{model_type}")
    fn = os.path.join(dn, f"sgd{seed:03d}.dat")
    if infl_type:
        gn = os.path.join(dn, f"infl_{infl_type}{seed:03d}.dat")
        return dn, fn, gn
    return dn, fn


def infl_true(key, model_type, seed=0, gpu=0):
    dn, fn, gn = get_file_paths(key, model_type, seed, "true")
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key, res["n_tr"], res["n_val"], res["n_test"], seed, device
    )

    model = res["models"].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    # influence
    z = model(x_val)
    loss = loss_fn(z, y_val)
    infl = np.zeros(res["n_tr"])
    for i in range(res["n_tr"]):
        m = res["counterfactual"][i].models[-1].to(device)
        m.eval()
        zi = m(x_val)
        lossi = loss_fn(zi, y_val)
        infl[i] = lossi.item() - loss.item()

    torch.save(infl, gn)


def infl_sgd(key, model_type, seed=0, gpu=0):
    dn, fn, gn = get_file_paths(key, model_type, seed, "sgd")
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key, res["n_tr"], res["n_val"], res["n_test"], seed, device
    )

    model = res["models"].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]

    models = [m.to(device) for m in res["models"].models[:-1]]
    alpha = res["alpha"]
    info = res["info"]
    infl = np.zeros(res["n_tr"])

    for t in range(len(models) - 1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]["idx"], info[t]["lr"]
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size

        # update u
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
        ug = sum((uu * g).sum() for uu, g in zip(u, grad_params))
        m.zero_grad()
        ug.backward()
        for j, param in enumerate(m.parameters()):
            u[j] -= lr * param.grad.data

    torch.save(infl, gn)


def infl_nohess(key, model_type, seed=0, gpu=0):
    dn, fn, gn = get_file_paths(key, model_type, seed, "nohess")
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key, res["n_tr"], res["n_val"], res["n_test"], seed, device
    )

    model = res["models"].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]

    models = [m.to(device) for m in res["models"].models[:-1]]
    alpha = res["alpha"]
    info = res["info"]
    infl = np.zeros(res["n_tr"])

    for t in range(len(models) - 1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]["idx"], info[t]["lr"]
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size

    torch.save(infl, gn)


def infl_icml(key, model_type, seed=0, gpu=0):
    dn, fn, gn = get_file_paths(key, model_type, seed, "icml")
    hn = os.path.join(dn, f"loss_icml{seed:03d}.dat")
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key, res["n_tr"], res["n_val"], res["n_test"], seed, device
    )

    model = res["models"].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]

    alpha = 1.0 if model_type == "dnn" else res["alpha"]
    num_steps = int(np.ceil(res["n_tr"] / BATCH_SIZE))
    v = [uu.clone().to(device).requires_grad_(True) for uu in u]
    optimizer = torch.optim.SGD(v, lr=LR, momentum=MOMENTUM)
    loss_train = []

    for epoch in range(NUM_EPOCHS):
        model.eval()
        np.random.seed(epoch)
        idx_list = np.array_split(np.random.permutation(res["n_tr"]), num_steps)
        for i in range(num_steps):
            idx = idx_list[i]
            z = model(x_tr[idx])
            loss = loss_fn(z, y_tr[idx])
            model.zero_grad()
            grad_params = torch.autograd.grad(
                loss, model.parameters(), create_graph=True
            )
            vg = sum((vv * g).sum() for vv, g in zip(v, grad_params))
            model.zero_grad()
            vgrad_params = torch.autograd.grad(
                vg, model.parameters(), create_graph=True
            )
            loss_i = sum(
                0.5 * (vgp * vv + alpha * vv * vv).sum() - (uu * vv).sum()
                for vgp, vv, uu in zip(vgrad_params, v, u)
            )
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()
            loss_train.append(loss_i.item())

    torch.save(np.array(loss_train), hn)

    infl = np.zeros(res["n_tr"])
    for i in range(res["n_tr"]):
        z = model(x_tr[[i]])
        loss = loss_fn(z, y_tr[[i]])
        model.zero_grad()
        loss.backward()
        infl_i = sum(
            (param.grad.data.cpu().numpy() * v[j].data.cpu().numpy()).sum()
            for j, param in enumerate(model.parameters())
        )
        infl[i] = infl_i / res["n_tr"]

    torch.save(infl, gn)


def infl_lie_helper(key, model_type, custom_epoch, seed=0, gpu=0, logger=None):
    _, fn = get_file_paths(key, model_type, seed)
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key, res["n_tr"], res["n_val"], res["n_test"], seed, device
    )

    model = res["models"].models[-1].to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]

    steps_per_epoch = (res["n_tr"] + res["batch_size"] - 1) // res["batch_size"]
    total_steps = custom_epoch * steps_per_epoch

    logger.info(f"SPE: {steps_per_epoch}")
    logger.info(f"Total steps: {total_steps}")

    assert total_steps <= len(res["info"])

    models = [m.to(device) for m in res["models"].models[:total_steps]]
    alpha = res["alpha"]
    info = res["info"]
    infl = np.zeros(res["n_tr"])

    logger.info(f"Starting influence computation for epoch {custom_epoch}")

    for t in range(total_steps - 1, -1, -1):
        m = models[t]
        m.eval()
        idx, lr = info[t]["idx"], info[t]["lr"]
        for i in idx:
            z = m(x_tr[[i]])
            loss = loss_fn(z, y_tr[[i]])
            for p in m.parameters():
                loss += 0.5 * alpha * (p * p).sum()
            m.zero_grad()
            loss.backward()
            for j, param in enumerate(m.parameters()):
                infl[i] += lr * (u[j].data * param.grad.data).sum().item() / idx.size
        z = m(x_tr[idx])
        loss = loss_fn(z, y_tr[idx])
        for p in m.parameters():
            loss += 0.5 * alpha * (p * p).sum()
        grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
        ug = sum((uu * g).sum() for uu, g in zip(u, grad_params))
        m.zero_grad()
        ug.backward()
        for j, param in enumerate(m.parameters()):
            u[j] -= lr * param.grad.data

        if t % steps_per_epoch == 0:
            logger.info(f"Completed step {t} of {total_steps}")

    logger.info(f"Finished influence computation for epoch {custom_epoch}")
    return infl


def infl_lie(key, model_type, seed=0, gpu=0, is_csv=True):
    dn, fn = get_file_paths(key, model_type, seed)
    os.makedirs(dn, exist_ok=True)
    csv_fn = os.path.join(dn, f"infl_lie_full_{seed}.csv")

    logger = setup_logging(f"infl_lie_{key}_{model_type}", seed)
    logger.info(f"Starting infl_lie computation for {key}, {model_type}, seed {seed}")

    res = torch.load(fn, map_location=f"cuda:{gpu}")
    num_epochs = res["num_epoch"]
    infl_list = []

    for epoch in range(num_epochs + 1):
        logger.info(f"Computing influence for epoch {epoch}")
        infl = infl_lie_helper(key, model_type, epoch, seed, gpu, logger)
        infl_list.append(infl)
        logger.info(f"Completed influence computation for epoch {epoch}")

    diff_dict = {
        f"diff_{i-1}_{i}": infl_list[i] - infl_list[i - 1]
        for i in range(1, len(infl_list))
    }

    if is_csv:
        pd.DataFrame(diff_dict).to_csv(csv_fn, index=False)
        logger.info(f"CSV results saved to {csv_fn}")

    torch.save(infl_list, os.path.join(dn, f"infl_lie_full_{seed:03d}.dat"))
    logger.info(
        f"Full results saved to {os.path.join(dn, f'infl_lie_full_{seed:03d}.dat')}"
    )

    logger.info("infl_lie computation completed")
    print(f"Results saved to {csv_fn}")


def main():
    parser = argparse.ArgumentParser(description="Compute Influence Functions")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--type", default="true", type=str, help="influence type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    args = parser.parse_args()

    if args.target not in ["mnist", "20news", "adult", "emnist"]:
        raise ValueError(
            "Invalid target data. Choose from 'mnist', '20news', or 'adult'."
        )
    if args.model not in ["logreg", "dnn"]:
        raise ValueError("Invalid model type. Choose from 'logreg' or 'dnn'.")
    if args.type not in ["true", "sgd", "nohess", "icml", "lie"]:
        raise ValueError(
            "Invalid influence type. Choose from 'true', 'sgd', 'nohess', 'icml', or 'lie'."
        )

    influence_functions = {
        "true": infl_true,
        "sgd": infl_sgd,
        "nohess": infl_nohess,
        "icml": infl_icml,
        "lie": infl_lie,
    }

    infl_func = influence_functions[args.type]

    if args.seed >= 0:
        infl_func(args.target, args.model, args.seed, args.gpu)
    else:
        for seed in range(100):
            infl_func(args.target, args.model, seed, args.gpu)


if __name__ == "__main__":
    main()
