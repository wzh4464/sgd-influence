import os
import argparse
import numpy as np
import pandas as pd
import torch
from DataModule import fetch_data_module, DATA_MODULE_REGISTRY
from NetworkModule import NETWORK_REGISTRY
import warnings
from logging_utils import setup_logging
import logging
from NetworkModule import get_network
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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


def load_full_results(dn: str, seed: int, prefix: str) -> List[np.ndarray]:
    full_results_path = os.path.join(dn, f"infl_{prefix}_full_{seed:03d}.dat")
    return torch.load(full_results_path)

def save_influence(infl: np.ndarray, dn: str, seed: int, suffix: str):
    save_path = os.path.join(dn, f"infl_{suffix}{seed:03d}.dat")
    torch.save(infl, save_path)

def infl_helper(key: str, model_type: str, seed: int, gpu: int, save_dir: str, epoch_index: int, suffix: str, prefix: str, relabel_percentage: float = None):
    logger = logging.getLogger(f"infl_{suffix}_{key}_{model_type}")
    logger.info(f"Starting infl_{suffix} computation for {key}, {model_type}, seed {seed}")

    dn, _ = get_file_paths(key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage)
    
    full_results = load_full_results(dn, seed, prefix)
    
    if epoch_index >= len(full_results) - 1:
        raise ValueError(f"Epoch index {epoch_index} is out of range for results with {len(full_results)} epochs.")
    
    infl = full_results[epoch_index + 1] - full_results[epoch_index]
    save_influence(infl, dn, seed, suffix)
    
    logger.info(f"Finished infl_{suffix} computation for {key}, {model_type}, seed {seed}")

def infl_dit_first(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    infl_helper(key, model_type, seed, gpu, save_dir, 0, "dit_first", "lie", relabel_percentage)

def infl_dit_middle(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    dn, _ = get_file_paths(key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage)
    full_results = load_full_results(dn, seed, "lie")
    middle_epoch = len(full_results) // 2 - 1
    infl_helper(key, model_type, seed, gpu, save_dir, middle_epoch, "dit_middle", "lie", relabel_percentage)

def infl_dit_last(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    dn, _ = get_file_paths(key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage)
    full_results = load_full_results(dn, seed, "lie")
    last_epoch = len(full_results) - 2
    infl_helper(key, model_type, seed, gpu, save_dir, last_epoch, "dit_last", "lie", relabel_percentage)

def infl_true_first(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    infl_helper(key, model_type, seed, gpu, save_dir, 0, "true_first", "segment_true", relabel_percentage)

def infl_true_middle(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    dn, _ = get_file_paths(key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage)
    full_results = load_full_results(dn, seed, "segment_true")
    middle_epoch = len(full_results) // 2 - 1
    infl_helper(key, model_type, seed, gpu, save_dir, middle_epoch, "true_middle", "segment_true", relabel_percentage)

def infl_true_last(key: str, model_type: str, seed: int = 0, gpu: int = 0, save_dir: str = None, relabel_percentage: float = None):
    dn, _ = get_file_paths(key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage)
    full_results = load_full_results(dn, seed, "segment_true")
    last_epoch = len(full_results) - 2
    infl_helper(key, model_type, seed, gpu, save_dir, last_epoch, "true_last", "segment_true", relabel_percentage)

def load_data(
    key,
    n_tr,
    n_val,
    n_test,
    seed,
    device,
    logger=None,
    relabel_percentage=None,
    dn=None,
):
    module = fetch_data_module(
        key, data_dir=os.path.join(SCRIPT_DIR, "data"), logger=logger, seed=seed
    )
    module.append_one = False

    z_tr, z_val, _ = module.fetch(n_tr, n_val, n_test, seed)
    (x_tr, y_tr), (x_val, y_val) = z_tr, z_val

    # Convert to tensor
    x_tr = torch.from_numpy(x_tr).to(torch.float32).to(device)
    y_tr = torch.from_numpy(y_tr).to(torch.float32).unsqueeze(1).to(device)
    x_val = torch.from_numpy(x_val).to(torch.float32).to(device)
    y_val = torch.from_numpy(y_val).to(torch.float32).unsqueeze(1).to(device)

    if relabel_percentage:
        idx_csv_name = os.path.join(dn, f"relabeled_indices_{seed:03d}.csv")
        # pd.DataFrame({"relabeled_indices": relabeled_indices}).to_csv(
        #     relabeled_indices_fn, index=False
        # )
        logger.debug(f"Relabeling {relabel_percentage}% of the training data")
        relabeled_indices = pd.read_csv(idx_csv_name)["relabeled_indices"].values
        y_tr[relabeled_indices] = 1 - y_tr[relabeled_indices]

    return x_tr, y_tr, x_val, y_val


def get_input_dim(x, model_type):
    if model_type == "cnn":
        if x.dim() == 2:
            # Flattened input, need to reshape
            img_size = int(np.sqrt(x.shape[1]))
            input_dim = (1, img_size, img_size)
        else:
            # For images, input_dim is the shape of the image
            input_dim = x.shape[1:]
    else:
        # For other models, input_dim is the number of features
        input_dim = x.shape[1:]
    return input_dim


def compute_gradient(x, y, model, loss_fn):
    z = model(x)
    loss = loss_fn(z, y)
    model.zero_grad()
    loss.backward()
    u = [param.grad.data.clone() for param in model.parameters()]
    for uu in u:
        uu.requires_grad = False
    return u


def get_file_paths(
    key, model_type, seed, infl_type=None, save_dir=None, relabel_percentage=None
):
    dn = (
        os.path.join(SCRIPT_DIR, save_dir)
        if save_dir
        else os.path.join(SCRIPT_DIR, f"{key}_{model_type}")
    )

    # Modify filename if relabel_percentage is set
    if relabel_percentage is not None:
        fn = os.path.join(
            dn, f"relabeled_indices_{seed:03d}_sgd{seed:03d}.dat"
        )  # New filename for relabeled data
    else:
        fn = os.path.join(dn, f"sgd{seed:03d}.dat")

    if infl_type:
        gn = os.path.join(dn, f"infl_{infl_type}{seed:03d}.dat")
        return dn, fn, gn
    return dn, fn


def infl_true(key, model_type, seed=0, gpu=0, save_dir=None, relabel_percentage=None):
    logger = logging.getLogger(f"infl_true_{key}_{model_type}")
    logger.info(f"Starting infl_true computation for {key}, {model_type}, seed {seed}")

    dn, fn, gn = get_file_paths(
        key, model_type, seed, "true", save_dir, relabel_percentage=relabel_percentage
    )
    os.makedirs(dn, exist_ok=True)  # Ensure the directory exists
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    # Get input_dim
    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    # Influence computation
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
    logger.info(f"Finished infl_true computation for {key}, {model_type}, seed {seed}")


def infl_segment_true(
    key, model_type, seed=0, gpu=0, save_dir=None, relabel_percentage=None
):
    logger = logging.getLogger(f"infl_segment_true_{key}_{model_type}")
    logger.info(
        f"Starting infl_segment_true computation for {key}, {model_type}, seed {seed}"
    )

    dn, fn = get_file_paths(
        key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage
    )
    os.makedirs(dn, exist_ok=True)
    csv_fn = os.path.join(dn, f"infl_segment_true_{seed}.csv")

    logger = setup_logging(
        f"infl_segment_true_{key}_{model_type}", seed, dn, level=logging.INFO
    )
    logger.debug(
        f"Starting infl_segment_true computation for {key}, {model_type}, seed {seed}"
    )

    device = f"cuda:{gpu}"
    res = torch.load(fn, map_location=device)
    x_tr, _, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    num_epochs = res["num_epoch"]
    infl_list = []

    # Compute influence for each epoch
    for epoch in range(num_epochs + 1):
        logger.debug(f"Computing influence for epoch {epoch}")

        # Calculate the number of steps for this epoch
        steps_per_epoch = int(np.ceil(res["n_tr"] / res["batch_size"]))
        total_steps = epoch * steps_per_epoch

        # Get input_dim
        input_dim = get_input_dim(x_tr, model_type)
        # Initialize model with correct input_dim
        model = get_network(model_type, input_dim).to(device)
        # Use the model state at the end of this epoch
        if epoch < num_epochs:
            # Load state dict
            model.load_state_dict(res["models"].models[total_steps].state_dict())

        else:
            # Load state dict
            model.load_state_dict(res["models"].models[-1].state_dict())

        model.eval()

        # Compute influence
        z = model(x_val)
        loss = loss_fn(z, y_val)
        infl = np.zeros(res["n_tr"])

        for i in range(res["n_tr"]):
            m = res["counterfactual"][i].models[epoch].to(device)
            m.eval()
            zi = m(x_val)
            lossi = loss_fn(zi, y_val)
            infl[i] = lossi.item() - loss.item()

        infl_list.append(infl)
        logger.debug(f"Completed influence computation for epoch {epoch}")

    # Compute differences between consecutive epochs
    diff_dict = {
        f"diff_{i-1}_{i}": infl_list[i] - infl_list[i - 1]
        for i in range(1, len(infl_list))
    }

    # Save to CSV
    pd.DataFrame(diff_dict).to_csv(csv_fn, index=False)
    logger.debug(f"CSV results saved to {csv_fn}")

    # Save full results
    torch.save(infl_list, os.path.join(dn, f"infl_segment_true_full_{seed:03d}.dat"))
    logger.debug(
        f"Full results saved to {os.path.join(dn, f'infl_segment_true_full_{seed:03d}.dat')}"
    )

    logger.debug(f"Results saved to {csv_fn}")
    logger.info(
        f"Finished infl_segment_true computation for {key}, {model_type}, seed {seed}"
    )


def infl_sgd(key, model_type, seed=0, gpu=0, save_dir=None, relabel_percentage=None):
    dn, fn, gn = get_file_paths(
        key, model_type, seed, "sgd", save_dir, relabel_percentage=relabel_percentage
    )
    device = f"cuda:{gpu}"

    logger = logging.getLogger(f"infl_sgd_{key}_{model_type}")
    logger.info(f"Starting infl_icml computation for {key}, {model_type}, seed {seed}")

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    # Get input_dim
    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

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


def infl_nohess(key, model_type, seed=0, gpu=0, save_dir=None, relabel_percentage=None):
    dn, fn, gn = get_file_paths(
        key, model_type, seed, "nohess", save_dir, relabel_percentage=relabel_percentage
    )
    device = f"cuda:{gpu}"

    logger = logging.getLogger(f"infl_nohess_{key}_{model_type}")
    logger.info(f"Starting infl_icml computation for {key}, {model_type}, seed {seed}")

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    # Get input_dim
    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

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


def infl_icml(key, model_type, seed=0, gpu=0, save_dir=None, relabel_percentage=None):
    logger = logging.getLogger(f"infl_icml_{key}_{model_type}")
    logger.info(f"Starting infl_icml computation for {key}, {model_type}, seed {seed}")

    dn, fn, gn = get_file_paths(
        key, model_type, seed, "icml", save_dir, relabel_percentage=relabel_percentage
    )
    hn = os.path.join(dn, f"loss_icml{seed:03d}.dat")
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)
    x_tr, y_tr, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    # Get input_dim
    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

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
    logger.info(f"Finished infl_icml computation for {key}, {model_type}, seed {seed}")


def infl_lie_helper(
    key,
    model_type,
    custom_epoch,
    seed=0,
    gpu=0,
    logger=None,
    fn=None,
    relabel_percentage=None,
    dn=None,
):
    device = f"cuda:{gpu}"

    res = torch.load(fn, map_location=device)

    x_tr, y_tr, x_val, y_val = load_data(
        key,
        res["n_tr"],
        res["n_val"],
        res["n_test"],
        seed,
        device,
        relabel_percentage=relabel_percentage,
        dn=dn,
        logger=logger,
    )

    # Get input_dim
    input_dim = get_input_dim(x_tr, model_type)
    # Initialize model with correct input_dim
    model = get_network(model_type, input_dim).to(device)
    # Load state dict
    model.load_state_dict(res["models"].models[-1].state_dict())

    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.eval()

    u = compute_gradient(x_val, y_val, model, loss_fn)
    u = [uu.to(device) for uu in u]

    steps_per_epoch = (res["n_tr"] + res["batch_size"] - 1) // res["batch_size"]
    total_steps = custom_epoch * steps_per_epoch

    logger.debug(f"SPE: {steps_per_epoch}")
    logger.debug(f"Total steps: {total_steps}")

    assert total_steps <= len(res["info"])

    models = [m.to(device) for m in res["models"].models[:total_steps]]
    alpha = res["alpha"]
    info = res["info"]
    infl = np.zeros(res["n_tr"])

    logger.debug(f"Starting influence computation for epoch {custom_epoch}")

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
            logger.debug(f"Completed step {t} of {total_steps}")

    logger.debug(f"Finished influence computation for epoch {custom_epoch}")
    return infl


def infl_lie(
    key, model_type, seed=0, gpu=0, is_csv=True, save_dir=None, relabel_percentage=None
):
    logger = logging.getLogger(f"infl_lie_{key}_{model_type}")
    logger.info(f"Starting infl_lie computation for {key}, {model_type}, seed {seed}")

    dn, fn = get_file_paths(
        key, model_type, seed, save_dir=save_dir, relabel_percentage=relabel_percentage
    )
    os.makedirs(dn, exist_ok=True)
    csv_fn = os.path.join(dn, f"infl_lie_full_{seed}.csv")

    logger = setup_logging(f"infl_lie_{key}_{model_type}", seed, dn, level=logging.INFO)
    logger.debug(f"Starting infl_lie computation for {key}, {model_type}, seed {seed}")

    res = torch.load(fn, map_location=f"cuda:{gpu}")
    num_epochs = res["num_epoch"]
    infl_list = []

    for epoch in range(num_epochs + 1):
        logger.debug(f"Computing influence for epoch {epoch}")
        infl = infl_lie_helper(
            key,
            model_type,
            epoch,
            seed,
            gpu,
            logger,
            fn,
            relabel_percentage=relabel_percentage,
            dn=dn,
        )
        infl_list.append(infl)
        logger.debug(f"Completed influence computation for epoch {epoch}")

    diff_dict = {
        f"diff_{i-1}_{i}": infl_list[i] - infl_list[i - 1]
        for i in range(1, len(infl_list))
    }

    if is_csv:
        pd.DataFrame(diff_dict).to_csv(csv_fn, index=False)
        logger.debug(f"CSV results saved to {csv_fn}")

    torch.save(infl_list, os.path.join(dn, f"infl_lie_full_{seed:03d}.dat"))
    logger.debug(
        f"Full results saved to {os.path.join(dn, f'infl_lie_full_{seed:03d}.dat')}"
    )

    # Save first, middle, and last epoch differences
    first_epoch_diff = infl_list[1] - infl_list[0]
    middle_epoch = num_epochs // 2
    middle_epoch_diff = infl_list[middle_epoch + 1] - infl_list[middle_epoch]
    last_epoch_diff = infl_list[-1] - infl_list[-2]

    torch.save(first_epoch_diff, os.path.join(dn, f"infl_lie_first{seed:03d}.dat"))
    torch.save(middle_epoch_diff, os.path.join(dn, f"infl_lie_middle{seed:03d}.dat"))
    torch.save(last_epoch_diff, os.path.join(dn, f"infl_lie_last{seed:03d}.dat"))

    logger.info(f"Finished infl_lie computation for {key}, {model_type}, seed {seed}")
    logger.debug(f"Results saved to {csv_fn}")
    logger.debug("Additional results saved for first, middle, and last epochs")


def main():
    parser = argparse.ArgumentParser(description="Compute Influence Functions")
    parser.add_argument("--target", default="adult", type=str, help="target data")
    parser.add_argument("--model", default="logreg", type=str, help="model type")
    parser.add_argument("--type", default="true", type=str, help="influence type")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--gpu", default=0, type=int, help="gpu index")
    parser.add_argument("--save_dir", type=str, help="directory to save results")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--relabel", type=float, help="percentage of training data to relabel"
    )

    args = parser.parse_args()

    # Log level setup
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Validate arguments
    if args.target not in DATA_MODULE_REGISTRY:
        raise ValueError(
            f"Invalid target data. Choose from {', '.join(DATA_MODULE_REGISTRY.keys())}."
        )
    if args.model not in NETWORK_REGISTRY:
        raise ValueError(
            f"Invalid model type. Choose from {', '.join(NETWORK_REGISTRY.keys())}."
        )
    if args.type not in ["true", "segment_true", "sgd", "nohess", "icml", "lie", "dit_first", "dit_middle", "dit_last", "true_first", "true_middle", "true_last"]:
        raise ValueError(
            "Invalid influence type. Choose from 'true', 'segment_true', 'sgd', 'nohess', 'icml', 'lie', 'dit_first', 'dit_middle', 'dit_last', 'true_first', 'true_middle', 'true_last'."
        )

    influence_functions = {
        "true": infl_true,
        "segment_true": infl_segment_true,
        "sgd": infl_sgd,
        "nohess": infl_nohess,
        "icml": infl_icml,
        "lie": infl_lie,
        "dit_first": infl_dit_first,
        "dit_middle": infl_dit_middle,
        "dit_last": infl_dit_last,
        "true_first": infl_true_first,
        "true_middle": infl_true_middle,
        "true_last": infl_true_last,
    }

    infl_func = influence_functions[args.type]

    if args.seed >= 0:
        infl_func(
            args.target,
            args.model,
            args.seed,
            args.gpu,
            save_dir=args.save_dir,
            relabel_percentage=args.relabel,
        )
    else:
        for seed in range(100):
            infl_func(
                args.target,
                args.model,
                seed,
                args.gpu,
                save_dir=args.save_dir,
                relabel_percentage=args.relabel,
            )


if __name__ == "__main__":
    main()
