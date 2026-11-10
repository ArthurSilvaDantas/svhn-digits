"""Código compartilhado pelos notebooks: configuração, dados, pré-processamento, modelos, treino e experimentos.

As funções e classes vêm das células do notebook original do estudo, sem mudanças de lógica; aqui elas
ficam num só lugar para que cada notebook possa ser executado de forma independente.
"""

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Configuração
NOTEBOOKS_DIR = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOKS_DIR.parent / "data"
RESULTS_DIR = NOTEBOOKS_DIR / "outputs" / "results"
WEIGHTS_DIR = NOTEBOOKS_DIR / "outputs" / "weights"
FIG_DIR = NOTEBOOKS_DIR / "outputs" / "figures"
for _d in (RESULTS_DIR, WEIGHTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED_SPLIT = 42      # semente da divisão treino/validação
VAL_FRACTION = 0.10  # 10% do split "train" vira validação
HF_REPO = "arthursilvadantas/SVHN"


def figures_dir(notebook):
    """Subpasta de outputs/figures/ onde um notebook salva suas figuras (ex.: "07_final_model")."""
    d = FIG_DIR / notebook
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Dados
def stratified_split(y, frac, seed):
    g = torch.Generator().manual_seed(seed)
    val_idx = []
    for c in range(10):
        idx = (y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        val_idx.append(idx[: int(round(len(idx) * frac))])
    val_idx = torch.cat(val_idx)
    mask = torch.ones(len(y), dtype=torch.bool)
    mask[val_idx] = False
    return mask.nonzero().squeeze(1), val_idx


def load_data(download=True):
    """Retorna (data, split): data = {"train", "val", "test"} -> (X uint8 [N,3,32,32], y [N]);
    split = índices de treino e validação no split oficial `train`."""
    from torchvision.datasets import SVHN

    train_set = SVHN(DATA_DIR, split="train", download=download)
    test_set = SVHN(DATA_DIR, split="test", download=download)
    X_all = torch.from_numpy(train_set.data)              # [N, 3, 32, 32] uint8
    y_all = torch.from_numpy(train_set.labels).long()      # rótulos 0..9
    X_test = torch.from_numpy(test_set.data)
    y_test = torch.from_numpy(test_set.labels).long()
    tr_idx, va_idx = stratified_split(y_all, VAL_FRACTION, SEED_SPLIT)
    data = {
        "train": (X_all[tr_idx], y_all[tr_idx]),
        "val": (X_all[va_idx], y_all[va_idx]),
        "test": (X_test, y_test),
    }
    return data, {"train": tr_idx, "val": va_idx}


# Pré-processamento e data augmentation
def channel_stats(X_uint8):
    Xf = X_uint8.float() / 255.0
    return Xf.mean(dim=(0, 2, 3)), Xf.std(dim=(0, 2, 3))


class Normalizer:
    def __init__(self, mean, std, device):
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1).to(device)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1).to(device)

    def __call__(self, x_uint8):
        return (x_uint8.float() / 255.0 - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean.flatten().cpu().tolist(), "std": self.std.flatten().cpu().tolist()}


def augment(x, max_rot=10.0, max_shift=0.1, scale=(0.9, 1.1), jitter=0.2):
    # x: float em [0,1], [B,3,H,W]; parâmetros aleatórios independentes por imagem
    B, dev = x.shape[0], x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(B, device=dev).uniform_(*scale)
    tx = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2   # coordenadas normalizadas [-1,1]
    ty = (torch.rand(B, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    brightness = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
    contrast = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return (((x - m) * contrast + m) * brightness).clamp(0, 1)


# Modelos
def he_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


class MLP(nn.Module):
    # 3072 -> camadas ocultas -> 10, opcionalmente com BatchNorm e Dropout
    def __init__(self, hidden=(512, 256), batchnorm=False, dropout=0.0):
        super().__init__()
        layers, d = [nn.Flatten()], 32 * 32 * 3
        for h in hidden:
            layers.append(nn.Linear(d, h, bias=not batchnorm))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 10))
        self.net = nn.Sequential(*layers)
        he_init(self)

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    # 32x32x3 -conv5x5,pad2-> 32x32x32 -pool-> 16x16x32 -conv5x5,pad2-> 16x16x64 -pool-> 8x8x64 -> FC256 -> FC10
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 8 * 64, 256), nn.ReLU(inplace=True), nn.Linear(256, 10))
        he_init(self)

    def forward(self, x):
        return self.classifier(self.features(x))


class VGGNet(nn.Module):
    # cada bloco = convs_per_block x [CONV3x3 -> (BN) -> ReLU] + MaxPool2x2 (+ Dropout2d opcional)
    # classificador = (Dropout) -> FC oculta -> (BN) -> ReLU -> (Dropout) -> FC 10
    def __init__(self, channels=(32, 64, 128), convs_per_block=2, batchnorm=False,
                 dropout=0.0, conv_dropout=0.0, fc=256):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            for _ in range(convs_per_block):
                layers.append(nn.Conv2d(c_in, c, 3, padding=1, bias=not batchnorm))
                if batchnorm:
                    layers.append(nn.BatchNorm2d(c))
                layers.append(nn.ReLU(inplace=True))
                c_in = c
            layers.append(nn.MaxPool2d(2))
            if conv_dropout > 0:
                layers.append(nn.Dropout2d(conv_dropout))
        self.features = nn.Sequential(*layers)
        spatial = 32 // (2 ** len(channels))
        head = [nn.Flatten()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(c_in * spatial * spatial, fc, bias=not batchnorm))
        if batchnorm:
            head.append(nn.BatchNorm1d(fc))
        head.append(nn.ReLU(inplace=True))
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(fc, 10))
        self.classifier = nn.Sequential(*head)
        he_init(self)

    def forward(self, x):
        return self.classifier(self.features(x))


ARCHS = {"mlp": MLP, "simple_cnn": SimpleCNN, "vgg": VGGNet}


def build_model(arch, **kw):
    return ARCHS[arch](**kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# Treino, avaliação e inferência
def iterate(X, y, batch_size, shuffle):
    n = len(X)
    order = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield X[idx], (y[idx] if y is not None else None)


@torch.no_grad()
def evaluate(model, X, y, norm, batch_size=1024):
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in iterate(X, y, batch_size, shuffle=False):
        logits = model(norm(xb))
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(y), correct / len(y)


@torch.no_grad()
def predict(model, X, norm, batch_size=1024):
    model.eval()
    return torch.cat([model(norm(xb)).argmax(1) for xb, _ in iterate(X, None, batch_size, False)])


@torch.no_grad()
def predict_proba(model, X, norm, batch_size=1024):
    model.eval()
    return torch.cat([F.softmax(model(norm(xb)), 1).cpu() for xb, _ in iterate(X, None, batch_size, False)])


def make_optimizer(name, params, lr, weight_decay=0.0):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    raise ValueError(name)


def make_scheduler(name, opt, epochs, steps_per_epoch):
    if name == "none":
        return None, None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=2), "epoch"
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=opt.param_groups[0]["lr"], total_steps=epochs * steps_per_epoch, pct_start=0.15), "step"
    raise ValueError(name)


def train(model, data, cfg, device, verbose=True):
    set_seed(cfg["seed"])
    Xtr, ytr = (t.to(device) for t in data["train"])
    Xva, yva = (t.to(device) for t in data["val"])
    norm = Normalizer(*channel_stats(data["train"][0]), device)

    model = model.to(device)
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"], cfg["weight_decay"])
    steps = math.ceil(len(ytr) / cfg["batch_size"])
    sched, sched_mode = make_scheduler(cfg["scheduler"], opt, cfg["epochs"], steps)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]}
    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        for xb, yb in iterate(Xtr, ytr, cfg["batch_size"], shuffle=True):
            x = (augment(xb.float() / 255.0) - norm.mean) / norm.std if cfg["augment"] else norm(xb)
            loss = F.cross_entropy(model(x), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched_mode == "step":
                sched.step()

        tr_loss, tr_acc = evaluate(model, Xtr, ytr, norm)
        va_loss, va_acc = evaluate(model, Xva, yva, norm)
        hist["lr"].append(opt.param_groups[0]["lr"])
        if sched_mode == "epoch":
            sched.step(va_loss)
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "time"],
                        [tr_loss, tr_acc, va_loss, va_acc, time.time() - t0]):
            hist[k].append(v)
        if verbose:
            print(f"  ep {epoch:3d} | treino loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {hist['lr'][-1]:.1e} | {hist['time'][-1]:.1f}s")

        if va_acc > best_acc:
            best_acc, best_epoch, bad = va_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  early stopping na época {epoch} (melhor época: {best_epoch})")
                break

    model.load_state_dict(best_state)
    hist["best_epoch"], hist["epochs_run"] = best_epoch, len(hist["train_loss"])
    return model, norm, hist


# Experimentos
BASE = dict(optimizer="adam", lr=1e-3, batch_size=128, epochs=30, patience=8,
            scheduler="none", augment=False, weight_decay=0.0, seed=0)


def exp(id, desc, arch, model_kw=None, base=None, **train_kw):
    return dict(id=id, desc=desc, arch=arch, model_kw=model_kw or {}, base=base, train={**BASE, **train_kw})


VGG = dict(channels=(32, 64, 128), convs_per_block=2)
VGG_BN = {**VGG, "batchnorm": True}
VGG_BN_DO = {**VGG_BN, "dropout": 0.5, "conv_dropout": 0.1}
AUG = dict(augment=True, epochs=40, patience=10)

EXPERIMENTS = [
    # MLP
    exp("E1", "MLP baseline (3072-512-256-10)", "mlp", dict(hidden=(512, 256))),
    exp("E2", "MLP + BatchNorm + Dropout 0.3", "mlp", dict(hidden=(512, 256), batchnorm=True, dropout=0.3), base="E1"),
    # CNN
    exp("E3", "CNN baseline (LeNet-like)", "simple_cnn", base="E1"),
    exp("E4", "CNN VGG-like (3 blocos × 2 conv3x3)", "vgg", VGG, base="E3"),
    exp("E5", "+ BatchNorm", "vgg", VGG_BN, base="E4"),
    exp("E6", "+ Dropout (0.5 FC, 0.1 conv)", "vgg", VGG_BN_DO, base="E5"),
    exp("E7", "+ Data Augmentation", "vgg", VGG_BN_DO, base="E6", **AUG),
    # Otimização (a partir de E7)
    exp("E8", "SGD+Momentum 0.9 (lr 0.05)", "vgg", VGG_BN_DO, base="E7", **AUG, optimizer="sgd_momentum", lr=0.05),
    exp("E9", "SGD sem momentum (lr 0.05)", "vgg", VGG_BN_DO, base="E8", **AUG, optimizer="sgd", lr=0.05),
    exp("E10", "+ LR decay (÷10 no platô)", "vgg", VGG_BN_DO, base="E7", **AUG, scheduler="plateau"),
    exp("E11", "+ OneCycle (warmup + cosseno)", "vgg", VGG_BN_DO, base="E7", **{**AUG, "patience": 40},
        scheduler="onecycle"),
    exp("E12", "lr alta (1e-2)", "vgg", VGG_BN_DO, base="E7", **AUG, lr=1e-2),
    exp("E13", "lr baixa (1e-4)", "vgg", VGG_BN_DO, base="E7", **AUG, lr=1e-4),
    exp("E14", "batch 32", "vgg", VGG_BN_DO, base="E7", **AUG, batch_size=32),
    exp("E15", "batch 512", "vgg", VGG_BN_DO, base="E7", **AUG, batch_size=512),
    # Capacidade (a partir de E11)
    exp("E16", "mais filtros (64-128-256)", "vgg", {**VGG_BN_DO, "channels": (64, 128, 256)},
        base="E11", **{**AUG, "patience": 40}, scheduler="onecycle"),
    exp("E17", "menos camadas (1 conv/bloco)", "vgg", {**VGG_BN_DO, "convs_per_block": 1},
        base="E11", **{**AUG, "patience": 40}, scheduler="onecycle"),
]
BY_ID = {e["id"]: e for e in EXPERIMENTS}


def run_experiment(e, data, device, redo=False):
    """Treina o experimento `e`, ou carrega o resultado salvo em outputs/results/ se ele existir e redo=False."""
    path = RESULTS_DIR / f"{e['id']}.json"
    if path.exists() and not redo:
        r = json.loads(path.read_text())
        print(f"{e['id']}: carregado do disco | val acc {r['val_acc']:.4f} | teste acc {r['test_acc']:.4f}")
        return r
    model = build_model(e["arch"], **e["model_kw"])
    print(f"\n=== {e['id']}: {e['desc']}  ({count_params(model):,} parâmetros)")
    model, norm, hist = train(model, data, e["train"], device)
    test_loss, test_acc = evaluate(model, *(t.to(device) for t in data["test"]), norm)
    b = hist["best_epoch"] - 1
    r = dict(id=e["id"], desc=e["desc"], arch=e["arch"], base=e["base"], model_kw=e["model_kw"],
             train_cfg=e["train"], params=count_params(model),
             best_epoch=hist["best_epoch"], epochs_run=hist["epochs_run"],
             train_loss=hist["train_loss"][b], train_acc=hist["train_acc"][b],
             val_loss=hist["val_loss"][b], val_acc=hist["val_acc"][b],
             test_loss=test_loss, test_acc=test_acc,
             sec_per_epoch=float(np.mean(hist["time"])), history=hist, normalizer=norm.state_dict())
    path.write_text(json.dumps(r, indent=1))
    torch.save({"arch": e["arch"], "model_kw": e["model_kw"], "state_dict": model.state_dict(),
                "normalizer": norm.state_dict()}, WEIGHTS_DIR / f"{e['id']}.pt")
    print(f"--> {e['id']} melhor época {hist['best_epoch']}: val acc {r['val_acc']:.4f} | teste acc {test_acc:.4f}")
    return r


def load_results():
    """Resultados salvos dos 17 experimentos, na ordem do registro."""
    return {e["id"]: json.loads((RESULTS_DIR / f"{e['id']}.json").read_text()) for e in EXPERIMENTS}


# Checkpoints
def checkpoint_path(name):
    """Caminho local de um checkpoint (ex.: "best_model", "E7"); baixa do Hugging Face se não existir."""
    path = WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        import shutil

        from huggingface_hub import hf_hub_download

        remote = "best_model.pt" if name == "best_model" else f"ablations/{name}.pt"
        shutil.copy(hf_hub_download(HF_REPO, remote), path)
    return path


def load_model(name, device):
    """Carrega um checkpoint e devolve (modelo em modo eval, normalizador, dicionário do checkpoint)."""
    ckpt = torch.load(checkpoint_path(name), map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], **ckpt["model_kw"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    norm = Normalizer(ckpt["normalizer"]["mean"], ckpt["normalizer"]["std"], device)
    return model, norm, ckpt
