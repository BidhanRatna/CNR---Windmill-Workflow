import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_dct
import torchvision
import torchvision.transforms as transforms
from foolbox import PyTorchModel
from foolbox.attacks.base import get_criterion
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


class QueryCountingModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.query_count = 0

    def reset_queries(self):
        self.query_count = 0

    def forward(self, x):
        self.query_count += x.shape[0]
        return self.model(x)


class AttackArgs:
    def __init__(self, side_length=224, max_queries=10000):
        self.side_length = side_length
        self.max_queries = max_queries
        self.ratio_mask = 0.1
        self.dim_num = 1
        self.max_iter_num_in_2d = 2
        self.init_alpha = np.pi / 2
        self.plus_learning_rate = 0.01
        self.minus_learning_rate = 0.0005
        self.half_range = 0.1
        self.boundary_search_steps = 15


class InceptionV3Small(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inception = models.inception_v3(
            weights=None,
            aux_logits=False
        )
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.inception(x)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["vgg16"])
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=2)
    parser.add_argument("--pgd-steps", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--defended-dir", default="")
    parser.add_argument("--checkpoints-dir", default="")
    return parser.parse_args()


ARGS = parse_args()

ROOT = Path(__file__).resolve().parent

DATA_DIR = (
    Path(ARGS.data_dir)
    if ARGS.data_dir
    else ROOT / "data"
)

MODELS_DIR = (
    Path(ARGS.models_dir)
    if ARGS.models_dir
    else ROOT / "models"
)

RESULTS_DIR = (
    Path(ARGS.results_dir)
    if ARGS.results_dir
    else ROOT / "results"
)

DEFENDED_DIR = (
    Path(ARGS.defended_dir)
    if ARGS.defended_dir
    else ROOT / "defended_models"
)

CKPT_DIR = (
    Path(ARGS.checkpoints_dir)
    if ARGS.checkpoints_dir
    else ROOT / "checkpoints"
)

for directory in [
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    DEFENDED_DIR,
    CKPT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

NUM_SAMPLES = ARGS.samples
MAX_QUERIES = ARGS.max_queries
SEED = ARGS.seed
RMSE_THRESHOLDS = [0.1, 0.05, 0.01]
AT_EPOCHS = ARGS.epochs
BASELINE_EPOCHS = ARGS.baseline_epochs
PGD_STEPS = ARGS.pgd_steps
AT_BATCH = ARGS.batch_size

AT_CONFIG = {
    "vgg16": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "lr": 0.01,
    },
    "vgg19": {
        "epsilon": 4 / 255,
        "alpha": 1 / 255,
        "lr": 0.01,
    },
    "densenet121": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "lr": 0.01,
    },
    "inception_v3": {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "lr": 0.01,
    },
}

MODEL_FILES = {
    "vgg16": "mnist_vgg16.pth",
    "vgg19": "mnist_vgg19.pth",
    "densenet121": "mnist_densenet121.pth",
    "inception_v3": "mnist_inception-v3.pth",
}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def compute_rmse(a, b):
    return torch.sqrt(
        torch.mean((b - a).float() ** 2)
    ).item()


def rotate_in_2d(v, d, theta=np.pi / 8):
    denom = torch.sum(v * v)

    if denom < 1e-10:
        return v

    orth = d - torch.sum(v * d) / denom * v
    on = torch.norm(orth, p=2)

    if on < 1e-10:
        return v

    r = (
        v * np.cos(theta)
        + torch.norm(v, p=2)
        / on
        * orth
        * np.sin(theta)
    )

    n = torch.norm(r)

    if n < 1e-10:
        return v

    return r / n * torch.norm(v)


def get_orthogonal_1d_in_subspace(
    args,
    v,
    n,
    ratio=0.3,
    if_left=1,
):
    m = torch.zeros(
        args.side_length,
        args.side_length,
        device=device,
    )

    s = int(args.side_length * ratio)

    if if_left:
        m[:s, :s] = 1
    else:
        m[-s:, -s:] = 1

    xs, ys = torch.where(m == 1)

    n = min(n, len(xs))

    masks = []

    for _ in range(3):
        sel = np.random.choice(
            len(xs),
            size=n,
            replace=False,
        )

        mi = torch.zeros_like(m)
        mi[xs[sel], ys[sel]] = 1

        masks.append(
            mi.reshape(
                -1,
                args.side_length,
                args.side_length,
            )
        )

    mask = torch.cat(
        masks,
        0,
    ).expand(
        1,
        3,
        args.side_length,
        args.side_length,
    )

    mask = mask * torch.randn_like(
        mask,
        device=device,
    )

    direction = rotate_in_2d(
        v,
        mask,
        theta=np.pi / 2,
    )

    norm = torch.norm(direction, p=2)

    if norm < 1e-10:
        return v, mask

    return (
        direction / norm * torch.norm(v, p=2),
        mask,
    )


class TriangleAttack:
    def __init__(self, fmodel, counter, args):
        self.net = fmodel
        self.counter = counter
        self.args = args
        self.alpha = args.init_alpha

    def _query(self, x):
        return torch.max(
            self.net(x),
            1,
        )[1]

    def _ok(self):
        return (
            self.counter.query_count
            < self.args.max_queries
        )

    def _get_x_hat_in_2d(
        self,
        x_o,
        x_adv,
        ax1,
        ax2,
        lbl,
    ):
        upper = (
            np.pi / 2
            + self.args.half_range
        )

        lower = (
            np.pi / 2
            - self.args.half_range
        )

        d = torch.norm(
            x_adv - x_o,
            p=2,
        )

        if d < 1e-10:
            return (
                torch_dct.idct_2d(x_adv),
                False,
            )

        x_hat = torch_dct.idct_2d(x_adv)

        theta = (
            max(
                np.pi - 2 * self.alpha,
                0,
            )
            + min(
                np.pi / 16,
                self.alpha / 2,
            )
        )

        def safe_sin(a):
            v = np.sin(a)

            return (
                max(abs(v), 1e-10)
                * (1 if v >= 0 else -1)
            )

        def cand(th, sign):
            x = (
                x_o
                + d
                * (
                    ax1 * np.cos(th)
                    + sign * ax2 * np.sin(th)
                )
                / safe_sin(self.alpha)
                * np.sin(
                    self.alpha + th
                )
            )

            return torch.clamp(
                torch_dct.idct_2d(x),
                0,
                1,
            )

        right_theta = (
            np.pi - self.alpha
        )

        if not self._ok():
            return x_hat, False

        t = cand(theta, 1)

        if self._query(t) != lbl:
            x_hat = t
            flag = 1
            left_theta = theta

        else:
            self.alpha = max(
                lower,
                self.alpha
                - self.args.minus_learning_rate,
            )

            theta = max(
                theta,
                np.pi
                - 2 * self.alpha
                + np.pi / 64,
            )

            if not self._ok():
                return x_hat, False

            t = cand(theta, -1)

            if self._query(t) != lbl:
                x_hat = t
                flag = -1
                left_theta = theta

            else:
                self.alpha = max(
                    lower,
                    self.alpha
                    - self.args.minus_learning_rate,
                )

                return x_hat, False

        for _ in range(
            self.args.max_iter_num_in_2d
        ):
            if not self._ok():
                break

            theta = (
                left_theta
                + right_theta
            ) / 2

            t = cand(
                theta,
                flag,
            )

            if self._query(t) != lbl:
                self.alpha = min(
                    upper,
                    self.alpha
                    + self.args.plus_learning_rate,
                )

                return t, True

            self.alpha = max(
                lower,
                self.alpha
                - self.args.minus_learning_rate,
            )

            flag = -flag

            theta = max(
                theta,
                np.pi
                - 2 * self.alpha
                + np.pi / 64,
            )

            if not self._ok():
                break

            t = cand(
                theta,
                flag,
            )

            if self._query(t) != lbl:
                self.alpha = min(
                    upper,
                    self.alpha
                    + self.args.plus_learning_rate,
                )

                return t, True

            self.alpha = max(
                lower,
                self.alpha
                - self.args.minus_learning_rate,
            )

            right_theta = theta

        self.alpha = min(
            upper,
            self.alpha
            + self.args.plus_learning_rate,
        )

        return x_hat, True

    def attack(
        self,
        x_orig,
        label,
        rmse_target=0.01,
    ):
        self.alpha = (
            self.args.init_alpha
        )

        self.counter.reset_queries()

        fail = (
            x_orig,
            self.counter.query_count,
            0.0,
            0.0,
            0.0,
            False,
            {},
        )

        if self._query(x_orig) != label:
            return fail

        try:
            x_adv = (
                LinearSearchBlendedUniformNoiseAttack(
                    steps=100
                ).run(
                    self.net,
                    x_orig,
                    get_criterion(
                        label.long()
                    ),
                    early_stop=None,
                )
            )
        except Exception:
            return fail

        if self._query(x_adv) == label:
            return fail

        lo = 0.0
        hi = 1.0

        for _ in range(
            self.args.boundary_search_steps
        ):
            if not self._ok():
                break

            mid = (lo + hi) / 2

            x_mid = torch.clamp(
                (1 - mid) * x_orig
                + mid * x_adv,
                0,
                1,
            )

            if self._query(x_mid) != label:
                hi = mid
            else:
                lo = mid

        x_adv = torch.clamp(
            (1 - hi) * x_orig
            + hi * x_adv,
            0,
            1,
        )

        best_adv = x_adv.clone()

        best_rmse = compute_rmse(
            x_orig,
            best_adv,
        )

        snaps = {
            self.counter.query_count:
            best_rmse
        }

        if best_rmse < rmse_target:
            return (
                best_adv,
                self.counter.query_count,
                best_rmse,
                torch.norm(
                    best_adv - x_orig,
                    p=2,
                ).item(),
                torch.norm(
                    best_adv - x_orig,
                    p=float("inf"),
                ).item(),
                True,
                snaps,
            )

        while (
            self.counter.query_count
            < self.args.max_queries
        ):
            try:
                diff = (
                    x_adv - x_orig
                )

                if torch.norm(
                    diff,
                    p=2,
                ) < 1e-10:
                    break

                v = torch_dct.dct_2d(
                    diff
                )

            except Exception:
                break

            norm = torch.norm(v)

            if norm < 1e-10:
                break

            ax1 = v / norm

            direction, _ = (
                get_orthogonal_1d_in_subspace(
                    self.args,
                    v,
                    self.args.dim_num,
                    self.args.ratio_mask,
                    True,
                )
            )

            dn = torch.norm(
                direction
            )

            if dn < 1e-10:
                break

            ax2 = (
                direction / dn
            )

            x_adv, is_adv = (
                self._get_x_hat_in_2d(
                    torch_dct.dct_2d(
                        x_orig
                    ),
                    torch_dct.dct_2d(
                        x_adv
                    ),
                    ax1,
                    ax2,
                    label,
                )
            )

            cur_rmse = compute_rmse(
                x_orig,
                x_adv,
            )

            if (
                is_adv
                and cur_rmse
                < best_rmse
            ):
                best_adv = (
                    x_adv.clone()
                )

                best_rmse = cur_rmse

            snaps[
                self.counter.query_count
            ] = best_rmse

            if (
                best_rmse
                < rmse_target
            ):
                break

        queries = (
            self.counter.query_count
        )

        if (
            self._query(best_adv)
            != label
        ).item():
            return (
                best_adv,
                queries,
                best_rmse,
                torch.norm(
                    best_adv - x_orig,
                    p=2,
                ).item(),
                torch.norm(
                    best_adv - x_orig,
                    p=float("inf"),
                ).item(),
                True,
                snaps,
            )

        return (
            x_orig,
            queries,
            float("inf"),
            0.0,
            0.0,
            False,
            snaps,
        )


def build_model(name):
    if name == "vgg16":
        model = models.vgg16(
            weights=None
        )

        model.classifier[6] = (
            nn.Linear(
                4096,
                10,
            )
        )

    elif name == "vgg19":
        model = models.vgg19(
            weights=None
        )

        model.classifier[6] = (
            nn.Linear(
                4096,
                10,
            )
        )

    elif name == "densenet121":
        model = models.densenet121(
            weights=None
        )

        model.classifier = (
            nn.Linear(
                model.classifier.in_features,
                10,
            )
        )

    elif name == "inception_v3":
        model = InceptionV3Small(
            10
        )

    else:
        raise ValueError(
            f"Unsupported model: {name}"
        )

    return model


def get_image_size(model_name):
    return (
        75
        if model_name
        == "inception_v3"
        else 32
    )


def load_mnist(img_size):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    img_size,
                    img_size,
                )
            ),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ]
    )

    trainset = (
        torchvision.datasets.MNIST(
            DATA_DIR,
            train=True,
            download=True,
            transform=transform,
        )
    )

    testset = (
        torchvision.datasets.MNIST(
            DATA_DIR,
            train=False,
            download=True,
            transform=transform,
        )
    )

    return trainset, testset


def load_original_weights(
    model,
    model_name,
):
    path = (
        MODELS_DIR
        / MODEL_FILES[model_name]
    )

    if not path.exists():
        print(
            f"Weights not found: {path}"
        )

        return False

    state_dict = torch.load(
        path,
        map_location=device,
    )

    if isinstance(
        state_dict,
        dict,
    ):
        state_dict = (
            state_dict.get(
                "state_dict",
                state_dict.get(
                    "model_state_dict",
                    state_dict,
                ),
            )
        )

    new_state_dict = {}

    for key, value in (
        state_dict.items()
    ):
        key = (
            key[7:]
            if key.startswith("module.")
            else key
        )

        if (
            model_name
            == "inception_v3"
            and not key.startswith(
                "inception."
            )
        ):
            key = (
                "inception."
                + key
            )

        new_state_dict[key] = value

    model.load_state_dict(
        new_state_dict,
        strict=False,
    )

    print(
        f"Loaded weights: {path.name}"
    )

    return True


def train_baseline(
    model,
    train_loader,
    model_name,
):
    model = model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    print(
        f"Training baseline model: "
        f"{model_name}"
    )

    for epoch in range(
        BASELINE_EPOCHS
    ):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Baseline "
                f"{epoch + 1}/"
                f"{BASELINE_EPOCHS}"
            ),
        )

        for images, labels in pbar:
            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = F.cross_entropy(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predicted = (
                outputs.argmax(1)
            )

            total += labels.size(0)

            correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )

            pbar.set_postfix(
                {
                    "Loss":
                    f"{loss.item():.4f}",
                    "Acc":
                    f"{100.0 * correct / total:.1f}%",
                }
            )

        accuracy = (
            100.0
            * correct
            / total
        )

        print(
            f"Baseline epoch "
            f"{epoch + 1}: "
            f"Loss="
            f"{total_loss / total:.4f} | "
            f"Accuracy="
            f"{accuracy:.1f}%"
        )

    save_path = (
        MODELS_DIR
        / MODEL_FILES[model_name]
    )

    torch.save(
        model.state_dict(),
        save_path,
    )

    print(
        f"Baseline model saved: "
        f"{save_path}"
    )

    return model


def pgd_attack(
    model,
    images,
    labels,
    epsilon,
    alpha,
):
    adv = (
        images.clone()
        .detach()
    )

    adv = (
        adv
        + torch.empty_like(
            adv
        ).uniform_(
            -epsilon,
            epsilon,
        )
    )

    adv = torch.clamp(
        adv,
        0,
        1,
    ).detach()

    for _ in range(PGD_STEPS):
        adv.requires_grad = True

        outputs = model(adv)

        loss = F.cross_entropy(
            outputs,
            labels,
        )

        grad = torch.autograd.grad(
            loss,
            adv,
        )[0]

        adv = (
            adv.detach()
            + alpha
            * grad.sign()
        )

        delta = torch.clamp(
            adv - images,
            -epsilon,
            epsilon,
        )

        adv = torch.clamp(
            images + delta,
            0,
            1,
        ).detach()

    return adv


def adversarial_training(
    model,
    train_loader,
    model_name,
):
    cfg = AT_CONFIG[
        model_name
    ]

    epsilon = cfg[
        "epsilon"
    ]

    alpha = cfg[
        "alpha"
    ]

    lr = cfg[
        "lr"
    ]

    print("=" * 60)
    print(
        f"AT DEFENSE: "
        f"mnist x {model_name}"
    )
    print(
        f"Epochs={AT_EPOCHS}, "
        f"lr={lr}, "
        f"epsilon={epsilon:.4f}, "
        f"alpha={alpha:.4f}"
    )
    print("=" * 60)

    ckpt_path = (
        CKPT_DIR
        / (
            f"AT_mnist_"
            f"{model_name}"
            f"_ckpt.pth"
        )
    )

    start_epoch = 0
    best_robust_acc = 0
    best_state = None

    model = model.to(
        device
    )

    if ckpt_path.exists():
        checkpoint = torch.load(
            ckpt_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint[
                "model_state"
            ]
        )

        start_epoch = (
            checkpoint["epoch"]
        )

        best_robust_acc = (
            checkpoint[
                "best_robust_acc"
            ]
        )

        best_state = (
            checkpoint[
                "best_state"
            ]
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    scheduler = (
        optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=AT_EPOCHS,
        )
    )

    for _ in range(
        start_epoch
    ):
        scheduler.step()

    for epoch in range(
        start_epoch,
        AT_EPOCHS,
    ):
        model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/"
                f"{AT_EPOCHS}"
            ),
        )

        for images, labels in pbar:
            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            model.eval()

            adv_images = pgd_attack(
                model,
                images,
                labels,
                epsilon,
                alpha,
            )

            model.train()

            optimizer.zero_grad()

            outputs = model(
                adv_images
            )

            loss = F.cross_entropy(
                outputs,
                labels,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predicted = (
                outputs.argmax(1)
            )

            total += (
                labels.size(0)
            )

            correct += (
                predicted.eq(
                    labels
                )
                .sum()
                .item()
            )

            pbar.set_postfix(
                {
                    "Loss":
                    f"{loss.item():.4f}",
                    "Acc":
                    f"{100.0 * correct / total:.1f}%",
                }
            )

        scheduler.step()

        train_acc = (
            100.0
            * correct
            / total
        )

        model.eval()

        robust_correct = 0
        robust_total = 0

        for index, (
            images,
            labels,
        ) in enumerate(
            train_loader
        ):
            if index >= 3:
                break

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )

            adv_images = pgd_attack(
                model,
                images,
                labels,
                epsilon,
                alpha,
            )

            with torch.no_grad():
                predictions = (
                    model(
                        adv_images
                    ).argmax(1)
                )

                robust_total += (
                    labels.size(0)
                )

                robust_correct += (
                    predictions.eq(
                        labels
                    )
                    .sum()
                    .item()
                )

        robust_acc = (
            100.0
            * robust_correct
            / robust_total
            if robust_total > 0
            else 0
        )

        print(
            f"Epoch {epoch + 1}: "
            f"Loss="
            f"{total_loss / total:.4f} | "
            f"Train="
            f"{train_acc:.1f}% | "
            f"Robust="
            f"{robust_acc:.1f}%"
        )

        if (
            train_acc < 15.0
            and epoch >= 3
        ):
            print(
                "Model collapsed — "
                "stopping early"
            )

            break

        if (
            robust_acc
            > best_robust_acc
        ):
            best_robust_acc = (
                robust_acc
            )

            best_state = (
                copy.deepcopy(
                    model.state_dict()
                )
            )

        torch.save(
            {
                "epoch":
                epoch + 1,
                "model_state":
                model.state_dict(),
                "best_state":
                best_state,
                "best_robust_acc":
                best_robust_acc,
            },
            ckpt_path,
        )

    save_path = (
        DEFENDED_DIR
        / (
            f"AT_mnist_"
            f"{model_name}.pth"
        )
    )

    if best_state is not None:
        torch.save(
            best_state,
            save_path,
        )

        model.load_state_dict(
            best_state
        )

    else:
        torch.save(
            model.state_dict(),
            save_path,
        )

    print(
        f"AT model saved: "
        f"{save_path} "
        f"(robust acc: "
        f"{best_robust_acc:.1f}%)"
    )

    if ckpt_path.exists():
        ckpt_path.unlink()

    return model


def wrap_for_attack(
    raw_model,
):
    raw_model = (
        raw_model
        .to(device)
        .eval()
    )

    counter = (
        QueryCountingModel(
            raw_model
        )
        .to(device)
        .eval()
    )

    preprocessing = {
        "mean":
        torch.tensor(
            [
                0.485,
                0.456,
                0.406,
            ]
        ).to(device),
        "std":
        torch.tensor(
            [
                0.229,
                0.224,
                0.225,
            ]
        ).to(device),
        "axis":
        -3,
    }

    return (
        PyTorchModel(
            counter,
            bounds=(0, 1),
            preprocessing=preprocessing,
        ),
        counter,
    )


def run_triangle_attack(
    fmodel,
    counter,
    images,
    labels,
    model_size,
    model_name,
):
    print(
        f"Triangle Attack vs AT | "
        f"{model_name} on mnist"
    )

    print(
        f"Samples: {NUM_SAMPLES} | "
        f"Query Budget: "
        f"{MAX_QUERIES}"
    )

    counter.reset_queries()

    indexes = []

    with torch.no_grad():
        for index in range(
            len(images)
        ):
            prediction = (
                fmodel(
                    images[
                        index:
                        index + 1
                    ]
                )
                .argmax(1)
                .item()
            )

            if (
                prediction
                == labels[index].item()
            ):
                indexes.append(
                    index
                )

            if (
                len(indexes)
                >= NUM_SAMPLES
            ):
                break

    counter.reset_queries()

    if not indexes:
        print(
            "No correctly classified "
            "samples found."
        )

        return None

    print(
        f"{len(indexes)} correctly "
        f"classified samples"
    )

    attack_images = torch.stack(
        [
            images[index]
            for index in indexes
        ]
    )

    attack_labels = torch.stack(
        [
            labels[index]
            for index in indexes
        ]
    )

    attack = TriangleAttack(
        fmodel,
        counter,
        AttackArgs(
            side_length=model_size,
            max_queries=MAX_QUERIES,
        ),
    )

    records = []

    start_time = time.time()

    for index in range(
        len(attack_images)
    ):
        sample_start = time.time()

        try:
            (
                x_adv,
                queries,
                rmse,
                l2,
                linf,
                ok,
                snaps,
            ) = attack.attack(
                attack_images[
                    index:
                    index + 1
                ],
                attack_labels[
                    index:
                    index + 1
                ],
                rmse_target=0.01,
            )

        except Exception as error:
            print(
                f"[{index + 1}] "
                f"Error: {error}"
            )

            records.append(
                {
                    "sample_idx":
                    index + 1,
                    "success":
                    False,
                    "total_queries":
                    0,
                    "rmse":
                    float("inf"),
                    "l2_distance":
                    0.0,
                    "linf_distance":
                    0.0,
                    **{
                        (
                            f"success_at_"
                            f"{budget}_"
                            f"rmse"
                            f"{threshold}"
                        ):
                        False
                        for budget
                        in [
                            1000,
                            10000,
                        ]
                        for threshold
                        in RMSE_THRESHOLDS
                    },
                    "time":
                    0.0,
                    "defense":
                    "AT",
                }
            )

            continue

        evals = {
            (
                f"success_at_"
                f"{budget}_"
                f"rmse{threshold}"
            ):
            any(
                value < threshold
                for query_index, value
                in snaps.items()
                if query_index
                <= budget
            )
            for budget
            in [
                1000,
                10000,
            ]
            for threshold
            in RMSE_THRESHOLDS
        }

        print(
            f"{index + 1:3d}/"
            f"{len(attack_images)}: "
            f"{'pass' if ok else 'fail'} | "
            f"Q:{queries:6d} | "
            f"RMSE:{rmse:.4f} | "
            f"L2:{l2:7.3f} | "
            f"{time.time() - sample_start:.1f}s"
        )

        records.append(
            {
                "sample_idx":
                index + 1,
                "success":
                ok,
                "total_queries":
                queries,
                "rmse":
                rmse,
                "l2_distance":
                l2,
                "linf_distance":
                linf,
                **evals,
                "time":
                round(
                    time.time()
                    - sample_start,
                    2,
                ),
                "defense":
                "AT",
            }
        )

    dataframe = pd.DataFrame(
        records
    )

    successful = dataframe[
        dataframe["success"]
    ]

    csv_path = (
        RESULTS_DIR
        / (
            f"AT_mnist_"
            f"{model_name}_"
            f"TriangleAttack.csv"
        )
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    total_time = (
        time.time()
        - start_time
    )

    print(
        f"AT | mnist x "
        f"{model_name} | "
        f"Success: "
        f"{dataframe['success'].mean() * 100:.1f}% | "
        f"Time: "
        f"{total_time:.1f}s"
    )

    print(
        f"Saved: {csv_path}"
    )

    return {
        "dataset":
        "mnist",
        "model":
        model_name,
        "defense":
        "AT",
        "success_rate":
        round(
            dataframe[
                "success"
            ].mean()
            * 100,
            1,
        ),
        "avg_queries":
        (
            round(
                successful[
                    "total_queries"
                ].mean(),
                1,
            )
            if len(successful)
            else 0
        ),
        "avg_l2":
        (
            round(
                successful[
                    "l2_distance"
                ].mean(),
                4,
            )
            if len(successful)
            else 0
        ),
        "avg_linf":
        (
            round(
                successful[
                    "linf_distance"
                ].mean(),
                4,
            )
            if len(successful)
            else 0
        ),
    }


summary_path = (
    RESULTS_DIR
    / "AT_mnist_summary.csv"
)


def get_completed():
    if not summary_path.exists():
        return set()

    dataframe = pd.read_csv(
        summary_path
    )

    bad = dataframe[
        (
            dataframe[
                "success_rate"
            ]
            == 0
        )
        & (
            dataframe[
                "avg_queries"
            ]
            == 0
        )
    ]

    if len(bad) > 0:
        dataframe = (
            dataframe.drop(
                bad.index
            )
        )

        dataframe.to_csv(
            summary_path,
            index=False,
        )

    return set(
        dataframe[
            "model"
        ].tolist()
    )


def save_result(result):
    write_header = (
        not summary_path.exists()
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        summary_path,
        mode="a",
        header=write_header,
        index=False,
    )

    print(
        f"Result saved to "
        f"{summary_path}"
    )


def main():
    for model_name in ARGS.models:
        if (
            model_name
            not in MODEL_FILES
        ):
            raise ValueError(
                f"Unsupported model: "
                f"{model_name}"
            )

        completed = (
            get_completed()
        )

        if (
            model_name
            in completed
        ):
            print(
                f"SKIPPING mnist x "
                f"{model_name} — "
                f"already completed"
            )

            continue

        print("=" * 60)

        print(
            f"mnist x "
            f"{model_name}"
        )

        print("=" * 60)

        image_size = (
            get_image_size(
                model_name
            )
        )

        trainset, testset = (
            load_mnist(
                image_size
            )
        )

        train_loader = (
            DataLoader(
                trainset,
                batch_size=AT_BATCH,
                shuffle=True,
                num_workers=2,
                pin_memory=(
                    torch.cuda
                    .is_available()
                ),
            )
        )

        model = build_model(
            model_name
        )

        if not load_original_weights(
            model,
            model_name,
        ):
            print(
                "No pretrained weights found."
            )

            print(
                "Training baseline model "
                "automatically."
            )

            model = train_baseline(
                model,
                train_loader,
                model_name,
            )

        at_model = (
            adversarial_training(
                model,
                train_loader,
                model_name,
            )
        )

        sample_count = min(
            len(testset),
            NUM_SAMPLES * 5,
        )

        test_images = torch.stack(
            [
                testset[index][0]
                for index
                in range(
                    sample_count
                )
            ]
        ).to(device)

        test_labels = torch.tensor(
            [
                testset[index][1]
                for index
                in range(
                    sample_count
                )
            ]
        ).to(device)

        fmodel, counter = (
            wrap_for_attack(
                at_model
            )
        )

        result = (
            run_triangle_attack(
                fmodel,
                counter,
                test_images,
                test_labels,
                image_size,
                model_name,
            )
        )

        if result:
            save_result(
                result
            )

        del (
            model,
            at_model,
            fmodel,
            counter,
            test_images,
            test_labels,
            train_loader,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_path.exists():
        dataframe = pd.read_csv(
            summary_path
        )

        print("=" * 80)

        print(
            "MNIST AT DEFENSE — RESULTS"
        )

        print("=" * 80)

        header = (
            f"{'Model':<14} "
            f"{'Success%':>9} "
            f"{'AvgQueries':>12} "
            f"{'AvgL2':>10} "
            f"{'AvgLinf':>10}"
        )

        print(header)

        print(
            "-" * len(header)
        )

        for _, row in (
            dataframe.iterrows()
        ):
            print(
                f"{row['model']:<14} "
                f"{row['success_rate']:>8.1f}% "
                f"{row['avg_queries']:>12.1f} "
                f"{row['avg_l2']:>10.4f} "
                f"{row['avg_linf']:>10.4f}"
            )

        print("DONE!")


if __name__ == "__main__":
    main()
