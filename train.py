import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import glob
import os
import math
from tqdm import tqdm

DATA_DIR = "processed_data"
BATCH_SIZE = 4096
EPOCHS = 2
MAX_LR = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 5000

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        resid = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += resid
        return F.relu(x)

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Conv2d(14, 128, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(128)
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(10)])
        self.conv_pol = nn.Conv2d(128, 32, 1)
        self.bn_pol = nn.BatchNorm2d(32)
        self.fc_pol = nn.Linear(32 * 8 * 8, 4096)
        self.conv_val = nn.Conv2d(128, 32, 1)
        self.bn_val = nn.BatchNorm2d(32)
        self.fc_val1 = nn.Linear(32 * 8 * 8, 128)
        self.fc_val2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks: x = block(x)
        p = F.relu(self.bn_pol(self.conv_pol(x)))
        p = p.view(-1, 32 * 8 * 8)
        p = self.fc_pol(p)
        v = F.relu(self.bn_val(self.conv_val(x)))
        v = v.view(-1, 32 * 8 * 8)
        v = F.relu(self.fc_val1(v))
        v = torch.tanh(self.fc_val2(v))
        return p, v

class ChessIterableDataset(IterableDataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_iter = self.files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            file_iter = self.files[worker_id::num_workers]

        for file_path in file_iter:
            try:
                with np.load(file_path) as data:
                    x = data['x']
                    p = data['p']
                    v = data['v']

                indices = np.arange(len(x))
                np.random.shuffle(indices)

                for i in indices:
                    yield (
                        torch.from_numpy(x[i]),
                        torch.tensor(p[i], dtype=torch.long),
                        torch.tensor(v[i], dtype=torch.float32)
                    )
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

def train():
    torch.backends.cudnn.benchmark = True
    print(f"Training on {DEVICE} with Torch 2.0 optimizations...")

    dataset = ChessIterableDataset(DATA_DIR)

    total_files = len(dataset.files)
    estimated_samples = total_files * CHUNK_SIZE
    estimated_batches = estimated_samples // BATCH_SIZE
    print(f"Data: {estimated_samples} pos | {estimated_batches} batches per epoch")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=2,
                        pin_memory=True, prefetch_factor=2)

    model = ChessNet().to(DEVICE)

    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile() for speed.")
    except Exception as e:
        print(f"Could not compile model (Safe to ignore): {e}")

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=estimated_batches * EPOCHS,
        pct_start=0.1,
        div_factor=25,
        final_div_factor=100
    )

    scaler = GradScaler('cuda')
    pol_loss_fn = nn.CrossEntropyLoss()
    val_loss_fn = nn.MSELoss()

    model.train()

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        loop = tqdm(loader, total=estimated_batches, unit="batch")

        total_loss = 0
        batch_count = 0

        for x, p, v in loop:
            x = x.to(DEVICE, non_blocking=True).float()
            p = p.to(DEVICE, non_blocking=True)
            v = v.to(DEVICE, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                p_pred, v_pred = model(x)
                loss_p = pol_loss_fn(p_pred, p)
                loss_v = val_loss_fn(v_pred.squeeze(), v)
                loss = loss_p + loss_v

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            batch_count += 1

            current_lr = scheduler.get_last_lr()[0]
            loop.set_postfix(loss=loss_val, lr=f"{current_lr:.5f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"chess_ai_epoch_{epoch}.pth")

    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    torch.save(state_dict, "chess_ai_medium.pth")
    print("Training Complete. Saved 'chess_ai_medium.pth'")

if __name__ == "__main__":
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        train()
    else:
        print("Error: No data.")
