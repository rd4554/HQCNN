import torch
import torch.nn as nn

from src.data import load_data, make_loaders
from src.model import HQCNN


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using:", device)


X_train, y_train, X_val, y_val, X_test, y_test = load_data()

train_loader, val_loader, test_loader = make_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    batch_size=64
)


model = HQCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(10):

    model.train()

    total_loss = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/10 "
        f"Loss: {total_loss:.4f}"
    )