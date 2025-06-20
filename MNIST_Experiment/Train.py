import time, random
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

# we first set a random seed for reproducibility
seed = 0
random.seed(seed)
torch.manual_seed(seed)

# then we define the LeNet model with dropout
model = nn.Sequential(
    nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120), nn.ReLU(),
    nn.Dropout(0.5), 
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10)
)

# we then load the MNIST data 
train_dl = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True, num_workers=0
)

# we define the optimizer and learning rate schedule
opt = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
total_steps = 1000 * len(train_dl)
sched = optim.lr_scheduler.LambdaLR(opt, lambda it: (1 - it / total_steps) ** 0.75)

# we define the loss function
loss_fn = nn.CrossEntropyLoss()

# we train the model for 1000 epochs
start = time.time()
for epoch in range(1, 1001):
    running = 0.0
    # tqdm shows a progress bar for each batch in the current epoch
    for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch}/1000", unit="batch", colour="green"):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item() * xb.size(0)
    print(f"  Avg loss: {running / len(train_dl.dataset):.4f}")

# we save the training model weights
torch.save(model.state_dict(), "lenet_mc1.pth")
