import pathlib, random
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

# we define the function to set all random seeds
def seed_all(s=0):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# we define the same model used during training
net = nn.Sequential(
    nn.Conv2d(1,6,5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(6,16,5), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16*4*4,120), nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(120,84), nn.ReLU(),
    nn.Linear(84,10)
)

# we load the trained model weights
if not pathlib.Path("lenet_mc1.pth").exists():
    raise SystemExit("No weights – run the training script first.")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load("lenet_mc.pth", map_location=dev))
net.to(dev).train()  # keep dropout ON

# we get a single digit "1" from the test set
data = datasets.MNIST('.', False, download=True, transform=transforms.ToTensor())
for img, lbl in data:
    if lbl == 1:
        one = img
        break

# we prepare to rotate the image and collect predictions
T = 100 # number of stochastic forward passes
angles = range(0, 360, 30)
logits, probs, classes = [], [], []


for ang in tqdm(angles, unit="rot"):   # tqdm shows a progress bar
    x = TF.rotate(one, ang, fill=0).unsqueeze(0).to(dev)

    # we apply MC Dropout at test time
    out = torch.stack([net(x) for _ in range(T)])      
    p   = out.softmax(-1)

    # we take the average prediction and keep top-3 distinct classes
    mean_p = p.mean(0)[0]
    top5   = mean_p.topk(5).indices

    cls = []
    for idx in top5:                     
        d = idx.item()
        if d not in cls:
            cls.append(d)
        if len(cls) == 3:
            break

    logits.append(out[:,0,cls].detach().cpu())  
    probs .append(p  [:,0,cls].detach().cpu())
    classes.append(cls)

# weplot logits and probabilities with color-coded digits
palette = plt.get_cmap("tab10", 10).colors          # 10 vivid hues

all_digits = sorted({d for fr in classes for d in fr})
style = {d: dict(c=palette[i % 10], m='o') for i, d in enumerate(all_digits)}


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
titles = ["Softmax input ", "Softmax output"]

for seq, axis, ttl in zip([logits, probs], ax, titles):
    for frame, cloud in enumerate(seq):
        for col, dig in enumerate(classes[frame]):
            axis.scatter([frame]*T,
                         cloud[:, col],
                         s=8, alpha=.3,
                         color=style[dig]['c'],
                         marker=style[dig]['m'])
    axis.set_xlabel("rotation frame (0° … 330°)")
    axis.set_title(ttl)
ax[0].set_ylabel("value")

# we then build a clean legend for all predicted classes
from matplotlib.lines import Line2D

handles = []
for d in all_digits:
    h = Line2D([], [], linestyle='',
               marker=style[d]['m'],
               markersize=6,
               markerfacecolor=style[d]['c'],
               markeredgecolor=style[d]['c'],
               label=f"class {d}")
    handles.append(h)

# put legend on the right-hand panel (probabilities)
ax[1].legend(handles=handles,
             bbox_to_anchor=(1.04, 1),
             loc="upper left",
             title="digit class")


fig.tight_layout()
fig.savefig("figure.png", dpi=200)

