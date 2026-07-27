# python3 eth_mnist_merge.py --method mse --n_compression 28

import argparse
import os
from time import time as t
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
#bmatplotlib.use('TkAgg')  # Use the TkAgg backend
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import random
import re

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from torchmetrics import PearsonCorrCoef

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--method", dest="method", default="mse")
parser.add_argument("--n_compression", dest="n_compression", default="0")

parser.set_defaults(plot=False, gpu=True, train=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
method = args.method
n_compression = int(args.n_compression)

# Set the random seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
# print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity


# Function to modify label assignment
def modify_label_assignment(assignments, proportions, indices_to_remove):
    """
    Removes specified elements from label assignments.
    This version supports asymmetric merged models and keeps the mask on the same device.
    """
    mask = torch.ones(assignments.size(0), dtype=torch.bool, device=assignments.device)
    if len(indices_to_remove) > 0:
        mask[indices_to_remove] = False
    assignments = assignments[mask]
    proportions = proportions[mask, :]
    return assignments, proportions


# Function to build the network
def build_network(n_neurons):
    return DiehlAndCook2015(
        n_inpt=784,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        norm=78.4,
        theta_plus=theta_plus,
        inpt_shape=(1, 28, 28),
    )

# Function to extract n_neurons from the filename
def extract_n_neurons(filename):
    match = re.search(r'.*_\d+_(\d+)_neurons_\d+_to_\d+\.pth', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match the expected pattern")

# Initialize the main network with neurons extracted from the first model file
MODELS_FOLDER = Path("models_to_merge")
model_files = sorted([f for f in os.listdir(MODELS_FOLDER) if f.startswith('model') and f.endswith('.pth')])

if not model_files:
    raise ValueError("No model files found in the specified folder")

# Extract n_neurons from any model file (since we assume they are the same for all files)
n_neurons = extract_n_neurons(model_files[0])
network = build_network(n_neurons)

# Load the first model and its assignments and proportions
first_model_file = model_files.pop(0)
MODEL_SAVE_PATH = MODELS_FOLDER / first_model_file

# Load the state dictionary instead of the tensor
state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device))
network.load_state_dict(state_dict)

# Load the label assignments for the first model
assignments_file = first_model_file.replace('model_', 'assignments_').replace('.pth', '.pth')
proportions_file = first_model_file.replace('model_', 'proportions_').replace('.pth', '.pth')
assignments = torch.load(MODELS_FOLDER / assignments_file, map_location=torch.device(device))
proportions = torch.load(MODELS_FOLDER / proportions_file, map_location=torch.device(device))


start_merge = t()
# Merge additional models
for model_file in model_files:
    # Extract n_neurons from each model file.
    # This allows asymmetric model sizes such as 200 + 100 + 50 neurons.
    model_n_neurons = extract_n_neurons(model_file)
    new_network = build_network(model_n_neurons)

    # Load the model state
    MODEL_SAVE_PATH = MODELS_FOLDER / model_file
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device(device))
    new_network.load_state_dict(state_dict)

    # Load label assignments
    assignments_file = model_file.replace('model_', 'assignments_').replace('.pth', '.pth')
    proportions_file = model_file.replace('model_', 'proportions_').replace('.pth', '.pth')
    assi = torch.load(MODELS_FOLDER / assignments_file, map_location=torch.device(device))
    prop = torch.load(MODELS_FOLDER / proportions_file, map_location=torch.device(device))

    # Concatenate network
    network.merge_model(new_network)

    # Merge label assignments
    assignments = torch.cat((assignments, assi), 0)
    proportions = torch.cat((proportions, prop), 0)

# Get weight for network
weight = network.X_to_Ae.w.transpose(0, 1)


# Function to calculate Manhattan distance
def manhattan(w1, w2):
  return torch.sum( torch.abs(w1-w2) )

# Function to calculate Mean Squared Error
MSELoss = torch.nn.MSELoss()

# Function to calculate Cosine similarity
cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)

# Function to calculate Pearson Correlation Coefficient
pearson = PearsonCorrCoef()

if method == "manhattan":
    sim = manhattan
elif method == "mse":
    sim = MSELoss
elif method == "cos":
    sim = cos
elif method == "corr":
    sim = pearson

# Store all the iterations of pairs
pairs = []
for i in range(0, network.n_neurons-1):
    for j in range(i+1, network.n_neurons):
        w1 = weight[i]
        w2 = weight[j]
        pairs.append( (i, j, sim(w1, w2)) )


# Sort pairs by similarity
if method == "manhattan" or method == "mse":
    pairs.sort(key = lambda x:x[2])
elif method == "cos" or method == "corr":
    pairs.sort(key = lambda x:x[2], reverse=True)


# Get indices of neurons to be removed from network
removes = set()
for i, j, _ in pairs:
    if len(removes) == n_compression or n_compression == 0:
        break
    removes.add(j)
removes = list(removes)


# Remove neurons from network
network.reduce_neurons(removes)

# Modify label assignment for model 2 accordingly
assignments, proportions = modify_label_assignment(assignments, proportions, removes)

# Adjust number of neurons
n_neurons = network.n_neurons
print(n_neurons)

end_merge = t()
print(end_merge-start_merge)

# Directs network to GPU
if gpu:
    network.to("cuda")

### Test merged model
# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)


# Neuron assignments and spike proportions.
n_classes = 10
# assignments = -torch.ones(n_neurons, device=device)
# proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")


# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None


# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)


# print("\nBegin testing\n")
network.train(mode=False)
start = t()


pbar = tqdm(total=n_test)

# Start measuring MNIST evaluation time.
evaluation_start = t()

for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

# Finish measuring MNIST evaluation time.
evaluation_end = t()
evaluation_time = evaluation_end - evaluation_start

acc = (accuracy["all"] / n_test)
print(
    f"Method: {method}, "
    f"n_compression: {n_compression}, "
    f"Accuracy: {acc}, "
    f"Merging time: {(end_merge - start_merge):.1f}, "
    f"Evaluation time: {evaluation_time:.2f} s"
)

"""
# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = f"model_mse_0_compressed_3K_each.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
torch.save(obj=network.state_dict(), # only saving the state_dict() only saves the learned parameters
        f=MODEL_SAVE_PATH)
torch.save(assignments, f"models/assignments_mse_0_compressed_3K_each.pth")
torch.save(proportions, f"models/proportions_mse_0_compressed_3K_each.pth")
"""
