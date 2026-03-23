from sklearn.datasets import fetch_openml
from tqdm import trange
import numpy as np
import random
import json
import os

# Set seeds for reproducibility
random.seed(1)
np.random.seed(1)

NUM_USERS = 100  
NUM_LABELS = 3  # Each user gets 3 types of digits

# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'

for path in [train_path, test_path]:
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 1. Get MNIST data using modern fetcher
print("Fetching MNIST data...")
mnist = fetch_openml('mnist_784', version=1, data_home='./data', as_frame=False, parser='liac-arff')

# 2. CRITICAL: Convert labels to integers (OpenML returns strings by default)
mnist.target = mnist.target.astype(np.int64)

# 3. Normalize data (mean=0, std=1)
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu) / (sigma + 0.001)

# 4. Group data by label (0-9)
mnist_data = []
for i in trange(10, desc="Grouping by label"):
    idx = (mnist.target == i)
    mnist_data.append(mnist.data[idx])

print("\nSamples per label:", [len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)

# Phase 1: Assign initial 10 samples per label to each user
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):
        l = (user + j) % 10
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += [int(l)] * 10
        idx[l] += 10

# Phase 2: Assign remaining samples using Power Law / Random distribution
# We use a simpler, more robust random distribution to avoid negative math errors
for user in trange(NUM_USERS, desc="Assigning remaining samples"):
    for j in range(NUM_LABELS):
        l = (user + j) % 10
        
        # Calculate a random number of additional samples
        num_samples = int(np.random.lognormal(0, 1.0)) 
        numran1 = random.randint(10, 200)
        numran2 = random.randint(1, 5)
        total_to_take = (num_samples * numran2) + numran1
        
        # Check how many are actually left for this digit
        available = len(mnist_data[l]) - idx[l]
        
        # Safety Check: Only proceed if we have samples left
        if available > 0:
            take = min(total_to_take, available)
            X[user] += mnist_data[l][idx[l] : idx[l] + take].tolist()
            y[user] += [int(l)] * take
            idx[l] += take

print("Final index counts (Samples used per digit):", idx)

# Create final data structure for Federated Learning
train_data = {'users': [], 'user_data': {}, 'num_samples': []}
test_data = {'users': [], 'user_data': {}, 'num_samples': []}

for i in trange(NUM_USERS, desc="Shuffling and splitting Train/Test"):
    uname = 'f_{0:05d}'.format(i)
    
    # Shuffle user's local data
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)
    
    # 75/25 Train-Test Split
    num_samples = len(X_shuffled)
    train_len = int(0.75 * num_samples)
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X_shuffled[:train_len], 'y': y_shuffled[:train_len]}
    train_data['num_samples'].append(train_len)
    
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X_shuffled[train_len:], 'y': y_shuffled[train_len:]}
    test_data['num_samples'].append(num_samples - train_len)

# Save to JSON
print(f"Saving to {train_path}...")
with open(train_path, 'w') as outfile:
    json.dump(train_data, outfile)

with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print(f"Finish! Total training samples: {sum(train_data['num_samples'])}")