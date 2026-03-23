import torch
import numpy as np
import copy
import argparse
import time

# Custom modules
from models import CNN, SVM
from datasets import get_dataloader, get_test_loader
from wireless_channel import WirelessChannel
from scheduler import FEELScheduler
from aggregator import aggregate_unbiased
from testing import test
from plotting import plot_accuracy

def run_feel_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='svm', choices=['svm', 'cnn'])
    parser.add_argument('--multi', action='store_true', help='Use multi-device scheduling')
    args = parser.parse_args()

    # --- 1. Hyperparameters & Settings (Section VI-A) ---
    if args.model_type == 'svm':
        rho = 5e-6           # [cite: 516]
        eta = 0.0001         # [cite: 510]
        M = 10 if args.multi else 1
        dataset_name = 'cifar10_svm'
        q, S = 16, 3072      # 16-bit quantization, SVM parameters [cite: 257, 472]
    else:
        rho = 5e-3           # [cite: 599]
        eta = 0.005          # [cite: 510]
        M = 3 if args.multi else 1
        dataset_name = 'mnist'
        q, S = 16, 1e5       # Estimated parameters for 6-layer CNN [cite: 480]

    K, B = 30, 1e6           # [cite: 468, 472]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Wireless Environment
    channel = WirelessChannel(K=K, B=B)
    distances = np.random.uniform(0.01, 0.5, K) # [cite: 468, 470]
    
    # Load Non-IID Data
    train_loaders = get_dataloader(dataset=dataset_name, K=K)
    test_loader = get_test_loader(dataset=dataset_name)
    train_iters = [iter(loader) for loader in train_loaders]
    
    global_model = SVM().to(device) if args.model_type == 'svm' else CNN().to(device)
    n_k = torch.tensor([len(loader.dataset) for loader in train_loaders]).float()
    n_total = n_k.sum()
    
    accuracy_history = []
    cumulative_time = []
    total_simulated_time = 0.0

    print(f"Starting {args.model_type.upper()} FEEL Experiment...")

    # --- 2. Communication Rounds ---
    for t in range(120): # 
        local_grads = []
        g_norms = []
        
        # Simulated Latency Components (Section III-D)
        # 1. Broadcast Latency (Step 1)
        snrs = channel.get_snr(distances)
        r_down = B * np.log2(1 + 10**(46/10)) # Simple downlink SNR proxy [cite: 150]
        T_broadcast = (q * S) / r_down # [cite: 259]

        # --- Steps 2 & 3: Local Gradient & Importance Reporting ---
        for k in range(K):
            # REAL Local Training to get actual update diversity
            temp_model = copy.deepcopy(global_model)
            temp_model.train()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=eta)
            
            # Fetch batch
            try:
                data, target = next(train_iters[k])
            except StopIteration:
                train_iters[k] = iter(train_loaders[k])
                data, target = next(train_iters[k])
            
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = temp_model(data)
            loss = torch.nn.functional.cross_entropy(output, target) if args.model_type == 'cnn' else torch.nn.functional.multi_margin_loss(output, target)
            loss.backward()
            
            # Extract actual gradient vector
            grads = torch.cat([p.grad.view(-1) for p in temp_model.parameters()])
            local_grads.append(grads)
            g_norms.append(torch.norm(grads)) # Importance indicator [cite: 11, 337]

        # --- Step 4: Device Selection (Scheduler) ---
        T_upload_all = torch.tensor([channel.get_latency(s, q, S) for s in snrs]) # [cite: 273]
        scheduler = FEELScheduler(rho, n_total)
        g_norms_tensor = torch.stack(g_norms).cpu()
        
        p_k = scheduler.get_pk(n_k, g_norms_tensor, T_upload_all) [cite: 322]
        selected_indices = np.random.choice(K, M, replace=False, p=p_k.numpy()) # [cite: 377]

        # Calculate One-Round Latency (Section III-D)
        # Max computation latency + Selected Upload latency
        T_compute = max([(n_k[k].item() * 1e4) / 1e9 for k in range(K)]) # Proxy for Eq (16) [cite: 264]
        T_upload_selected = T_upload_all[selected_indices].max().item() # Max straggler for selected [cite: 336]
        
        round_latency = T_broadcast + T_compute + T_upload_selected # [cite: 279]
        total_simulated_time += round_latency

        # --- Step 6: Global Update ---
        g_hat = aggregate_unbiased(
            [local_grads[i] for i in selected_indices],
            n_k[selected_indices],
            p_k[selected_indices],
            n_total
        ) [cite: 209, 380]
        
        with torch.no_grad():
            ptr = 0
            for p in global_model.parameters():
                numel = p.numel()
                p.copy_(p - eta * g_hat[ptr:ptr+numel].view_as(p))
                ptr += numel

        # Evaluation
        current_acc = test(global_model, test_loader, device)
        accuracy_history.append(current_acc)
        cumulative_time.append(total_simulated_time)
        
        if t % 10 == 0:
            print(f"Round {t} | Time: {total_simulated_time:.2f}s | Acc: {current_acc:.4f}")

    # Plotting: Accuracy vs. Simulated Time
    history_dict = {f"Proposed (rho={rho})": accuracy_history}
    plot_accuracy(history_dict, title=f"FEEL {args.model_type.upper()} Accuracy", filename=f"{args.model_type}_results.png")
    
    # Save raw data for Figure 3 comparison
    np.savez(f"{args.model_type}_data.npz", time=cumulative_time, acc=accuracy_history)

if __name__ == "__main__":
    run_feel_experiment()