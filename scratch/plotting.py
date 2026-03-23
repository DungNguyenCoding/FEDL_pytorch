import matplotlib.pyplot as plt

def plot_accuracy(history, title="FEEL Convergence", filename="results.png"):
    plt.figure(figsize=(10, 6))
    for label, accs in history.items():
        plt.plot(accs, label=label)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    
    # Save first, then show
    plt.savefig(filename) 
    plt.show()