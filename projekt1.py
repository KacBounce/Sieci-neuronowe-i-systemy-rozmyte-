import numpy as np
import matplotlib.pyplot as plt


def dzialaj2(W1, W2, X):
    beta = 5
    X1 = np.vstack((-np.ones((1, X.shape[1])), X))
    U1 = W1.T @ X1
    Y1 = 1 / (1 + np.exp(-beta * U1))
    X2 = np.vstack((-np.ones((1, Y1.shape[1])), Y1))
    U2 = W2.T @ X2
    Y2 = 1 / (1 + np.exp(-beta * U2))
    return Y1, Y2


def ucz2_improved(W1, W2, P, T, n, wspUcz=0.1, beta=5, momentum=0.9, adaptive=True, mini_batch_size=1, early_stop_mse=None):
    liczbaPrzykladow = P.shape[1]
    W1_hist, W2_hist = [W1.copy()], [W2.copy()]
    mse_history = []
    velocity_W1, velocity_W2 = np.zeros_like(W1), np.zeros_like(W2)

    for i in range(n):
        # Mini-batch selection
        indices = np.random.choice(
            liczbaPrzykladow, mini_batch_size, replace=False)
        X_batch = P[:, indices]
        T_batch = T[:, indices]

        dW1_total = np.zeros_like(W1)
        dW2_total = np.zeros_like(W2)

        for j in range(mini_batch_size):
            X = X_batch[:, [j]]
            T_target = T_batch[:, [j]]

            # Forward pass
            X1 = np.vstack((-np.ones((1, 1)), X))
            Y1, Y2 = dzialaj2(W1, W2, X)
            X2 = np.vstack((-np.ones((1, 1)), Y1))

            # Backpropagation
            D2 = T_target - Y2
            E2 = beta * D2 * Y2 * (1 - Y2)
            D1 = W2[1:, :] @ E2
            E1 = beta * D1 * Y1 * (1 - Y1)

            # Gradients
            dW1 = X1 @ E1.T
            dW2 = X2 @ E2.T
            dW1_total += dW1
            dW2_total += dW2

        # Average gradients
        dW1_avg = dW1_total / mini_batch_size
        dW2_avg = dW2_total / mini_batch_size

        # Apply momentum and update weights
        velocity_W1 = momentum * velocity_W1 + wspUcz * dW1_avg
        velocity_W2 = momentum * velocity_W2 + wspUcz * dW2_avg
        W1 += velocity_W1
        W2 += velocity_W2

        # Adaptive learning rate
        if adaptive and i % 100 == 0:
            wspUcz *= 0.99

        # Calculate MSE
        _, Y2_all = dzialaj2(W1, W2, P)
        mse = np.mean((T - Y2_all) ** 2)
        mse_history.append(mse)
        W1_hist.append(W1.copy())
        W2_hist.append(W2.copy())

        # Early stopping
        if early_stop_mse is not None and mse < early_stop_mse:
            print(f"Early stopping at iteration {i+1}, MSE={mse:.6f}")
            break

    return W1, W2, mse_history, W1_hist, W2_hist


def plot_results(mse_history, W1_hist, W2_hist):
    plt.figure(figsize=(15, 5))

    # MSE history
    plt.subplot(1, 3, 1)
    plt.plot(mse_history, label="MSE")
    plt.title("MSE History")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.legend(loc="upper right")

    # W1 weights evolution
    plt.subplot(1, 3, 2)
    W1_all = np.array(W1_hist).reshape(len(W1_hist), -1)
    for idx in range(W1_all.shape[1]):
        plt.plot(W1_all[:, idx], label=f"W1[{idx}]")
    plt.title("W1 Weights Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Weights")
    plt.legend(loc="upper right")

    # W2 weights evolution
    plt.subplot(1, 3, 3)
    W2_all = np.array(W2_hist).reshape(len(W2_hist), -1)
    for idx in range(W2_all.shape[1]):
        plt.plot(W2_all[:, idx], label=f"W2[{idx}]")
    plt.title("W2 Weights Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Weights")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()



# Example usage
P = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
T = np.array([[0, 1, 1, 0]])

# Initialize weights
W1przed = np.random.randn(3, 3)
W2przed = np.random.randn(4, 1)

# Train the network
W1po, W2po, mse_history, W1_hist, W2_hist = ucz2_improved(
    W1przed, W2przed, P, T, n=5000, wspUcz=0.1, beta=5, momentum=0.9, adaptive=True, mini_batch_size=2, early_stop_mse=0.001
)

_, Ypo = dzialaj2(W1po, W2po, P)
np.set_printoptions(precision=6, suppress=True)
# Print the final output
print("Ypo (Network outputs after training):")
print(np.round(Ypo, decimals=5))

# Plot results
plot_results(mse_history, W1_hist, W2_hist)
