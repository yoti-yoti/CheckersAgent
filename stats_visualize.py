import os
import pickle
import matplotlib.pyplot as plt

STATS_DIR = os.path.join("training", "training_stats")


def load_most_recent_stats(stats_dir=STATS_DIR):
    if not os.path.exists(stats_dir):
        raise FileNotFoundError(f"Stats directory not found: {stats_dir}")

    files = [
        os.path.join(stats_dir, f)
        for f in os.listdir(stats_dir)
        if f.endswith(".pkl")
    ]

    if not files:
        raise FileNotFoundError("No stats pickle files found.")

    latest_file = max(files, key=os.path.getmtime)

    with open(latest_file, "rb") as f:
        stats = pickle.load(f)

    print(f"Loaded stats file: {latest_file}")
    return stats, latest_file


def ensure_output_dir(pickle_path):
    base_name = os.path.splitext(os.path.basename(pickle_path))[0]
    out_dir = os.path.join(STATS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_and_save_win_loss_counts(stats, out_dir):
    win_history = stats.get("win_history", [])

    wins = []
    losses = []
    draws = []

    w = l = d = 0
    for r in win_history:
        if r == 1:
            w += 1
        elif r == -1:
            l += 1
        else:
            d += 1
        wins.append(w)
        losses.append(l)
        draws.append(d)

    plt.figure()
    plt.plot(wins, label="Wins")
    plt.plot(losses, label="Losses")
    plt.plot(draws, label="Draws")
    plt.xlabel("Episodes")
    plt.ylabel("Count")
    plt.title("Win / Loss / Draw Count Over Episodes")
    plt.legend()
    plt.grid(True)

    path = os.path.join(out_dir, "win_loss_draw_counts.png")
    plt.savefig(path)
    plt.close()


def plot_and_save_win_minus_loss(stats, out_dir):
    win_history = stats.get("win_history", [])

    diff = []
    score = 0
    for r in win_history:
        score += r  # win=+1, loss=-1, draw=0
        diff.append(score)

    plt.figure()
    plt.plot(diff)
    plt.xlabel("Episodes")
    plt.ylabel("Wins - Losses")
    plt.title("Wins Minus Losses Over Episodes")
    plt.grid(True)

    path = os.path.join(out_dir, "wins_minus_losses.png")
    plt.savefig(path)
    plt.close()


def plot_and_save_returns_and_lengths(stats, out_dir):
    returns = stats.get("episode_returns", [])
    lengths = stats.get("episode_lengths", [])

    plt.figure()
    plt.plot(returns)
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Episode Returns Over Time")
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "episode_returns.png"))
    plt.close()

    plt.figure()
    plt.plot(lengths)
    plt.xlabel("Episodes")
    plt.ylabel("Episode Length")
    plt.title("Episode Lengths Over Time")
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "episode_lengths.png"))
    plt.close()


def plot_and_save_win_count_per_length(stats, out_dir):
    win_history = stats.get("win_history", [])
    lengths = stats.get("episode_lengths", [])

    win_lengths = [l for r, l in zip(win_history, lengths) if r == 1]

    if not win_lengths:
        print("No wins recorded, skipping win-length plot.")
        return

    plt.figure()
    plt.hist(win_lengths, bins=30)
    plt.xlabel("Episode Length")
    plt.ylabel("Win Count")
    plt.title("Win Count per Episode Length")
    plt.grid(True)

    plt.savefig(os.path.join(out_dir, "win_count_per_length.png"))
    plt.close()


def visualize():
    stats, pickle_path = load_most_recent_stats()
    out_dir = ensure_output_dir(pickle_path)

    plot_and_save_win_loss_counts(stats, out_dir)
    plot_and_save_win_minus_loss(stats, out_dir)
    plot_and_save_returns_and_lengths(stats, out_dir)
    plot_and_save_win_count_per_length(stats, out_dir)

    print(f"Saved plots to: {out_dir}")

def main():
    visualize()

if __name__ == "__main__":
    main()
