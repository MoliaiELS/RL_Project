import matplotlib.pyplot as plt


def plot_learning_curve(
    rewards,
    window: int = 10,
    title: str = "Learning Curve",
    save_path: str | None = None,
    secondary_rewards: list[float] | None = None,
    secondary_x: list[int] | None = None,
    secondary_label: str = "Greedy eval",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, label="Episode reward", alpha=0.4)
    if len(rewards) >= window:
        smoothed = [
            sum(rewards[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
            for i in range(len(rewards))
        ]
        ax.plot(smoothed, label=f"Running mean ({window})", color="tab:red")

    if secondary_rewards is not None and secondary_x is not None:
        ax.plot(secondary_x, secondary_rewards, marker="o", linestyle="--", label=secondary_label)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


__all__ = ["plot_learning_curve"]
