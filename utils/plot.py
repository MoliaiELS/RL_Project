import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking
import matplotlib.pyplot as plt


def plot_learning_curve(
    rewards,
    window: int = 10,
    title: str = "Learning Curve",
    save_path: str | None = None,
    x_values: list[int] | None = None,
    primary_label: str = "Episode reward",
    secondary_rewards: list[float] | None = None,
    secondary_x: list[int] | None = None,
    secondary_label: str = "Greedy eval",
    ylabel: str = "Reward",
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_axis = x_values if x_values is not None else list(range(1, len(rewards) + 1))
    ax.plot(x_axis, rewards, label=primary_label, alpha=0.4)
    if len(rewards) >= window:
        smoothed = [
            sum(rewards[max(0, i - window + 1) : i + 1]) / min(window, i + 1)
            for i in range(len(rewards))
        ]
        ax.plot(x_axis, smoothed, label=f"Running mean ({window})", color="tab:red")

    if secondary_rewards is not None and secondary_x is not None:
        ax.plot(secondary_x, secondary_rewards, marker="o", linestyle="--", label=secondary_label)

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)  # Always close the figure to free memory


__all__ = ["plot_learning_curve"]
