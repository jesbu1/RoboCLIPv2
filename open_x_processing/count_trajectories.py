import h5py
import argparse
import json
from collections import defaultdict
from tqdm import tqdm

# Constants
LANG_EMBEDDING_KEYS = [
    "liv_lang_embedding",
    "minilm_lang_embedding",
    "liv_lang_embedding_individual",
    "minilm_lang_embedding_individual",
]


def count_trajectories(h5_path):
    """
    Count trajectories in an H5 file and provide detailed statistics.

    Args:
        h5_path (str): Path to the H5 file

    Returns:
        dict: Statistics about the trajectories
    """
    stats = {
        "total_trajectories": 0,
        "total_tasks": 0,
        "trajectories_per_task": {},
        "top_tasks": [],
    }

    with h5py.File(h5_path, "r") as f:
        # Iterate through all tasks
        for task_name in tqdm(f.keys(), desc="Analyzing tasks"):
            task_group = f[task_name]

            # Count trajectories for this task (excluding language embeddings)
            trajectory_count = len(
                [
                    k
                    for k in task_group.keys()
                    if not any(x in k for x in LANG_EMBEDDING_KEYS)
                ]
            )

            if trajectory_count > 0:
                stats["trajectories_per_task"][task_name] = trajectory_count
                stats["total_trajectories"] += trajectory_count

        stats["total_tasks"] = len(stats["trajectories_per_task"])

        # Get top 10 tasks by number of trajectories
        top_tasks = sorted(
            stats["trajectories_per_task"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        stats["top_tasks"] = [
            {"task": task, "count": count} for task, count in top_tasks
        ]

        # Calculate average trajectories per task
        stats["avg_trajectories_per_task"] = (
            stats["total_trajectories"] / stats["total_tasks"]
            if stats["total_tasks"] > 0
            else 0
        )

    return stats


def print_statistics(stats):
    """Print statistics in a readable format."""
    print("\n=== Trajectory Statistics ===")
    print(f"Total Trajectories: {stats['total_trajectories']:,}")
    print(f"Total Tasks: {stats['total_tasks']:,}")
    print(f"Average Trajectories per Task: {stats['avg_trajectories_per_task']:.2f}")

    print("\nTop 10 Tasks by Number of Trajectories:")
    for i, task_info in enumerate(stats["top_tasks"], 1):
        print(f"\n{i}. Task: {task_info['task']}")
        print(f"   Count: {task_info['count']:,}")


def main():
    parser = argparse.ArgumentParser(description="Count trajectories in H5 file")
    parser.add_argument("input_path", type=str, help="Path to H5 file")
    parser.add_argument(
        "--save_stats", type=str, help="Path to save statistics as JSON"
    )

    args = parser.parse_args()

    print(f"Analyzing {args.input_path}...")
    stats = count_trajectories(args.input_path)

    print_statistics(stats)

    if args.save_stats:
        with open(args.save_stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to {args.save_stats}")


if __name__ == "__main__":
    main()
