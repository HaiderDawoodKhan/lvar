import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ACTION_NAMES = ["THINK", "STOP", "GLOBAL", "REGION", "PATCH"]


def load_jsonl(path: Path):
    rows = []
    bad_rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                bad_rows.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "context": stripped[max(0, exc.pos - 120) : exc.pos + 120],
                    }
                )
    return rows, bad_rows


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def safe_mean(values):
    return mean(values) if values else 0.0


def safe_median(values):
    return median(values) if values else 0.0


def pairwise(items):
    return zip(items, items[1:])


def flatten_traces(rows):
    steps = []
    for row_index, row in enumerate(rows):
        for step in row.get("trace") or []:
            action = str(step.get("action", "UNKNOWN")).upper()
            record = {
                "row_index": row_index,
                "example_id": row.get("example_id"),
                "correct": bool(row.get("correct")),
                "domain": row.get("domain"),
                "topic": row.get("topic"),
                "action": action,
                "step_idx": step.get("step_idx"),
                "patch_index": step.get("patch_index"),
                "region_index": step.get("region_index"),
                "action_probs": step.get("action_probs") or [],
                "sequence_length_before": step.get("sequence_length_before"),
                "sequence_length_after": step.get("sequence_length_after"),
            }
            steps.append(record)
    return steps


def top_counter(counter: Counter, n: int = 10):
    total = sum(counter.values())
    return [
        {"value": key, "count": count, "percent": pct(count, total)}
        for key, count in counter.most_common(n)
    ]


def summarize(rows):
    steps = flatten_traces(rows)
    num_steps = [int(row.get("num_steps") or len(row.get("trace") or [])) for row in rows]
    controller_tokens = [int(row.get("num_controller_tokens") or 0) for row in rows]
    output_tokens = [int(row.get("num_output_tokens") or 0) for row in rows]
    total_tokens = [int(row.get("num_total_tokens") or 0) for row in rows]
    correct_rows = [row for row in rows if bool(row.get("correct"))]
    incorrect_rows = [row for row in rows if not bool(row.get("correct"))]

    action_counter = Counter(step["action"] for step in steps)
    patch_counter = Counter(step["patch_index"] for step in steps if step["patch_index"] is not None)
    region_counter = Counter(step["region_index"] for step in steps if step["region_index"] is not None)

    by_domain = defaultdict(lambda: {"total": 0, "correct": 0, "steps": []})
    by_topic = defaultdict(lambda: {"total": 0, "correct": 0, "steps": []})
    for row in rows:
        row_steps = int(row.get("num_steps") or len(row.get("trace") or []))
        for key, bucket in ((row.get("domain") or "UNKNOWN", by_domain), (row.get("topic") or "UNKNOWN", by_topic)):
            bucket[key]["total"] += 1
            bucket[key]["correct"] += int(bool(row.get("correct")))
            bucket[key]["steps"].append(row_steps)

    repeated_patch_examples = 0
    adjacent_repeat_examples = 0
    repeated_patch_total = 0
    transition_counter = Counter()
    stop_examples = 0
    all_max_action_probs = []
    action_prob_sums = Counter()
    action_prob_counts = Counter()

    for row in rows:
        trace = row.get("trace") or []
        actions = [str(step.get("action", "UNKNOWN")).upper() for step in trace]
        patches = [step.get("patch_index") for step in trace if step.get("patch_index") is not None]
        if len(set(patches)) < len(patches):
            repeated_patch_examples += 1
            repeated_patch_total += len(patches) - len(set(patches))
        if any(left == right for left, right in pairwise(actions)):
            adjacent_repeat_examples += 1
        transition_counter.update(pairwise(actions))
        if "STOP" in actions:
            stop_examples += 1
        for step in trace:
            probs = step.get("action_probs") or []
            if probs:
                all_max_action_probs.append(max(float(value) for value in probs))
                for idx, value in enumerate(probs):
                    label = ACTION_NAMES[idx] if idx < len(ACTION_NAMES) else str(idx)
                    action_prob_sums[label] += float(value)
                    action_prob_counts[label] += 1

    def grouped_summary(group):
        result = {}
        for key, values in group.items():
            result[key] = {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": pct(values["correct"], values["total"]),
                "avg_steps": safe_mean(values["steps"]),
            }
        return result

    return {
        "num_examples": len(rows),
        "num_valid_trace_steps": len(steps),
        "accuracy": pct(len(correct_rows), len(rows)),
        "correct": len(correct_rows),
        "incorrect": len(incorrect_rows),
        "steps": {
            "min": min(num_steps) if num_steps else 0,
            "max": max(num_steps) if num_steps else 0,
            "mean": safe_mean(num_steps),
            "median": safe_median(num_steps),
        },
        "controller_tokens": {
            "mean": safe_mean(controller_tokens),
            "median": safe_median(controller_tokens),
            "max": max(controller_tokens) if controller_tokens else 0,
        },
        "output_tokens": {
            "mean": safe_mean(output_tokens),
            "median": safe_median(output_tokens),
            "max": max(output_tokens) if output_tokens else 0,
        },
        "total_tokens": {
            "mean": safe_mean(total_tokens),
            "median": safe_median(total_tokens),
            "max": max(total_tokens) if total_tokens else 0,
        },
        "actions": {
            "counts": dict(action_counter),
            "top": top_counter(action_counter),
            "most_common": action_counter.most_common(1)[0][0] if action_counter else None,
        },
        "patches": {
            "unique": len(patch_counter),
            "top": top_counter(patch_counter),
            "repeated_patch_examples": repeated_patch_examples,
            "repeated_patch_example_rate": pct(repeated_patch_examples, len(rows)),
            "repeated_patch_extra_count": repeated_patch_total,
        },
        "regions": {
            "unique": len(region_counter),
            "top": top_counter(region_counter),
        },
        "repetition": {
            "adjacent_repeat_examples": adjacent_repeat_examples,
            "adjacent_repeat_example_rate": pct(adjacent_repeat_examples, len(rows)),
            "top_transitions": top_counter(Counter({f"{a}->{b}": c for (a, b), c in transition_counter.items()}), 15),
        },
        "stop": {
            "examples_with_stop": stop_examples,
            "stop_rate": pct(stop_examples, len(rows)),
        },
        "confidence": {
            "mean_max_action_prob": safe_mean(all_max_action_probs),
            "mean_action_probs": {
                key: action_prob_sums[key] / action_prob_counts[key]
                for key in action_prob_sums
            },
        },
        "by_domain": grouped_summary(by_domain),
        "by_topic": grouped_summary(by_topic),
    }


def print_summary(summary):
    print("Inference Log Summary")
    print("=" * 60)
    print(f"Examples: {summary['num_examples']:,}")
    print(f"Trace steps: {summary['num_valid_trace_steps']:,}")
    print(f"Accuracy: {summary['accuracy']:.4f} ({summary['correct']:,}/{summary['num_examples']:,})")
    print()
    print("Steps per example")
    print(f"  min={summary['steps']['min']} max={summary['steps']['max']} mean={summary['steps']['mean']:.3f} median={summary['steps']['median']:.3f}")
    print("Tokens")
    print(f"  controller mean={summary['controller_tokens']['mean']:.3f}")
    print(f"  output mean={summary['output_tokens']['mean']:.3f}")
    print(f"  total mean={summary['total_tokens']['mean']:.3f}")
    print()
    print("Action counts")
    for action, count in Counter(summary["actions"]["counts"]).most_common():
        print(f"  {action}: {count:,}")
    print()
    print("Top patches")
    for item in summary["patches"]["top"]:
        print(f"  patch {item['value']}: {item['count']:,} ({item['percent']:.2%})")
    print()
    print("Top regions")
    for item in summary["regions"]["top"]:
        print(f"  region {item['value']}: {item['count']:,} ({item['percent']:.2%})")
    print()
    print("Repetition")
    print(f"  examples with repeated patch: {summary['patches']['repeated_patch_examples']:,} ({summary['patches']['repeated_patch_example_rate']:.2%})")
    print(f"  examples with adjacent repeated action: {summary['repetition']['adjacent_repeat_examples']:,} ({summary['repetition']['adjacent_repeat_example_rate']:.2%})")
    print()
    print("Top domains")
    for domain, values in sorted(summary["by_domain"].items(), key=lambda item: item[1]["total"], reverse=True)[:10]:
        print(f"  {domain}: acc={values['accuracy']:.4f} total={values['total']:,} avg_steps={values['avg_steps']:.2f}")


def plot_bar(counter, title, xlabel, output_path, top_n=20):
    if not counter:
        return
    items = Counter(counter).most_common(top_n)
    labels = [str(key) for key, _ in items]
    values = [value for _, value in items]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hist(values, title, xlabel, output_path, bins=None):
    if not values:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(values, bins=bins or 20, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Examples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_plots(rows, summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    num_steps = [int(row.get("num_steps") or len(row.get("trace") or [])) for row in rows]
    correct_steps = [int(row.get("num_steps") or len(row.get("trace") or [])) for row in rows if bool(row.get("correct"))]
    incorrect_steps = [int(row.get("num_steps") or len(row.get("trace") or [])) for row in rows if not bool(row.get("correct"))]
    controller_tokens = [int(row.get("num_controller_tokens") or 0) for row in rows]
    output_tokens = [int(row.get("num_output_tokens") or 0) for row in rows]

    max_step = max(num_steps) if num_steps else 0
    plot_hist(num_steps, "Controller steps per example", "Number of steps", output_dir / "steps_hist.png", bins=range(0, max_step + 2))
    plot_hist(controller_tokens, "Controller tokens per example", "Controller tokens", output_dir / "controller_tokens_hist.png")
    plot_hist(output_tokens, "Output tokens per example", "Output tokens", output_dir / "output_tokens_hist.png")
    plot_bar(summary["actions"]["counts"], "Controller action frequency", "Action", output_dir / "action_counts.png")
    plot_bar({item["value"]: item["count"] for item in summary["patches"]["top"]}, "Top patch indices", "Patch index", output_dir / "top_patches.png")
    plot_bar({item["value"]: item["count"] for item in summary["regions"]["top"]}, "Top region indices", "Region index", output_dir / "top_regions.png")

    if correct_steps or incorrect_steps:
        plt.figure(figsize=(9, 5))
        if correct_steps:
            plt.hist(correct_steps, alpha=0.6, label="correct", bins=range(0, max_step + 2), edgecolor="black")
        if incorrect_steps:
            plt.hist(incorrect_steps, alpha=0.6, label="incorrect", bins=range(0, max_step + 2), edgecolor="black")
        plt.title("Controller steps by correctness")
        plt.xlabel("Number of steps")
        plt.ylabel("Examples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "steps_by_correctness.png", dpi=160)
        plt.close()

    domains = summary["by_domain"]
    if domains:
        top_domains = sorted(domains.items(), key=lambda item: item[1]["total"], reverse=True)[:20]
        labels = [str(key) for key, _ in top_domains]
        values = [value["accuracy"] for _, value in top_domains]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, values)
        plt.title("Accuracy by domain")
        plt.xlabel("Domain")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_by_domain.png", dpi=160)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze LVAR inference JSONL logs.")
    parser.add_argument("--input", default="outputs/m3cot_lvar_predictions.jsonl", help="Path to inference JSONL.")
    parser.add_argument("--output-dir", default=None, help="Directory for summary JSON and plots.")
    parser.add_argument("--no-plots", action="store_true", help="Only print/write summary; skip plot generation.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis") / f"{input_path.stem}_analysis"
    rows, bad_rows = load_jsonl(input_path)
    if not rows:
        raise ValueError(f"No valid rows found in {input_path}. Bad rows: {bad_rows[:1]}")

    summary = summarize(rows)
    summary["input_path"] = str(input_path)
    summary["bad_jsonl_rows"] = bad_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_summary(summary)
    print(f"\nWrote summary to {output_dir / 'summary.json'}")
    if not args.no_plots:
        save_plots(rows, summary, output_dir)
        print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
