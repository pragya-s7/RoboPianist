import json
import matplotlib.pyplot as plt
import argparse

def plot_plan(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    events = data["events"]

    rh_x, rh_y = [], []
    lh_x, lh_y = [], []
    rh_t, lh_t = [], []

    for e in events:
        wt = e.get("wrist_target")
        if not wt:
            continue

        if e["staff"] == 1:
            rh_x.append(wt[0])
            rh_y.append(wt[1])
            rh_t.append(e["onset_sec"])
        else:
            lh_x.append(wt[0])
            lh_y.append(wt[1])
            lh_t.append(e["onset_sec"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot X (Lateral) movement over time
    ax1.plot(rh_t, rh_x, "r.-", label="Right Hand X", alpha=0.7)
    ax1.plot(lh_t, lh_x, "b.-", label="Left Hand X", alpha=0.7)
    ax1.set_ylabel("Lateral Position (m)")
    ax1.set_title("Wrist Lateral Trajectory (Piano Length)")
    ax1.legend()
    ax1.grid(True)

    # Plot Y (Depth) movement over time
    ax2.plot(rh_t, rh_y, "r.-", label="Right Hand Y", alpha=0.7)
    ax2.plot(lh_t, lh_y, "b.-", label="Left Hand Y", alpha=0.7)
    ax2.set_ylabel("Depth Position (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Wrist Depth Trajectory (In/Out)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    plot_plan(args.file)
