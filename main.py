"""
Cognitive Modeling (INFOMCM) - Lab 1, Section 1
Model Human Processor (Card, Moran & Newell, 1983)
"""
import itertools
import matplotlib.pyplot as plt

# Times (ms) per process type and performer type (table in section 1.b)
TIMES = {
    "perceptual": {"fast": 50, "middle": 100, "slow": 200},
    "cognitive":  {"fast": 25, "middle": 70,  "slow": 170},
    "motor":      {"fast": 30, "middle": 70,  "slow": 100},
}
TYPES = ["fast", "middle", "slow"]


# ---------------------------------------------------------------- 1A / 1B
def start():
    return 0                                   # stimulus appears at t = 0

def perceptualstep(type="middle"):
    return TIMES["perceptual"][type]

def cognitivestep(type="middle"):
    return TIMES["cognitive"][type]

def motorstep(type="middle"):
    return TIMES["motor"][type]


def example1():
    """Simple reaction task: see symbol -> decide -> press button."""
    return start() + perceptualstep() + cognitivestep() + motorstep()


def example2(completeness="extremes"):
    if completeness == "extremes":
        # all-fast, all-middle, all-slow
        return [start() + perceptualstep(t) + cognitivestep(t) + motorstep(t)
                for t in TYPES]

    elif completeness == "all":
        results = []
        # 3 x 3 x 3 = 27 combinations of (perceptual, cognitive, motor) type
        for p, c, m in itertools.product(TYPES, repeat=3):
            total = start() + perceptualstep(p) + cognitivestep(c) + motorstep(m)
            print(f"p={p:6s} c={c:6s} m={m:6s} -> {total} ms")
            results.append(total)

        plt.figure(figsize=(4, 5))
        plt.boxplot(results)
        plt.ylim(0, max(results) * 1.1)       # make 0 part of the y-range
        plt.ylabel("time (ms)")
        plt.xticks([1], ["example 2 (all combinations)"])
        plt.title("Simple reaction time")
        plt.tight_layout()
        plt.savefig("example2_boxplot.png", dpi=150)
        plt.close()
        return results


# ---------------------------------------------------------------- 1C1
def example3(p="middle", c="middle", m="middle", completeness=None):
    """
    Two stimuli, press if identical:
      perceive stim 1 (p) -> perceive stim 2 (p) -> compare in WM (c)
      -> prepare motor command (c) -> execute (m)
    Speed type is consistent *within* a process type (both p steps share p),
    but independent *between* process types.
    """
    if completeness == "all":
        out = {}
        for pp, cc, mm in itertools.product(TYPES, repeat=3):
            out[(pp, cc, mm)] = example3(pp, cc, mm)
        return out

    return (start()
            + perceptualstep(p)    # stimulus 1
            + perceptualstep(p)    # stimulus 2
            + cognitivestep(c)     # compare
            + cognitivestep(c)     # prepare motor command
            + motorstep(m))        # press


# ---------------------------------------------------------------- 1C2
def example4(p="middle", c="middle", m="middle"):
    """
    The second stimulus appears at a given time after trial start.
    Its processing can only begin once (a) it has appeared AND
    (b) perception of stimulus 1 is finished  -> start = max(onset, p1_end).
    Returns {onset: total_time}.
    """
    results = {}
    for onset in [40, 80, 110, 150, 210, 240]:
        t = start()
        t += perceptualstep(p)                 # stimulus 1 finished at time t
        t = max(t, onset)                      # wait for stimulus 2 if needed
        t += perceptualstep(p)                 # stimulus 2
        t += cognitivestep(c) + cognitivestep(c) + motorstep(m)
        results[onset] = t
    return results


def example4_all():
    """All 27 speed combinations x 6 onsets."""
    rows = []
    for pp, cc, mm in itertools.product(TYPES, repeat=3):
        for onset, t in example4(pp, cc, mm).items():
            rows.append((pp, cc, mm, onset, t))
    return rows


# ---------------------------------------------------------------- 1C3
ERROR_FACTOR = {"fast": 3, "middle": 2, "slow": 0.5}

def example5(p="middle", c="middle", m="middle"):
    """Returns (time, error probability) for one combination.
    Error = 0.01 multiplied by a factor for EVERY step (interpretation A)."""
    time = example3(p, c, m)
    steps = [p, p, c, c, m]                    # the 5 steps of example 3
    error = 0.01
    for s in steps:
        error *= ERROR_FACTOR[s]
    return time, min(error, 1.0)               # cap at 100 %


def example5_plot():
    times, errors = [], []
    for pp, cc, mm in itertools.product(TYPES, repeat=3):
        t, e = example5(pp, cc, mm)
        times.append(t)
        errors.append(e * 100)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(times, errors)
    plt.xlabel("trial time (ms)")
    plt.ylabel("error probability (%)")
    plt.xlim(0, max(times) * 1.1)
    plt.ylim(0, 105)
    plt.title("Speed-accuracy trade-off (example 5)")
    plt.tight_layout()
    plt.savefig("example5_scatter.png", dpi=150)
    plt.close()
    return list(zip(times, errors))


if __name__ == "__main__":
    print("example1:", example1())                                  # 240
    print("example2 extremes:", example2())                         # 105, 240, 470
    example2("all")
    print("Q1A example3 all fast:", example3("fast", "fast", "fast"))
    print("Q1B example3 fast p / middle c / slow m:", example3("fast", "middle", "slow"))
    print("Q1C example4 slowest:", max(r[-1] for r in example4_all()))
    example5_plot()
