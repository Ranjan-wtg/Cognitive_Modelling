"""
Cognitive Modeling (INFOMCM) - Lab 1, Sections 2 & 3
Driver distraction model: e-mailing while driving (after Janssen & Brumby, 2010)

NOTE: the helper functions/parameters at the top mirror the starter file
"drivermodel.py" from Brightspace. If your starter file differs slightly
(names/values), keep the Brightspace versions and plug runTrial /
runSimulations below into it.
"""
import math
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------- parameters
def resetParameters():
    global timeStepPerDriftUpdate, startingPositionInLane
    global gaussDeviateMean, gaussDeviateSD, gaussDriveNoiseMean, gaussDriveNoiseSD
    global maxLateralVelocity, minLateralVelocity
    global wordsPerMinuteMean, wordsPerMinuteSD
    global retrievalTimeWord, retrievalTimeSentence, steeringUpdateTime

    timeStepPerDriftUpdate = 50      # ms: a lane position is stored every 50 ms
    startingPositionInLane = 0.27    # m from lane centre at trial start
    gaussDeviateMean = 0             # m/s drift when NOT steering
    gaussDeviateSD = 0.13
    gaussDriveNoiseMean = 0          # m/s noise while steering (road/car noise)
    gaussDriveNoiseSD = 0.1
    maxLateralVelocity = 1.7         # m/s, steering wheel limits
    minLateralVelocity = -1.7
    wordsPerMinuteMean = 39.33       # two-finger typing (Jiang et al., 2020)
    wordsPerMinuteSD = 10.3
    retrievalTimeWord = 200          # ms "where was I?" after each word switch
    retrievalTimeSentence = 300      # ms to think of the next sentence
    steeringUpdateTime = 250         # ms per steering movement (Salvucci, 2005)


resetParameters()


def velocityCheck(v):
    """Clamp lateral velocity to what the steering wheel allows."""
    return max(minLateralVelocity, min(maxLateralVelocity, v))


def vehicleUpdateActiveSteering(LD):
    """
    Lateral velocity (m/s) produced by active steering, given lane deviation LD.
    The equation assumes LD >= 0 (distance from centre), so we feed it |LD|
    and point the correction back toward the centre (Hint 2B, option A).
    """
    dist = abs(LD)
    latVel = 0.2617 * dist ** 2 + 0.0233 * dist - 0.022
    latVel = velocityCheck(latVel)
    return -latVel if LD > 0 else latVel     # right of centre -> steer left


def vehicleUpdateNotSteering():
    """Random drift velocity (m/s) when nobody is steering."""
    return velocityCheck(np.random.normal(gaussDeviateMean, gaussDeviateSD))


# ------------------------------------------------------------- helpers
def drift(pos, duration_ms, locPos, locColor):
    """No steering for duration_ms: one drift update per 50 ms (floored)."""
    nUpdates = math.floor(duration_ms / timeStepPerDriftUpdate)
    for _ in range(nUpdates):
        vel = vehicleUpdateNotSteering()                         # m/s
        pos += vel * (timeStepPerDriftUpdate / 1000)            # -> metres per 50 ms
        locPos.append(pos)
        locColor.append("red")
    return pos


def steer(pos, nrMovements, locPos, locColor):
    """Active steering: each movement lasts 250 ms = 5 updates of 50 ms.
    Steering velocity is computed ONCE per movement from the current position."""
    updatesPerMovement = steeringUpdateTime // timeStepPerDriftUpdate   # 5
    for _ in range(nrMovements):
        steerVel = vehicleUpdateActiveSteering(pos)
        for _ in range(updatesPerMovement):
            noise = np.random.normal(gaussDriveNoiseMean, gaussDriveNoiseSD)
            pos += (steerVel + noise) * (timeStepPerDriftUpdate / 1000)
            locPos.append(pos)
            locColor.append("blue")
    return pos


# ------------------------------------------------------------- runTrial
def runTrial(nrWordsPerSentence=17, nrSentences=10,
             nrSteeringMovementsWhenSteering=4, interleaving="word",
             plot=True):
    resetParameters()
    locPos, locColor = [], []
    trialTime = 0
    pos = startingPositionInLane

    # typing speed for THIS trial (sampled once): WPM -> ms per word
    wpm = max(np.random.normal(wordsPerMinuteMean, wordsPerMinuteSD), 1)
    timePerWord = 60000 / wpm

    if interleaving == "word":
        for s in range(nrSentences):
            for w in range(nrWordsPerSentence):
                # 1. type one word (eyes off road -> car drifts)
                typeTime = timePerWord + retrievalTimeWord
                if w == 0:
                    typeTime += retrievalTimeSentence
                trialTime += typeTime
                pos = drift(pos, typeTime, locPos, locColor)

                # 2. steer, unless this was the very last word of the e-mail
                lastWord = (s == nrSentences - 1) and (w == nrWordsPerSentence - 1)
                if not lastWord:
                    pos = steer(pos, nrSteeringMovementsWhenSteering, locPos, locColor)
                    trialTime += nrSteeringMovementsWhenSteering * steeringUpdateTime

    elif interleaving == "sentence":
        for s in range(nrSentences):
            # whole sentence in one go: no per-word retrieval
            typeTime = retrievalTimeSentence + nrWordsPerSentence * timePerWord
            trialTime += typeTime
            pos = drift(pos, typeTime, locPos, locColor)
            if s < nrSentences - 1:
                pos = steer(pos, nrSteeringMovementsWhenSteering, locPos, locColor)
                trialTime += nrSteeringMovementsWhenSteering * steeringUpdateTime

    elif interleaving in ("none", "drivingOnly"):
        # how long typing the e-mail WOULD take (sentence retrieval, no word retrieval)
        typeTime = nrSentences * (retrievalTimeSentence + nrWordsPerSentence * timePerWord)
        trialTime = typeTime
        if interleaving == "none":           # only type, nobody steers
            pos = drift(pos, typeTime, locPos, locColor)
        else:                                # only drive, the whole time
            nrMoves = math.floor(typeTime / steeringUpdateTime)
            pos = steer(pos, nrMoves, locPos, locColor)

    else:
        raise ValueError(f"unknown interleaving strategy: {interleaving}")

    absPos = np.abs(locPos)
    meanDev, maxDev = float(np.mean(absPos)), float(np.max(absPos))

    if plot:
        times = np.arange(1, len(locPos) + 1) * timeStepPerDriftUpdate
        plt.figure(figsize=(9, 4.5))
        plt.scatter(times, locPos, c=locColor, s=6)
        plt.axhline(0, color="grey", lw=0.8, ls="--")
        plt.xlabel("time (ms)")
        plt.ylabel("lane position (m)")
        plt.title(f"Interleaving: {interleaving}")
        plt.text(0.01, 0.97,
                 f"Total trial time = {trialTime:.0f} ms; "
                 f"Mean position on the road = {meanDev:.3f} m; "
                 f"Max position on the road (absolute) = {maxDev:.3f} m",
                 transform=plt.gca().transAxes, va="top", fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.8))
        plt.tight_layout()

    return trialTime, meanDev, maxDev


# ------------------------------------------------------------- runSimulations
def runSimulations(nrSims=100):
    totalTime, meanDeviation, maxDeviation, Condition = [], [], [], []
    conditions = ["word", "sentence", "drivingOnly", "none"]

    for cond in conditions:
        for _ in range(nrSims):
            nrWords = np.random.randint(15, 21)          # uniform 15..20
            t, mean_d, max_d = runTrial(nrWordsPerSentence=nrWords, nrSentences=10,
                                        nrSteeringMovementsWhenSteering=4,
                                        interleaving=cond, plot=False)
            totalTime.append(t); meanDeviation.append(mean_d)
            maxDeviation.append(max_d); Condition.append(cond)

    totalTime, maxDeviation, Condition = map(np.array, (totalTime, maxDeviation, Condition))
    markers = {"word": "o", "sentence": "^", "drivingOnly": "s", "none": "X"}
    colors = {"word": "tab:blue", "sentence": "tab:red",
              "drivingOnly": "tab:green", "none": "black"}

    plt.figure(figsize=(9, 6))
    for cond in conditions:                               # raw trials, grey
        sel = Condition == cond
        plt.scatter(totalTime[sel], maxDeviation[sel], marker=markers[cond],
                    color="grey", alpha=0.4, s=20)
    for cond in conditions:                               # means + SD error bars
        sel = Condition == cond
        mx, my = totalTime[sel].mean(), maxDeviation[sel].mean()
        sx, sy = totalTime[sel].std(), maxDeviation[sel].std()
        plt.errorbar(mx, my, xerr=sx, yerr=sy, fmt=markers[cond], color=colors[cond],
                     markersize=11, capsize=5, lw=1.5, label=cond)
    plt.xlabel("total trial time (ms)")
    plt.ylabel("max lateral deviation (m)")
    plt.title(f"Interleaving strategies ({nrSims} simulations per condition)")
    plt.legend(title="Condition")
    plt.tight_layout()
    return totalTime, meanDeviation, maxDeviation, Condition


if __name__ == "__main__":
    np.random.seed(1)
    for strat in ["word", "sentence", "drivingOnly", "none"]:
        print(strat, runTrial(interleaving=strat))
        plt.savefig(f"trial_{strat}.png", dpi=130); plt.close()
    tt, md, mx, cond = runSimulations(100)
    plt.savefig("simulations.png", dpi=130); plt.close()
    for c in ["word", "sentence", "drivingOnly", "none"]:
        s = cond == c
        print(f"{c:12s} time={tt[s].mean():8.0f} ms  maxDev={mx[s].mean():.3f} m")