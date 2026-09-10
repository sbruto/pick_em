#!/usr/bin/env python
"""Monte Carlo of the pick'em league to tune the two-entry strategy.

    python league_sim.py                 # default sweep, prints tables
    python league_sim.py --seasons 20000 --sharps 4 --amp 1.0

Model (see CLAUDE.md "League rules" / "Strategy conclusions"):
  * 18 weeks, ~16 games/week. Each game has a vegas favorite win prob p_fav and a CBS
    public pick % on the favorite. Slates are resampled (with jitter) from real weekly
    data: every sim_data/week_NN.json if present, else the built-in Week 1 numbers.
  * Field: N_ENTRIES total. "Public" players pick the favorite with prob = CBS % (logit
    amplified by --amp); tiebreak guess is loose. "Sharps" play vegas chalk with a small
    deviation rate; tiebreak near the vegas total.
  * Entry A: pure chalk, tiebreak floor(total). Entry B: chalk, but in toss-ups
    (p_fav < TOSSUP_P and CBS > 50% on fav) take the dog; once B is >= fuse picks behind
    3rd place cumulatively, it switches to hunter mode for good: also flip the top
    `flips` games by edge = (1 - p_fav) * cbs_fav. Tiebreak: bracket the total when A/B
    picks differ, straddle +-3 when identical.
  * Weekly $40 to most correct (tiebreak: closest total; exact ties split). Season
    $160/$140/$120 to top 3 cumulative (ties broken at random).
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Week 1 2026, from the notebook's edge table (vegas vig-free P(fav), CBS % on fav)
WEEK1 = [(0.630, 83), (0.639, 86), (0.767, 92), (0.641, 76), (0.619, 83), (0.612, 86),
         (0.510, 81), (0.580, 87), (0.551, 80), (0.733, 95), (0.814, 95), (0.628, 81),
         (0.665, 93), (0.515, 37), (0.580, 77), (0.572, 55)]

N_ENTRIES = 36
WEEKS = 18
GAMES = 16
TOSSUP_P = 0.53          # p_fav below this = toss-up (|spread| < ~1.5)
TOTAL_SD = 13.5          # actual total vs vegas total
PUBLIC_TB_SD = 7.0       # public tiebreak guess vs vegas total
SHARP_TB_SD = 1.5
PAYOUT_WEEK = 40
PAYOUT_SEASON = np.array([160, 140, 120])


def load_pool(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'week_*.json')))
    pool = []
    for f in files:
        d = json.load(open(f))
        pool += [(g['p_fav'], g['cbs_fav']) for g in d['games'] if g.get('cbs_fav') is not None]
    return (np.array(pool), len(files)) if pool else (np.array(WEEK1), 0)


def amplify(pct, amp):
    """Logit-scale amplification of CBS % (amp>1 = crowd is even more lopsided)."""
    p = np.clip(pct / 100, 0.02, 0.98)
    z = np.log(p / (1 - p)) * amp
    return 1 / (1 + np.exp(-z))


def simulate(pool, seasons, sharps, amp, fuse, flips, sharp_dev=0.03, seed=0):
    """Returns dict of per-season arrays: dollars for A, B, weekly wins, podium flags."""
    rng = np.random.default_rng(seed)
    n_pub = N_ENTRIES - sharps - 2
    S = seasons

    cum = np.zeros((S, N_ENTRIES), dtype=np.int16)   # 0..n_pub-1 public, then sharps, A, B
    iA, iB = N_ENTRIES - 2, N_ENTRIES - 1
    week_dollars = np.zeros((S, N_ENTRIES))
    hunter = np.zeros(S, dtype=bool)
    hunter_start = np.full(S, WEEKS + 1)

    for w in range(WEEKS):
        idx = rng.integers(0, len(pool), size=(S, GAMES))
        p_fav = np.clip(pool[idx, 0] + rng.normal(0, 0.03, (S, GAMES)), 0.5, 0.97)
        cbs = np.clip(pool[idx, 1] + rng.normal(0, 4, (S, GAMES)), 5, 99)
        q_pub = amplify(cbs, amp)                                        # P(public picks fav)

        fav_wins = rng.random((S, GAMES)) < p_fav                        # (S, G)

        # public: 1 = picked favorite
        pub_pick = rng.random((S, n_pub, GAMES)) < q_pub[:, None, :]
        pub_correct = (pub_pick == fav_wins[:, None, :]).sum(axis=2)
        # sharps
        sh_pick = rng.random((S, sharps, GAMES)) >= sharp_dev
        sh_correct = (sh_pick == fav_wins[:, None, :]).sum(axis=2)
        # A: chalk
        a_correct = fav_wins.sum(axis=1)
        # B: chalk + toss-up fade (+ hunter flips)
        b_pick = np.ones((S, GAMES), dtype=bool)
        fade = (p_fav < TOSSUP_P) & (cbs > 50)
        b_pick[fade] = False
        if flips > 0:
            edge = (1 - p_fav) * cbs / 100
            edge[fade] = -1
            top = np.argsort(-edge, axis=1)[:, :flips]
            rows = np.arange(S)[:, None]
            flip_mask = np.zeros((S, GAMES), dtype=bool)
            flip_mask[rows, top] = True
            flip_mask &= hunter[:, None]
            b_pick[flip_mask] = False
        b_correct = (b_pick == fav_wins).sum(axis=1)
        same_picks = ~(fade.any(axis=1) | (hunter & (flips > 0)))

        correct = np.concatenate([pub_correct, sh_correct, a_correct[:, None], b_correct[:, None]], axis=1)

        # tiebreak distances
        actual = rng.normal(0, TOTAL_SD, S)                              # relative to vegas total
        dist = np.empty((S, N_ENTRIES))
        dist[:, :n_pub] = np.abs(rng.normal(0, PUBLIC_TB_SD, (S, n_pub)) - actual[:, None])
        dist[:, n_pub:n_pub + sharps] = np.abs(rng.normal(0, SHARP_TB_SD, (S, sharps)) - actual[:, None])
        tbA = np.where(same_picks, -3.0, -0.1)
        tbB = np.where(same_picks, 3.0, 0.9)
        dist[:, iA] = np.abs(tbA - actual)
        dist[:, iB] = np.abs(tbB - actual)

        best = correct.max(axis=1, keepdims=True)
        tied = correct == best
        d_masked = np.where(tied, dist, np.inf)
        dmin = d_masked.min(axis=1, keepdims=True)
        winners = tied & (np.abs(d_masked - dmin) < 1e-9)
        week_dollars += winners * (PAYOUT_WEEK / winners.sum(axis=1, keepdims=True))

        cum += correct.astype(np.int16)

        # fuse check: B's deficit to 3rd place
        third = np.sort(cum, axis=1)[:, -3]
        newly = (~hunter) & ((third - cum[:, iB]) >= fuse)
        hunter_start[newly] = w + 2
        hunter |= newly

    # season payouts: rank by cum, random tiebreak
    jitter = rng.random((S, N_ENTRIES)) * 0.5
    order = np.argsort(-(cum + jitter), axis=1)
    season_dollars = np.zeros((S, N_ENTRIES))
    rows = np.arange(S)[:, None]
    season_dollars[rows, order[:, :3]] = PAYOUT_SEASON

    return {
        'A_week': week_dollars[:, iA], 'B_week': week_dollars[:, iB],
        'A_season': season_dollars[:, iA], 'B_season': season_dollars[:, iB],
        'A_podium': season_dollars[:, iA] > 0, 'B_podium': season_dollars[:, iB] > 0,
        'A_anywin': week_dollars[:, iA] > 0, 'B_anywin': week_dollars[:, iB] > 0,
        'hunter_start': hunter_start,
    }


def summarize(r):
    A = r['A_week'].mean() + r['A_season'].mean()
    B = r['B_week'].mean() + r['B_season'].mean()
    hs = r['hunter_start']
    return {
        'A_$': A, 'B_$': B, 'total_$': A + B,
        'A_week_$': r['A_week'].mean(), 'B_week_$': r['B_week'].mean(),
        'A_season_$': r['A_season'].mean(), 'B_season_$': r['B_season'].mean(),
        'A_podium': r['A_podium'].mean(), 'B_podium': r['B_podium'].mean(),
        'P(B wins a week)': r['B_anywin'].mean(),
        'hunter_by': f"{(hs <= WEEKS).mean():.0%} (median wk {np.median(hs[hs <= WEEKS]) if (hs <= WEEKS).any() else '-'})",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, default=20000)
    ap.add_argument('--sharps', type=int, nargs='+', default=[2, 4, 6])
    ap.add_argument('--amp', type=float, default=1.0)
    ap.add_argument('--fuse', type=int, nargs='+', default=[0, 2, 3, 4, 5, 99])
    ap.add_argument('--flips', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--data', default=os.path.join(HERE, 'sim_data'))
    args = ap.parse_args()

    pool, n_weeks = load_pool(args.data)
    print(f"slate pool: {len(pool)} games from {n_weeks or 'built-in Week 1'} week(s); "
          f"mean P(fav)={pool[:, 0].mean():.3f}, mean CBS% on fav={pool[:, 1].mean():.0f}, "
          f"{N_ENTRIES} entries, {args.seasons} seasons, amp={args.amp}")
    print(f"buy-in for two entries: $60. fuse=0 means hunter from week 1, fuse=99 never.\n")

    for sharps in args.sharps:
        print(f"=== {sharps} sharps, {N_ENTRIES - sharps - 2} public ===")
        print(f"{'fuse':>4} {'flips':>5} | {'A $':>6} {'B $':>6} {'both $':>7} | {'A wk$':>6} {'A pod':>6} | "
              f"{'B wk$':>6} {'B seas$':>7} {'B pod':>6} {'P(B wk win)':>11} | hunter by")
        base = None
        for fuse in args.fuse:
            for flips in (args.flips if fuse != 99 else [0]):
                s = summarize(simulate(pool, args.seasons, sharps, args.amp, fuse, flips))
                if base is None:
                    base = s['total_$']
                print(f"{fuse:>4} {flips:>5} | {s['A_$']:6.1f} {s['B_$']:6.1f} {s['total_$']:7.1f} | "
                      f"{s['A_week_$']:6.1f} {s['A_podium']:6.1%} | {s['B_week_$']:6.1f} {s['B_season_$']:7.1f} "
                      f"{s['B_podium']:6.1%} {s['P(B wins a week)']:11.1%} | {s['hunter_by']}")
        print()


if __name__ == '__main__':
    main()
