#!/usr/bin/env python
"""Rest-of-season Monte Carlo with the REAL league: every rival picks per their fitted
profile (profiles.py), starting from the actual standings.

    python season_sim.py                       # sweep B's hunter flips 0..3
    python season_sim.py --seasons 20000 --flips 0 1 2 3

Model
  * Standings: ytd points from the latest league/week_NN.csv; remaining weeks = 18 - N.
  * Slates: each remaining week draws 15 games from the real (p_fav, cbs_fav) pairs in
    sim_data/ (with jitter). cbs_fav < 50 = the crowd sits on the vegas dog (split game).
  * Rivals: P(picks favorite) from profiles.predict() on the drawn slate: chalk 0.97,
    hunters take the dog in their usual number of closest games, public = CBS% shifted by
    their offset and the close-game boost. Tiebreak guess ~ N(0, their mean |tb - vegas|).
  * Entry A (Hannah): chalk. Entry B (Sean): chalk + toss-up fade (p_fav <= 0.53 and
    cbs_fav > 50) + `flips` highest-edge dogs (edge = (1-p_fav) * cbs_fav).
    Tiebreaks: bracket the total when picks differ, straddle +-3 when identical.
  * Vegas probabilities are taken as true (the strategy's premise).
  * Weekly $40 to the top score (closest tiebreak; exact ties split). Season
    $140/$120/$100 to the top 3 cumulative (ties broken at random).
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import profiles as P  # noqa: E402

WEEKS_TOTAL = 18
GAMES = 15
TOTAL_SD = 13.5
TOSSUP_P = 0.53
PAYOUT_WEEK = 40
PAYOUT_SEASON = np.array([140, 120, 100])
A_NAME, B_NAME = 'Hannah Brady', 'Sean Bruton'


def load_standings():
    files = sorted(glob.glob(os.path.join(HERE, 'league', 'week_[0-9][0-9].csv')))
    week = int(re.search(r'week_(\d+)', files[-1]).group(1))
    ytd = {r['name']: int(r['ytd_pts']) for r in csv.DictReader(open(files[-1]))}
    return week, ytd


def load_slates():
    pool = []
    for f in sorted(glob.glob(os.path.join(HERE, 'sim_data', 'week_*.json'))):
        pool += [(g['p_fav'], g['cbs_fav']) for g in json.load(open(f))['games']]
    return np.array(pool, dtype=float)


def rival_fav_probs(profs, names, p_fav, cbs):
    """(n_rivals, G) P(picks favorite) for one drawn slate, via profiles.predict."""
    games = [{'fav': f'F{i}', 'dog': f'D{i}', 'p_fav': float(p), 'cbs_fav': float(c)}
             for i, (p, c) in enumerate(zip(p_fav, cbs))]
    pred = P.predict({n: profs[n] for n in names}, games)
    return np.array([pred[n] for n in names])


def simulate(seasons, flips, profs, names, ytd, week_now, pool, seed=0):
    rng = np.random.default_rng(seed)
    S, R = seasons, len(names)
    weeks_left = WEEKS_TOTAL - week_now
    cum = np.zeros((S, R + 2), dtype=np.int16)
    cum[:, :R] = np.array([ytd[n] for n in names], dtype=np.int16)
    iA, iB = R, R + 1
    cum[:, iA], cum[:, iB] = ytd[A_NAME], ytd[B_NAME]
    tb_sd = np.array([max(1.0, float(profs[n]['tb_mean_abs_dev'] or 6.0)) * 1.25 for n in names])
    week_dollars = np.zeros((S, R + 2))
    b_week_wins = np.zeros(S)

    for _ in range(weeks_left):
        # one drawn slate per simulated season is expensive through profiles.predict, so
        # draw a modest number of distinct slates per week and assign seasons to them
        n_slates = 40
        idx = rng.integers(0, len(pool), size=(n_slates, GAMES))
        p_fav = np.clip(pool[idx, 0] + rng.normal(0, 0.02, (n_slates, GAMES)), 0.5, 0.97)
        cbs = np.clip(pool[idx, 1] + rng.normal(0, 3, (n_slates, GAMES)), 2, 99)
        q_riv = np.stack([rival_fav_probs(profs, names, p_fav[k], cbs[k]) for k in range(n_slates)])  # (n_slates, R, G)
        which = rng.integers(0, n_slates, size=S)
        pf, cb, qr = p_fav[which], cbs[which], q_riv[which]            # (S,G), (S,G), (S,R,G)

        fav_wins = rng.random((S, GAMES)) < pf
        riv_pick = rng.random((S, R, GAMES)) < qr
        riv_correct = (riv_pick == fav_wins[:, None, :]).sum(axis=2)
        a_correct = fav_wins.sum(axis=1)

        b_pick = np.ones((S, GAMES), dtype=bool)
        fade = (pf <= TOSSUP_P) & (cb > 50)
        b_pick[fade] = False
        if flips > 0:
            edge = (1 - pf) * cb / 100
            edge[fade] = -1
            top = np.argsort(-edge, axis=1)[:, :flips]
            b_pick[np.arange(S)[:, None], top] = False
        b_correct = (b_pick == fav_wins).sum(axis=1)
        same = b_pick.all(axis=1)

        correct = np.concatenate([riv_correct, a_correct[:, None], b_correct[:, None]], axis=1)
        actual = rng.normal(0, TOTAL_SD, S)
        dist = np.empty((S, R + 2))
        dist[:, :R] = np.abs(rng.normal(0, 1, (S, R)) * tb_sd - actual[:, None])
        dist[:, iA] = np.abs(np.where(same, -3.0, -0.1) - actual)
        dist[:, iB] = np.abs(np.where(same, 3.0, 0.9) - actual)
        best = correct.max(axis=1, keepdims=True)
        tied = correct == best
        dm = np.where(tied, dist, np.inf)
        winners = tied & (np.abs(dm - dm.min(axis=1, keepdims=True)) < 1e-9)
        week_dollars += winners * (PAYOUT_WEEK / winners.sum(axis=1, keepdims=True))
        b_week_wins += winners[:, iB]
        cum += correct.astype(np.int16)

    jitter = rng.random((S, R + 2)) * 0.5
    order = np.argsort(-(cum + jitter), axis=1)
    season_dollars = np.zeros((S, R + 2))
    season_dollars[np.arange(S)[:, None], order[:, :3]] = PAYOUT_SEASON
    return {
        'A_week': week_dollars[:, iA], 'B_week': week_dollars[:, iB],
        'A_season': season_dollars[:, iA], 'B_season': season_dollars[:, iB],
        'A_podium': season_dollars[:, iA] > 0, 'B_podium': season_dollars[:, iB] > 0,
        'B_wins': b_week_wins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seasons', type=int, default=10000)
    ap.add_argument('--flips', type=int, nargs='+', default=[0, 1, 2, 3])
    args = ap.parse_args()

    hist = P.load_history()
    profs = P.build_profiles(hist)
    P.CLOSE_BOOST = P.fit_close_boost(hist, profs)
    week_now, ytd = load_standings()
    names = [n for n in profs if n not in (A_NAME, B_NAME) and n in ytd]
    pool = load_slates()
    third = sorted(ytd.values(), reverse=True)[2]
    print(f"after week {week_now}: {len(names)} rivals, 3rd place {third}, "
          f"A {ytd[A_NAME]} ({third - ytd[A_NAME]} back), B {ytd[B_NAME]} ({third - ytd[B_NAME]} back); "
          f"{WEEKS_TOTAL - week_now} weeks left, {args.seasons} seasons, close boost {P.CLOSE_BOOST}")
    print(f"\n{'flips':>5} | {'A $':>6} {'B $':>6} {'both $':>7} | {'A pod':>6} {'A wk$':>6} | "
          f"{'B wk$':>6} {'B seas$':>7} {'B pod':>6} {'P(B wins >=1 wk)':>16} {'E[B wins]':>9}")
    for flips in args.flips:
        r = simulate(args.seasons, flips, profs, names, ytd, week_now, pool)
        A = r['A_week'].mean() + r['A_season'].mean(); B = r['B_week'].mean() + r['B_season'].mean()
        print(f"{flips:>5} | {A:6.1f} {B:6.1f} {A + B:7.1f} | {r['A_podium'].mean():6.1%} {r['A_week'].mean():6.1f} | "
              f"{r['B_week'].mean():6.1f} {r['B_season'].mean():7.1f} {r['B_podium'].mean():6.1%} "
              f"{(r['B_wins'] > 0).mean():16.1%} {r['B_wins'].mean():9.2f}")


if __name__ == '__main__':
    main()
