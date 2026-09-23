#!/usr/bin/env python
"""Model the other players in the league and predict their picks.

    python profiles.py                 # profiles from all league/week_NN.csv; predict next week
    python profiles.py --predict 3     # predict a specific week (needs sim_data/week_03.json
                                       #   and picks/week_03.json for that week's slate)
    python profiles.py --score 3       # score league/predicted_week_03.csv vs league/week_03.csv

Inputs (all produced by the weekly workflow):
    league/week_NN.csv       every entry's picks + tiebreak (league_log.py)
    sim_data/week_NN.json    per game: vegas favorite, P(fav), CBS % on fav (notebook edge table)
    picks/week_NN.json       away/home/pick per game (emailer snapshot; supplies the dog)

Outputs (league/ is gitignored):
    league/profiles.csv                 one row per player
    league/predicted_week_NN.csv        P(picks favorite) per player x game, plus expected dogs

Model: each player's pick in a game is Bernoulli(P(fav)). Baseline P(fav) is the CBS
public % on the favorite, shifted on the logit scale by a per-player "chalkiness" offset
fitted so their predicted deviation count matches history. Archetypes override:
  chalk   -> 0.97 flat;  hunter -> takes the dog in their usual number of closest games;
  public  -> CBS + offset.  Minnesota lean: an extra logit shift toward MIN/GB when a
player has taken them more often than their model predicted (most players are in MN).
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from league_log import ABBR  # nickname -> CBS abbreviation  # noqa: E402

HOME_TEAMS = {'MIN', 'GB'}          # local-fan teams (league is mostly Minnesotan)
CLOSE, MID = 0.60, 0.70             # P(fav) buckets: close < 0.60 <= mid < 0.70 <= big


def logit(p):
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- data loading

def load_week_games(week):
    """Games for a week: list of dicts {fav, dog, p_fav, cbs_fav} keyed by abbr. None if missing."""
    sd = os.path.join(HERE, 'sim_data', f'week_{week:02d}.json')
    pk = os.path.join(HERE, 'picks', f'week_{week:02d}.json')
    if not (os.path.exists(sd) and os.path.exists(pk)):
        return None
    slate = {ABBR[g['fav']]: g for g in json.load(open(sd))['games']}
    games = []
    for g in json.load(open(pk))['games']:
        fav, dog = ABBR[g['pick']], ABBR[g['away'] if g['pick'] == g['home'] else g['home']]
        s = slate.get(fav)
        if s is None:
            continue
        games.append({'fav': fav, 'dog': dog, 'p_fav': float(s['p_fav']), 'cbs_fav': float(s['cbs_fav'])})
    return games


def load_history():
    """Long table of every (week, player, game) pick with game context."""
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'league', 'week_[0-9][0-9].csv'))):
        week = int(re.search(r'week_(\d+)', f).group(1))
        games = load_week_games(week) or []
        by_team = {}
        for g in games:
            by_team[g['fav']] = g
            by_team[g['dog']] = g
        for r in csv.DictReader(open(f)):
            for pick in r['picks'].split():
                g = by_team.get(pick)
                if g is None:
                    continue                      # game not in slate data (e.g. already-final game)
                rows.append({'week': week, 'name': r['name'], 'fav': g['fav'], 'dog': g['dog'],
                             'p_fav': g['p_fav'], 'cbs_fav': g['cbs_fav'], 'pick': pick,
                             'picked_fav': pick == g['fav'],
                             'tb_vs_vegas': float(r['tb_vs_vegas']) if r['tb_vs_vegas'] not in ('', 'None') else np.nan})
    return rows


# ---------------------------------------------------------------- profiles

def fit_offset(picked_fav, cbs_fav):
    """Logit shift so that sum(expit(logit(cbs)+b)) == observed favorites picked."""
    target = picked_fav.sum()
    z = logit(cbs_fav / 100)
    lo, hi = -6, 6
    for _ in range(50):
        mid = (lo + hi) / 2
        if expit(z + mid).sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build_profiles(hist):
    by = defaultdict(list)
    for r in hist:
        by[r['name']].append(r)
    profiles = {}
    for name, rs in by.items():
        pf = np.array([r['picked_fav'] for r in rs], dtype=float)
        p = np.array([r['p_fav'] for r in rs])
        cbs = np.array([r['cbs_fav'] for r in rs])
        weeks = sorted({r['week'] for r in rs})
        n_weeks = len(weeks)
        devs = (~pf.astype(bool)).sum()
        tb = np.array([r['tb_vs_vegas'] for r in rs if not np.isnan(r['tb_vs_vegas'])])
        tb_by_week = {}
        for r in rs:
            tb_by_week[r['week']] = r['tb_vs_vegas']
        tbs = np.array([v for v in tb_by_week.values() if not np.isnan(v)])
        tb_mad = float(np.mean(np.abs(tbs))) if len(tbs) else np.nan

        def rate(mask):
            return float(pf[mask].mean()) if mask.any() else np.nan
        close, mid, big = p < CLOSE, (p >= CLOSE) & (p < MID), p >= MID

        offset = fit_offset(pf, cbs)
        # local-team lean, per team: how much more often they take MIN (or GB) than the
        # CBS+offset model says. Kept separate because GB@MIN games confound a pooled number.
        lean = {}
        for team in HOME_TEAMS:
            trs = [r for r in rs if team in (r['fav'], r['dog'])]
            if not trs:
                lean[team] = np.nan
                continue
            picked = np.array([r['pick'] == team for r in trs], dtype=float)
            model = np.array([expit(logit(r['cbs_fav'] / 100) + offset) if r['fav'] == team
                              else 1 - expit(logit(r['cbs_fav'] / 100) + offset) for r in trs])
            lean[team] = float((picked - model).mean())            # >0 = takes this team more than modeled

        devs_per_week = devs / n_weeks
        tb_vegas = (not np.isnan(tb_mad)) and tb_mad <= 3.0
        if devs_per_week <= 0.5:
            arch = 'chalk'
        elif devs_per_week >= 2.5 and tb_vegas:
            arch = 'hunter'
        else:
            arch = 'public'
        profiles[name] = {
            'name': name, 'weeks': n_weeks, 'games': len(rs), 'devs_per_week': round(devs_per_week, 2),
            'chalk_rate': round(float(pf.mean()), 3),
            'fav_rate_close': round(rate(close), 2), 'fav_rate_mid': round(rate(mid), 2), 'fav_rate_big': round(rate(big), 2),
            'cbs_offset': round(offset, 2), 'tb_mean_abs_dev': round(tb_mad, 1) if not np.isnan(tb_mad) else '',
            'tb_vegas': tb_vegas,
            'min_lean': round(lean['MIN'], 2) if not np.isnan(lean['MIN']) else '',
            'gb_lean': round(lean['GB'], 2) if not np.isnan(lean['GB']) else '',
            'archetype': arch,
        }
    return profiles


# ---------------------------------------------------------------- prediction

def predict(profiles, games):
    """P(picks favorite) per player per game."""
    out = {}
    for name, pr in profiles.items():
        probs = []
        if pr['archetype'] == 'hunter':
            k = int(round(pr['devs_per_week']))
            closest = sorted(range(len(games)), key=lambda i: games[i]['p_fav'])[:k]
        for i, g in enumerate(games):
            z = logit(g['cbs_fav'] / 100) + pr['cbs_offset']
            if pr['archetype'] == 'chalk':
                q = 0.97
            elif pr['archetype'] == 'hunter':
                q = 0.15 if i in closest else 0.95
            else:
                q = float(expit(z))
            for team, key in (('MIN', 'min_lean'), ('GB', 'gb_lean')):
                lean = pr[key] if pr[key] != '' else 0.0
                if lean and team in (g['fav'], g['dog']):
                    shift = 1.5 * lean                     # logit shift toward the local team
                    q = float(expit(logit(q) + (shift if g['fav'] == team else -shift)))
            probs.append(q)
        out[name] = probs
    return out


def write_prediction(week, games, preds, profiles):
    path = os.path.join(HERE, 'league', f'predicted_week_{week:02d}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'archetype'] + [f"{g['fav']}>{g['dog']}" for g in games])
        for name, probs in preds.items():
            w.writerow([name, profiles[name]['archetype']] + [f"{p:.2f}" for p in probs])
    return path


def score(week):
    """Compare predicted P(fav) with actual picks: log loss, Brier, and dog-count error per game."""
    pred_path = os.path.join(HERE, 'league', f'predicted_week_{week:02d}.csv')
    rows = list(csv.reader(open(pred_path)))
    header, body = rows[0], rows[1:]
    games = [h.split('>') for h in header[2:]]
    actual = {r['name']: r['picks'].split() for r in csv.DictReader(open(os.path.join(HERE, 'league', f'week_{week:02d}.csv')))}
    ll, br, n = 0.0, 0.0, 0
    exp_dogs = np.zeros(len(games)); act_dogs = np.zeros(len(games))
    per_player = []
    for r in body:
        name, probs = r[0], np.array([float(x) for x in r[2:]])
        if name not in actual:
            continue
        picks = set(actual[name])
        y = np.array([g[0] in picks for g in games], dtype=float)
        pcl = np.clip(probs, 0.02, 0.98)
        ll += -(y * np.log(pcl) + (1 - y) * np.log(1 - pcl)).sum(); br += ((probs - y) ** 2).sum(); n += len(y)
        exp_dogs += 1 - probs; act_dogs += 1 - y
        per_player.append((name, r[1], int((1 - y).sum()), round(float((1 - probs).sum()), 1)))
    print(f"week {week}: {len(per_player)} players scored. log loss {ll / n:.3f}/pick, Brier {br / n:.3f}  "
          f"(baseline: always-fav Brier = {np.mean(act_dogs) / len(per_player):.3f})")
    print(f"\n{'game':>10} {'exp dogs':>8} {'actual':>6}")
    for g, e, a in zip(games, exp_dogs, act_dogs):
        print(f"{g[0]+'>'+g[1]:>10} {e:8.1f} {int(a):6d}")
    print(f"\n{'name':24} {'arch':7} {'act devs':>8} {'exp':>5}")
    for name, arch, a, e in sorted(per_player, key=lambda t: -abs(t[2] - t[3])):
        print(f"{name[:24]:24} {arch:7} {a:8d} {e:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predict', type=int, help='week to predict (default: latest with slate data)')
    ap.add_argument('--score', type=int, help='score predicted_week_NN.csv against actual picks')
    args = ap.parse_args()

    if args.score:
        score(args.score)
        return

    hist = load_history()
    profiles = build_profiles(hist)
    weeks_logged = sorted({r['week'] for r in hist})
    print(f"history: weeks {weeks_logged}, {len(profiles)} players, {len(hist)} picks")

    with open(os.path.join(HERE, 'league', 'profiles.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(profiles.values())).keys()))
        w.writeheader(); w.writerows(profiles.values())

    print(f"\n{'name':24} {'arch':7} {'dev/wk':>6} {'close':>5} {'mid':>5} {'big':>5} {'offset':>6} {'tb|dev|':>7} {'MIN':>5} {'GB':>5}")
    for pr in sorted(profiles.values(), key=lambda p: (p['archetype'], -p['devs_per_week'])):
        print(f"{pr['name'][:24]:24} {pr['archetype']:7} {pr['devs_per_week']:6.2f} {pr['fav_rate_close']:5.2f} "
              f"{pr['fav_rate_mid']:5.2f} {pr['fav_rate_big']:5.2f} {pr['cbs_offset']:6.2f} {str(pr['tb_mean_abs_dev']):>7} {str(pr['min_lean']):>5} {str(pr['gb_lean']):>5}")

    for team, key in (('MIN', 'min_lean'), ('GB', 'gb_lean')):
        v = [p[key] for p in profiles.values() if p[key] != '']
        if v:
            print(f"\n{team} lean (picked {team} minus model): mean {np.mean(v):+.2f}; "
                  f"{sum(1 for x in v if x > 0.1)} lean toward, {sum(1 for x in v if x < -0.1)} against, of {len(v)}")

    week = args.predict or (max(weeks_logged) + 1)
    games = load_week_games(week)
    if not games:
        print(f"\nno slate data for week {week} (need sim_data/week_{week:02d}.json and picks/week_{week:02d}.json)")
        return
    preds = predict(profiles, games)
    path = write_prediction(week, games, preds, profiles)
    print(f"\nweek {week} prediction -> {path}")
    print(f"{'game':>10} {'P(fav)':>6} {'CBS%':>5} | {'exp. on dog':>11}  likely dog-takers")
    for i, g in enumerate(games):
        takers = sorted(((1 - preds[n][i], n) for n in preds), reverse=True)
        exp_dog = sum(t[0] for t in takers)
        names = ', '.join(n.split()[0] + (' ' + n.split()[1][0] if len(n.split()) > 1 else '') for q, n in takers if q >= 0.5)
        print(f"{g['fav']+'>'+g['dog']:>10} {g['p_fav']:6.2f} {g['cbs_fav']:5.0f} | {exp_dog:11.1f}  {names}")


if __name__ == '__main__':
    main()
