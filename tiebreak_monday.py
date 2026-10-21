#!/usr/bin/env python
"""Sunday-night / Monday-morning tiebreak optimizer.

By Monday the week's picks are locked except the Monday game, so we know exactly who is
tied with (or within reach of) Hannah. Pick her tiebreak number against that group.

    python tiebreak_monday.py --week 7 --total 44.3 --fav DAL --dog PHI --p-fav 0.66 \\
        --tied "Justin Broederdorf" "Ralph Llewellyn pSXgla" "Brade Schweitzer" \\
        [--hannah-pick DAL] [--guesses "Brade Schweitzer:44" "Justin Broederdorf:48"] \\
        [--entry Hannah|Sean]

  --tied      rivals currently tied with Hannah at the top (ignore anyone 1+ behind)
  --behind    rivals ONE point behind Hannah (they join the tie only if their Monday pick
              wins and Hannah's loses). Optional.
  --guesses   rivals' tiebreak numbers if visible on the pool page; otherwise each is
              sampled from that rival's own history (league/week_NN.csv tb_vs_vegas)
  --total     this week's vegas total for the Monday game (notebook / email prints it)
  --fav/--dog/--p-fav   Monday game vegas favorite, dog, P(fav) (notebook edge table)

Model: the Monday pick of each rival is predicted from profiles.py (fan teams included).
Actual total = vegas total + a draw from this season's (actual - closing line)
distribution (ESPN closing O/U; right-skewed, mean ~+1). For each candidate number N,
P(Hannah wins the tiebreak) = sum over Monday outcomes of P(outcome) x P(N closest
among the rivals still tied after that outcome). Exact ties on distance split the prize.
"""
import argparse
import csv
import glob
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import profiles as P  # noqa: E402

A_NAME, B_NAME = 'Hannah Brady', 'Sean Bruton'
N_SIM = 40000


def actual_minus_line(season=2026, weeks=None, cache=os.path.join(HERE, 'league', 'closing_totals.json')):
    """(actual total - closing O/U) for every completed game this season, cached."""
    rows = json.load(open(cache)) if os.path.exists(cache) else []
    have = {(r[0], r[1]) for r in rows}
    H = {'User-Agent': 'Mozilla/5.0'}
    get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=30))
    for w in (weeks or range(1, 19)):
        try:
            d = get(f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={w}&dates={season}")
        except Exception:
            break
        if not d['events']:
            break
        new = 0
        for e in d['events']:
            c = e['competitions'][0]
            if not c['status']['type']['completed'] or (w, e['shortName']) in have:
                continue
            tot = sum(int(x['score']) for x in c['competitors'])
            try:
                s = get(f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={e['id']}")
                ou = [o.get('overUnder') for o in s.get('pickcenter', []) if o.get('overUnder')]
                if ou:
                    rows.append([w, e['shortName'], tot, float(ou[0])]); new += 1
            except Exception:
                pass
        if new == 0 and any(not e['competitions'][0]['status']['type']['completed'] for e in d['events']):
            break
    json.dump(rows, open(cache, 'w'))
    return np.array([r[2] - r[3] for r in rows])


def rival_guess_history(name):
    """That rival's past (tiebreak - vegas total) offsets."""
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, 'league', 'week_[0-9][0-9].csv'))):
        for r in csv.DictReader(open(f)):
            if r['name'] == name and r['tb_vs_vegas'] not in ('', 'None'):
                out.append(float(r['tb_vs_vegas']))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--week', type=int, required=True)
    ap.add_argument('--total', type=float, required=True, help='vegas total for the Monday game')
    ap.add_argument('--fav', required=True); ap.add_argument('--dog', required=True)
    ap.add_argument('--p-fav', type=float, required=True)
    ap.add_argument('--tied', nargs='*', default=[])
    ap.add_argument('--behind', nargs='*', default=[], help='rivals one point behind Hannah')
    ap.add_argument('--guesses', nargs='*', default=[], help='"Name:number" if visible')
    ap.add_argument('--entry', default=A_NAME.split()[0], help='Hannah or Sean')
    ap.add_argument('--our-pick', default=None, help="our Monday pick (default: favorite)")
    args = ap.parse_args()

    us = A_NAME if args.entry.lower().startswith('h') else B_NAME
    our_pick = args.our_pick or args.fav
    rng = np.random.default_rng(0)

    hist = P.load_history()
    profs = P.build_profiles(hist)
    P.CLOSE_BOOST = P.fit_close_boost(hist, profs)
    game = [{'fav': args.fav, 'dog': args.dog, 'p_fav': args.p_fav, 'cbs_fav': 50.0}]

    known = {}
    for g in args.guesses:
        n, v = g.rsplit(':', 1); known[n.strip()] = float(v)

    rivals = []
    for name in args.tied + args.behind:
        pr = profs.get(name)
        p_fav_pick = P.predict({name: pr}, game)[name][0] if pr else 0.7
        hist_off = rival_guess_history(name)
        rivals.append({'name': name, 'tied': name in args.tied, 'p_fav_pick': p_fav_pick,
                       'known': known.get(name), 'hist': hist_off})
    print(f"week {args.week}: {us} vs {len(args.tied)} tied + {len(args.behind)} one back; "
          f"Monday {args.fav} ({args.p_fav:.0%}) vs {args.dog}, vegas total {args.total}, our pick {our_pick}")
    print(f"{'rival':24} {'tied':>4} {'P(picks fav)':>12} {'guess':>7}  past offsets from line")
    for r in rivals:
        g = f"{r['known']:.0f}" if r['known'] is not None else 'sampled'
        print(f"{r['name'][:24]:24} {str(r['tied']):>4} {r['p_fav_pick']:12.2f} {g:>7}  {np.round(r['hist'], 1).tolist()}")

    diffs = actual_minus_line()
    print(f"\nactual - closing line this season: n={len(diffs)}, mean {diffs.mean():+.1f}, median {np.median(diffs):+.1f}, sd {diffs.std():.1f}")

    # simulate
    actual = args.total + rng.choice(diffs, N_SIM) + rng.normal(0, 1.0, N_SIM)
    fav_wins = rng.random(N_SIM) < args.p_fav
    we_win_game = fav_wins if our_pick == args.fav else ~fav_wins
    # each rival: monday pick, whether they end tied with us after the game
    still = []
    guesses = []
    for r in rivals:
        picks_fav = rng.random(N_SIM) < r['p_fav_pick']
        r_wins = np.where(picks_fav, fav_wins, ~fav_wins)
        if r['tied']:
            tied_after = r_wins == we_win_game
        else:                                              # one back: tie only if they win and we lose
            tied_after = r_wins & ~we_win_game
        still.append(tied_after)
        if r['known'] is not None:
            guesses.append(np.full(N_SIM, r['known']))
        elif len(r['hist']):
            guesses.append(args.total + rng.choice(r['hist'], N_SIM) + rng.normal(0, 1.5, N_SIM))
        else:
            guesses.append(args.total + rng.normal(-1, 7, N_SIM))
    still = np.array(still); guesses = np.array(guesses)
    # note: rivals one back who DON'T tie but we lose and they win -> they beat us outright
    beaten = np.zeros(N_SIM, dtype=bool)
    for r, s in zip(rivals, still):
        if not r['tied']:
            pass  # handled: if they win and we lose they are tied (same score), covered by `still`

    lo, hi = int(np.floor(args.total)) - 15, int(np.ceil(args.total)) + 15
    print(f"\n{'guess':>5} {'P(win tiebreak)':>15}")
    best = (None, -1)
    results = []
    for N in range(lo, hi + 1):
        d = np.abs(actual - N)
        dr = np.where(still, np.abs(actual[None, :] - guesses), np.inf)   # rivals not tied don't compete
        dmin = dr.min(axis=0)
        n_tied_closer = (dr <= dmin + 1e-9).sum(axis=0)
        win = d < dmin
        split = np.isclose(d, dmin) & ~win
        p = win.mean() + (split / (n_tied_closer + 1)).mean()
        results.append((N, p))
        if p > best[1]:
            best = (N, p)
    for N, p in results:
        print(f"{N:>5} {p:15.3f}{'  <-- best' if N == best[0] else ''}")
    print(f"\nbest tiebreak for {us}: {best[0]}  (P(win | tie survives) {best[1]:.1%}; "
          f"P(no rival still tied) {(~still.any(axis=0)).mean():.1%})")


if __name__ == '__main__':
    main()
