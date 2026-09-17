#!/usr/bin/env python
"""Parse the CBS pool standings page (copy/pasted text) into a CSV and score it.

    python league_log.py league/week_01_paste.txt [--chalk SEA ...]

--chalk adds vegas favorites missing from the snapshot (e.g. a game that had already
finished when Thursday's emailer run overwrote picks/week_NN.json).

Paste format (one field per line, repeated per entry):
    <rank>  <name>  "<weekly pts>    <ytd pts>"  <16 team abbrs in CBS game order>  <tiebreak|->

Writes league/week_NN.csv (one row per entry) and prints:
  * the actual results and the chalk (vegas favorite) line for the week, from the
    emailer's snapshot picks/week_NN.json and ESPN's public scoreboard
  * every entry's deviations from chalk, tiebreak, and distance to the actual total
  * a summary of who looks like a vegas follower ("sharp")

league/ is gitignored (league members' names).
"""
import csv
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

ABBR = {'Patriots': 'NE', 'Seahawks': 'SEA', '49ers': 'SF', 'Rams': 'LAR', 'Buccaneers': 'TB',
        'Bengals': 'CIN', 'Bills': 'BUF', 'Texans': 'HOU', 'Ravens': 'BAL', 'Colts': 'IND',
        'Bears': 'CHI', 'Panthers': 'CAR', 'Saints': 'NO', 'Lions': 'DET', 'Browns': 'CLE',
        'Jaguars': 'JAC', 'Falcons': 'ATL', 'Steelers': 'PIT', 'Jets': 'NYJ', 'Titans': 'TEN',
        'Packers': 'GB', 'Vikings': 'MIN', 'Commanders': 'WAS', 'Eagles': 'PHI', 'Dolphins': 'MIA',
        'Raiders': 'LV', 'Cardinals': 'ARI', 'Chargers': 'LAC', 'Cowboys': 'DAL', 'Giants': 'NYG',
        'Broncos': 'DEN', 'Chiefs': 'KC'}
ESPN_ABBR_FIX = {'JAX': 'JAC', 'WSH': 'WAS'}


def parse_paste(path):
    lines = [l.rstrip('\n') for l in open(path) if l.strip()]
    entries, i = [], 0
    rank_re = re.compile(r'^\d+(st|nd|rd|th)$')
    while i < len(lines):
        if not rank_re.match(lines[i].strip()):
            i += 1
            continue
        rank = int(re.match(r'\d+', lines[i].strip()).group())
        name = lines[i + 1].strip()
        pts = lines[i + 2].split()
        week_pts, ytd_pts = int(pts[0]), int(pts[1])
        picks = [l.strip() for l in lines[i + 3:i + 19]]
        tb = lines[i + 19].strip()
        entries.append({'rank': rank, 'name': name, 'week_pts': week_pts, 'ytd_pts': ytd_pts,
                        'picks': picks, 'tiebreak': int(tb) if tb.isdigit() else None})
        i += 20
    return entries


def espn_results(week, season=2026):
    """{'WINNER_ABBR': (away, home, away_score, home_score)} for completed games + last-game total."""
    url = (f"https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
           f"?seasontype=2&week={week}&dates={season}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    d = json.load(urllib.request.urlopen(req, timeout=30))
    games = []
    for e in d['events']:
        comp = e['competitions'][0]
        if not comp['status']['type']['completed']:
            continue
        c = {x['homeAway']: x for x in comp['competitors']}
        ab = lambda x: ESPN_ABBR_FIX.get(x['team']['abbreviation'], x['team']['abbreviation'])
        games.append({'date': e['date'], 'away': ab(c['away']), 'home': ab(c['home']),
                      'away_pts': int(c['away']['score']), 'home_pts': int(c['home']['score'])})
    games.sort(key=lambda g: g['date'])
    winners = {(g['home'] if g['home_pts'] > g['away_pts'] else g['away']) for g in games}
    last = games[-1]
    return winners, last['away_pts'] + last['home_pts'], games


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--chalk', nargs='*', default=[], help='extra vegas favorites (abbr) missing from snapshot')
    args = ap.parse_args()
    path = args.path
    week = int(re.search(r'week_(\d+)', path).group(1))
    entries = parse_paste(path)
    print(f"parsed {len(entries)} entries for week {week}")

    # chalk line from the emailer's snapshot (vegas favorite per game)
    snap = json.load(open(os.path.join(HERE, 'picks', f'week_{week:02d}.json')))
    chalk = {ABBR[g['pick']] for g in snap['games']} | set(args.chalk)
    vegas_total = snap['tiebreak']['total']

    winners, actual_total, games = espn_results(week)
    print(f"results: {len(games)} games final; last game total {actual_total} (vegas {vegas_total:.1f})")
    print(f"chalk losses: {sorted(chalk - winners)}\n")

    all_teams = {t for e in entries for t in e['picks']}
    rows = []
    for e in entries:
        picks = set(e['picks'])
        dev = sorted(picks - chalk)                   # picks that were NOT the vegas favorite
        correct = len(picks & winners)
        rows.append({'week': week, 'rank': e['rank'], 'name': e['name'], 'week_pts': e['week_pts'],
                     'ytd_pts': e['ytd_pts'], 'scored': correct, 'n_dev': len(dev),
                     'deviations': ' '.join(dev), 'tiebreak': e['tiebreak'],
                     'tb_err': None if e['tiebreak'] is None else e['tiebreak'] - actual_total,
                     'tb_vs_vegas': None if e['tiebreak'] is None else round(e['tiebreak'] - vegas_total, 1),
                     'picks': ' '.join(e['picks'])})

    out = os.path.join(HERE, 'league', f'week_{week:02d}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}\n")

    print(f"{'rank':>4} {'name':26} {'pts':>3} {'chk':>3} {'dev':>3}  {'tb':>4} {'tb-vegas':>8}  deviations")
    for r in rows:
        flag = ' *' if r['scored'] != r['week_pts'] else ''
        print(f"{r['rank']:>4} {r['name'][:26]:26} {r['week_pts']:>3} {r['scored']:>3}{flag} {r['n_dev']:>3}  "
              f"{str(r['tiebreak']):>4} {str(r['tb_vs_vegas']):>8}  {r['deviations']}")

    # deviation popularity
    from collections import Counter
    c = Counter(t for r in rows for t in r['deviations'].split())
    print("\nnon-chalk picks by count:", ', '.join(f"{t} {n}" for t, n in c.most_common()))
    near = [r['name'] for r in rows if r['tb_vs_vegas'] is not None and abs(r['tb_vs_vegas']) <= 1.5]
    print(f"tiebreak within 1.5 of vegas total ({len(near)}): {', '.join(near)}")


if __name__ == '__main__':
    main()
