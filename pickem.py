"""Shared scraping / cleaning code for the NFL pick'em notebook and the emailer.

Single data source: the Las Vegas odds table on vegasinsider.com. The page stacks the
spread, total and moneyline sections in ONE table (rotation numbers restart at each
section) and the HTML is flaky, so everything here validates each cell and drops
inconsistent data loudly (print) rather than crashing. See CLAUDE.md for the details.
"""
import re

import numpy as np
import pandas as pd

URL = "https://www.vegasinsider.com/nfl/odds/las-vegas/"


def parse_line(cell):
    """Pull the leading number out of a cell like '+3.5 -110 +'.
    NaN for anything malformed (e.g. the '--4.5' junk the site sometimes renders)."""
    if not isinstance(cell, str):
        return np.nan
    tok = cell.split()[0]
    if tok.upper() in ('PK', 'PICK', 'EVEN'):
        return 0.0
    if re.fullmatch(r'[+-]?\d+(?:\.\d+)?', tok):
        return float(tok)
    return np.nan


def parse_ml(cell):
    """Moneyline: 'EVEN' / 'EV' mean +100 (parse_line would give 0, which is a spread
    convention and poisons the average). Anything with |odds| < 100 is a bad render."""
    if not isinstance(cell, str):
        return np.nan
    tok = cell.split()[0].upper()
    if tok in ('EVEN', 'EV', 'PK', 'PICK'):
        return 100.0
    v = parse_line(cell)
    return v if abs(v) >= 100 else np.nan


def implied_prob(m):
    """American odds -> implied probability (vig included)."""
    return 100 / (m + 100) if m > 0 else -m / (-m + 100)


def parse_total(cell):
    """Pull the total out of an over/under cell like 'o47.5 -110 +' or 'u 47.5 -105'."""
    if not isinstance(cell, str):
        return np.nan
    m = re.match(r'^[ou]\s*(\d+(?:\.\d+)?)', cell.strip(), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def load_sections(tab):
    """The page stacks the spread / total / moneyline sections into one table, with
    rotation numbers restarting at each section. Split them apart, keeping raw cell
    strings (each section parses its own format). Book columns are auto-detected,
    so no more hand-maintaining the sources list."""
    sources = [c for c in tab.columns
               if c not in ('Time', 'Open') and not str(c).startswith('Unnamed')]
    # Completed games are moved to the END of each section under a 'Final' marker row,
    # with their closing lines still shown, so they would otherwise look like live games
    # (and the "last game listed" tiebreaker logic would pick them). Drop them. Second
    # guard: rotation numbers ascend within a section, so a drop in rotation number also
    # means we've hit the finished games.
    rows, seen, section, final, last_rot, dropped = [], set(), 0, False, -1, []
    for _, r in tab.iterrows():
        t = r['Time']
        if not isinstance(t, str):
            continue
        if re.match(r'^(Final|In Progress|Live|Postponed)', t.strip(), re.IGNORECASE):
            final = True
            continue
        m = re.match(r'^(\d+)\s+(.*\S)', t)      # team rows look like '451 Patriots'
        if not m:
            continue                              # skips 'Matchup', header junk
        rot = int(m.group(1))
        if rot in seen:                           # rotation number repeated -> new section
            section += 1
            seen, final, last_rot = set(), False, -1
        seen.add(rot)
        if rot < last_rot and rot % 2 == 1:       # away team out of order -> finished game
            final = True
        last_rot = rot
        if final:
            if section == 0:
                dropped.append(m.group(2))
            continue
        rows.append({'section': section, 'Team': m.group(2),
                     **{s: r[s] for s in sources}})
    if dropped:
        print(f"dropping finished game(s): {dropped}")
    return pd.DataFrame(rows), sources



def drop_swapped(df, sources, mirror, min_consensus, label='lines'):
    """The site sometimes renders a book's column with the two teams swapped
    (confirmed against the book's own site for HardRock). A swapped value sits near
    the *mirror image* of the consensus, so flag a value only if it is closer to
    mirror(median) than to the median AND the consensus is far enough from a toss-up
    that the two are distinguishable. Books genuinely disagreeing about who is
    favored in a pick'em game are kept: that disagreement is real information.

    mirror: function giving the other side's value (-x for spreads, 1-p for probs)
    min_consensus: |median - mirror(median)| must exceed this to flag anything
    """
    df = df.reset_index(drop=True).copy()
    med = df[sources].median(axis=1)
    decidable = (med - mirror(med)).abs() >= min_consensus
    for s in sources:
        v = df[s]
        swapped = decidable & ((v - mirror(med)).abs() < (v - med).abs())
        if swapped.any():
            print(f"{s}: ignoring swapped {label} for {df.loc[swapped, 'Team'].tolist()}")
        df.loc[swapped, s] = np.nan
    return df


def report_splits(df, sources):
    """Print games where the surviving books disagree on who is favored."""
    for g in range(0, len(df) - 1, 2):
        row = df.loc[g, sources]
        neg = [s for s in sources if row[s] < 0]
        pos = [s for s in sources if row[s] > 0]
        if neg and pos:
            print(f"books split: {df.loc[g, 'Team']} favored by {neg} | "
                  f"{df.loc[g + 1, 'Team']} favored by {pos}")


def clean_spreads(df, sources):
    """Games are consecutive row pairs, and a book's two lines in a game must be
    mirror images (+3.5 / -3.5). Anything else is a bad render -> NaN both sides."""
    df = df.reset_index(drop=True).copy()
    for g in range(0, len(df) - 1, 2):
        for s in sources:
            a, b = df.loc[g, s], df.loc[g + 1, s]
            if np.isnan(a) or np.isnan(b) or a != -b:
                df.loc[[g, g + 1], s] = np.nan
    # a swapped column reads e.g. +3.5 where consensus is -3.5; only decidable when the
    # consensus is >= 2.5 points from pick'em (so |med - (-med)| >= 5)
    df = drop_swapped(df, sources, mirror=lambda x: -x, min_consensus=5.0, label='spreads')
    df = df.dropna(subset=sources, how='all')      # drops games already final
    report_splits(df.reset_index(drop=True), sources)
    return df


# ---------------------------------------------------------------------------
# Wrappers used by email_picks.py (the notebook does the same steps cell by cell)
# ---------------------------------------------------------------------------

def fetch_odds(url=URL):
    """Scrape the page and split it into sections.

    Returns (raw, full, sources, spread_sections, ml_sections, total_sections):
    raw has the cell strings, full has parse_line applied to every book column.
    """
    tab = pd.read_html(url)[0]
    raw, sources = load_sections(tab)
    full = raw.copy()
    for s in sources:
        full[s] = raw[s].map(parse_line)
    # classify sections by typical magnitude: spreads are small, moneylines are +-100 and
    # up, and totals ('o47.5 ...') never parse with parse_line at all
    med_abs = full.groupby('section')[sources].apply(lambda d: d.abs().median().median())
    spread_sections = med_abs[med_abs < 50].index
    ml_sections = med_abs[med_abs >= 100].index
    total_sections = med_abs[med_abs.isna()].index
    return raw, full, sources, spread_sections, ml_sections, total_sections


def spread_table(full, sources, spread_sections):
    spreads = clean_spreads(full[full['section'].isin(spread_sections)], sources)
    spreads['ave_spread'] = spreads[sources].mean(axis=1)
    spreads['n_books'] = spreads[sources].notna().sum(axis=1)
    return spreads


def win_prob_table(raw, full, sources, ml_sections):
    """Per-book implied probabilities, swapped columns dropped, averaged, vig removed."""
    ml = full[full['section'].isin(ml_sections)].reset_index(drop=True).copy()
    raw_ml = raw[raw['section'].isin(ml_sections)].reset_index(drop=True)
    for s in sources:
        ml[s] = raw_ml[s].map(parse_ml).map(lambda m: implied_prob(m) if not np.isnan(m) else np.nan)
    ml = drop_swapped(ml, sources, mirror=lambda p: 1 - p, min_consensus=0.10, label='moneylines')
    ml = ml.dropna(subset=sources, how='all').reset_index(drop=True)
    ml['q'] = ml[sources].mean(axis=1)
    ml['n_books'] = ml[sources].notna().sum(axis=1)
    for g in range(0, len(ml) - 1, 2):
        tot = ml.loc[g, 'q'] + ml.loc[g + 1, 'q']
        ml.loc[[g, g + 1], 'win_prob'] = ml.loc[[g, g + 1], 'q'] / tot
    return ml


def total_table(raw, sources, total_sections):
    tot = raw[raw['section'].isin(total_sections)].reset_index(drop=True).copy()
    for s in sources:
        tot[s] = tot[s].map(parse_total)
    tot['ave_total'] = tot[sources].mean(axis=1)
    return tot.dropna(subset=['ave_total'])


def chalk_picks(url=URL):
    """Entry A: the vegas favorite (higher vig-free win prob) in every game, site order.

    Returns (games, tiebreak) where games is a list of dicts
    {'away', 'home', 'pick', 'p_pick', 'spread'} and tiebreak is
    {'away', 'home', 'total'} for the last game listed. Games with no moneyline data
    (already final) are skipped.
    """
    raw, full, sources, spread_sections, ml_sections, total_sections = fetch_odds(url)
    sp = spread_table(full, sources, spread_sections).set_index('Team')['ave_spread']
    wp = win_prob_table(raw, full, sources, ml_sections).set_index('Team')['win_prob']
    tot = total_table(raw, sources, total_sections)

    order = full[full['section'].isin(spread_sections)].reset_index(drop=True)
    games = []
    for g in range(0, len(order) - 1, 2):
        away, home = order.loc[g, 'Team'], order.loc[g + 1, 'Team']
        pa, ph = wp.get(away, np.nan), wp.get(home, np.nan)
        if np.isnan(pa) or np.isnan(ph):
            continue
        pick = home if ph >= pa else away
        games.append({'away': away, 'home': home, 'pick': pick,
                      'p_pick': float(max(pa, ph)), 'spread': float(sp.get(pick, np.nan))})

    tiebreak = {'away': tot['Team'].iloc[-2], 'home': tot['Team'].iloc[-1],
                'total': float(tot['ave_total'].iloc[-2:].mean())}
    return games, tiebreak
