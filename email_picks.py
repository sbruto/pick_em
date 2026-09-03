#!/usr/bin/env python
"""Email Hannah's (Entry A) picks: the vegas favorite in every game.

Scheduled by launchd Tuesday and Thursday 8:30am (see launchd/). The Tuesday run saves
the picks; the Thursday run diffs against them and flags anything that changed.

    python email_picks.py              # scheduled run: send to everyone in config
    python email_picks.py --dry-run    # print the email, send nothing, save nothing
    python email_picks.py --to me      # send only to the first address in config (Sean)

config.json (gitignored; see config.example.json):
    to                    list of recipient addresses; the first one is "me"
    sender                From address as Mail.app knows it, e.g. "Sean <x@y.edu>"
    season_week1_tuesday  ISO date of the Tuesday of Week 1 (runs before it count as Week 1)
    state_dir             where week_NN.json snapshots and the log go (gitignored)
"""
import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickem  # noqa: E402


def load_config():
    with open(os.path.join(HERE, 'config.json')) as f:
        return json.load(f)


def week_number(cfg, today=None):
    today = today or dt.date.today()
    start = dt.date.fromisoformat(cfg['season_week1_tuesday'])
    return max(1, (today - start).days // 7 + 1)   # preseason runs count as Week 1


def load_previous(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


RED = '\x00'   # marker prefix: this line gets colored red in Mail (stripped before sending)


def build_email(week, games, tiebreak, prev, updated):
    """Return (subject, body, changed_line_indexes) as plain text.

    Layout: copy/paste list first, then what changed since the last email, then the
    details. Lines that changed are colored red in Mail (no text markers).
    """
    prev_picks = {(g['away'], g['home']): g['pick'] for g in (prev or {}).get('games', [])}
    prev_tb = (prev or {}).get('tiebreak')
    tb_num = int(math.floor(tiebreak['total']))
    old_tb = int(math.floor(prev_tb['total'])) if prev_tb else None

    changed = {(g['away'], g['home']) for g in games
               if (g['away'], g['home']) in prev_picks and prev_picks[(g['away'], g['home'])] != g['pick']}
    tb_changed = old_tb is not None and old_tb != tb_num

    subject = f"Hannah's picks: Week {week}" + (" (updated Thu)" if updated else "")
    L = []
    L.append(f"Hannah's picks, Week {week}" + (" (Thursday update)" if updated else ""))
    L.append("")
    for g in games:
        L.append((RED if (g['away'], g['home']) in changed else "") + g['pick'])
    L.append((RED if tb_changed else "") + f"tiebreak {tb_num}")
    L.append("")

    if prev is not None:
        if changed or tb_changed:
            L.append("Changes since the last email:")
            for g in games:
                k = (g['away'], g['home'])
                if k in changed:
                    L.append(RED + f"  {g['away']} @ {g['home']}: now {g['pick']} (was {prev_picks[k]})")
            if tb_changed:
                L.append(RED + f"  tiebreaker: now {tb_num} (was {old_tb})")
        else:
            L.append("No changes since the last email.")
        L.append("")

    L.append(f"Details (vegas favorite in every game; spread, win probability). "
             f"Generated {dt.datetime.now():%a %b %d %H:%M}.")
    for g in games:
        L.append((RED if (g['away'], g['home']) in changed else "") +
                 f"{g['away']:>12} @ {g['home']:<12} ->  {g['pick']:<10} ({g['spread']:+.1f}, {g['p_pick']:.0%})")
    L.append((RED if tb_changed else "") +
             f"Tiebreaker: {tiebreak['away']} @ {tiebreak['home']}, vegas total {tiebreak['total']:.1f} -> enter {tb_num}")

    # color indexes are computed on the FINAL line list, so they match Mail's paragraphs
    changed_idx = [i for i, ln in enumerate(L) if ln.startswith(RED)]
    body = "\n".join(ln[1:] if ln.startswith(RED) else ln for ln in L)
    return subject, body, changed_idx


def send_mail(subject, body, to, sender, changed_idx=()):
    """Send via Mail.app. Plain text; changed lines colored red as a best effort."""
    def q(s):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    recips = "\n".join(
        f'        make new to recipient at end of to recipients with properties {{address:{q(a)}}}'
        for a in to)
    # AppleScript paragraphs are 1-indexed; body lines map 1:1
    color = "\n".join(
        f'        try\n            set color of paragraph {i + 1} of content to {{65535, 0, 0}}\n        end try'
        for i in changed_idx)
    script = f'''
tell application "Mail"
    set msg to make new outgoing message with properties {{subject:{q(subject)}, content:{q(body)}, visible:false, sender:{q(sender)}}}
    tell msg
{recips}
{color}
        send
    end tell
end tell
'''
    subprocess.run(['osascript', '-e', script], check=True, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--to', choices=['me', 'all'], default='all')
    args = ap.parse_args()

    cfg = load_config()
    to = cfg['to'][:1] if args.to == 'me' else cfg['to']
    state_dir = os.path.join(HERE, cfg.get('state_dir', 'picks'))
    os.makedirs(state_dir, exist_ok=True)
    week = week_number(cfg)
    snap_path = os.path.join(state_dir, f'week_{week:02d}.json')
    stamp = f"{dt.datetime.now():%Y-%m-%d %H:%M}"

    try:
        games, tiebreak = pickem.chalk_picks(cfg.get('url', pickem.URL))
        if not games:
            raise RuntimeError("scrape returned no games with moneyline data")
    except Exception:
        err = traceback.format_exc()
        print(err, file=sys.stderr)
        subject = f"Hannah's picks: Week {week} FAILED"
        body = f"email_picks.py could not build the picks.\n\n{err}"
        if args.dry_run:
            print(subject, body, sep="\n")
        else:
            send_mail(subject, body, cfg['to'][:1], cfg['sender'])
            with open(os.path.join(state_dir, 'log.txt'), 'a') as f:
                f.write(f"{stamp} week {week} FAILED\n")
        return 0

    prev = load_previous(snap_path)
    updated = prev is not None
    subject, body, changed_idx = build_email(week, games, tiebreak, prev, updated)

    if args.dry_run:
        print(f"To: {', '.join(to)}\nSubject: {subject}\n\n{body}")
        return 0

    send_mail(subject, body, to, cfg['sender'], changed_idx)
    with open(snap_path, 'w') as f:
        json.dump({'week': week, 'generated': stamp, 'games': games, 'tiebreak': tiebreak}, f, indent=1)
    with open(os.path.join(state_dir, 'log.txt'), 'a') as f:
        f.write(f"{stamp} week {week} sent to {', '.join(to)}; {len(changed_idx)} change(s)\n")
    print(f"sent '{subject}' to {', '.join(to)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
