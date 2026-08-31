# pick_em

Notebooks for weekly football pick'em: scrape the Las Vegas odds table from
vegasinsider.com, average the point spread across books, and (2026+) convert
moneylines to implied win probabilities.

- `pick_em_2026.ipynb` — current, NFL-only. Robust parsing: auto-detects book
  columns, splits the stacked spread/total/moneyline sections, and drops
  malformed or sign-flipped lines instead of crashing.
- `pick_em_2025.ipynb` — last season's version (CFB + NFL, with the
  officefootballpool.com spread comparison). Kept for reference; the scraping
  in it is broken against the current site HTML.
