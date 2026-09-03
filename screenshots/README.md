Weekly screenshots of the CBS pool picks page, one subdirectory per week
(`week_01`, `week_02`, ...). Gitignored except this README.

Workflow: drop screenshot(s) of the "% picking" bars here, then ask Claude to read
them into `cbs_pick_percents` in pick_em_2026.ipynb. Claude reads the numbers back
for a sanity check before editing the notebook. Team names must match the
vegasinsider tags (e.g. 'Vikings', not 'MIN'); the notebook's validation cell warns
on any mismatch.
