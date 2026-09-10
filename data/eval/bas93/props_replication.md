## BAS-93 — hits props after fee, replicated on the July contracts

July window 2026-07-05 … 2026-07-30: 37360 settled contracts, 292 games, 601 players. Constants frozen: matchup weight 1.0, threshold 2%, tau 0.65, taker rate 0.07, flat 1u, DRAW_STREAMS=shared. Whole window, no half split.

### Vacuity check (run before scoring)

| check | value | threshold | verdict |
|---|---|---|---|
| settled hits contracts | 13578 | ≥ 10000 | pass |
| median candles per contract | 2.0 | ≥ 3 | **FAIL** |
| July stat mix vs August, worst ratio | 1.07× | ≤ 1.5× | pass |
| contracts with candles | 47189 of 47189 | — | — |

Hits closes, July vs August — what the thin path does and does not damage:

| | July | August |
|---|---|---|
| hits contracts | 13578 | 21665 |
| median candles/contract | 2.0 | 3.0 |
| median close, min before pitch | 15 | 15 |
| median pre-pitch volume | 4 | 20 |
| median whole-life volume | 211 | 256 |
| share that traded pre-pitch | 56.3% | 66.4% |
| settled yes/no | 99.4% | 99.0% |
| player ids resolved | 99.0% | 98.9% |

### Pre-registered predictions

| # | prediction | numbers | verdict |
|---|---|---|---|
| 1 | hits @ 2 pts: fee-waived ROI ≥ +4.0%, interval excludes zero, as-quoted > 0 | 4336 bets, fee-waived +0.6% (-4.4%, +6.1%), as-quoted -4.7% | **FAIL** |
| 2 | league-rate control negative as quoted and ≥ 5 pts below the model fee-waived | control 5520 bets, as-quoted -6.8%, fee-waived -1.6%; gap +2.2% | **FAIL** |
| 3 | strikeouts lose as quoted | 2631 bets, as-quoted -6.1% | **PASS** |

**Verdict.** The vacuity check fails, so every number below is descriptive and settles nothing. Nothing in serving. The ledger (BAS-77, Stage 0.1) is already running this rule on live tickets and it, not this backtest, decides Stage 1.

### Money, July, flat 1u, edge ≥ 2 pts (whole window)

| stat | model | n bets | hit | ROI fee-waived | 95% CI | ROI as-quoted | 95% CI |
|---|---|---|---|---|---|---|---|
| all | marcel_partial + matchup | 17537 | 0.561 | -1.4% | (-4.2%, +1.4%) | -5.8% | (-8.8%, -2.9%) |
| all | marcel_partial | 18522 | 0.540 | -3.1% | (-5.9%, -0.1%) | -7.7% | (-10.7%, -4.7%) |
| all | league_rate (control) | 21092 | 0.538 | -2.1% | (-4.9%, +0.7%) | -7.0% | (-9.9%, -4.1%) |
| all | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| all | random_edge (control) | 11757 | 0.430 | +0.6% | (-4.3%, +5.6%) | -8.6% | (-13.6%, -3.5%) |
| hits | marcel_partial + matchup | 4336 | 0.480 | +0.6% | (-4.4%, +6.1%) | -4.7% | (-10.0%, +1.0%) |
| hits | marcel_partial | 4992 | 0.462 | -2.0% | (-7.0%, +3.3%) | -7.5% | (-12.6%, -2.0%) |
| hits | league_rate (control) | 5520 | 0.466 | -1.6% | (-6.1%, +3.1%) | -6.8% | (-11.5%, -2.0%) |
| hits | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| hits | random_edge (control) | 4140 | 0.456 | +6.9% | (-1.3%, +15.6%) | -1.1% | (-9.2%, +7.7%) |
| hr | marcel_partial + matchup | 1464 | 0.522 | -6.7% | (-16.2%, +3.6%) | -13.2% | (-23.1%, -2.7%) |
| hr | marcel_partial | 1573 | 0.489 | -11.1% | (-20.4%, -1.9%) | -18.0% | (-27.4%, -8.5%) |
| hr | league_rate (control) | 3019 | 0.484 | -5.1% | (-12.3%, +2.7%) | -12.6% | (-20.0%, -4.8%) |
| hr | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| hr | random_edge (control) | 2441 | 0.328 | -20.2% | (-31.2%, -8.1%) | -38.9% | (-50.1%, -26.7%) |
| k | marcel_partial + matchup | 2631 | 0.574 | -1.2% | (-9.8%, +7.8%) | -6.1% | (-14.9%, +3.0%) |
| k | marcel_partial | 2861 | 0.529 | -5.8% | (-14.4%, +3.5%) | -11.1% | (-19.9%, -1.8%) |
| k | league_rate (control) | 3207 | 0.509 | -7.6% | (-15.6%, +0.9%) | -13.2% | (-21.3%, -4.5%) |
| k | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| k | random_edge (control) | 1377 | 0.477 | -6.4% | (-15.0%, +3.2%) | -12.5% | (-21.2%, -2.7%) |
| tb | marcel_partial + matchup | 9106 | 0.602 | -1.5% | (-3.5%, +0.6%) | -5.1% | (-7.1%, -3.0%) |
| tb | marcel_partial | 9096 | 0.594 | -1.4% | (-3.6%, +0.9%) | -5.0% | (-7.3%, -2.6%) |
| tb | league_rate (control) | 9346 | 0.607 | +0.4% | (-1.9%, +2.8%) | -3.2% | (-5.5%, -0.8%) |
| tb | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| tb | random_edge (control) | 3770 | 0.461 | +11.7% | (+4.5%, +18.7%) | +5.8% | (-1.3%, +12.9%) |

### Money, July, posterior rule at tau = 0.65

| stat | model | n bets | hit | ROI fee-waived | 95% CI | ROI as-quoted | 95% CI |
|---|---|---|---|---|---|---|---|
| all | marcel_partial + matchup | 26463 | 0.585 | -1.0% | (-3.6%, +1.6%) | -5.8% | (-8.5%, -3.1%) |
| hits | marcel_partial + matchup | 8297 | 0.501 | +2.5% | (-2.7%, +8.1%) | -3.9% | (-9.0%, +1.7%) |
| hr | marcel_partial + matchup | 4677 | 0.675 | -4.9% | (-9.0%, -0.7%) | -9.6% | (-13.9%, -5.4%) |
| k | marcel_partial + matchup | 2912 | 0.597 | -1.4% | (-9.4%, +7.1%) | -6.1% | (-14.4%, +2.3%) |
| tb | marcel_partial + matchup | 10577 | 0.608 | -2.0% | (-3.9%, -0.1%) | -5.5% | (-7.4%, -3.5%) |

### Brier per stat, July

| stat | n | games | over rate | current | matchup | market | league-rate |
|---|---|---|---|---|---|---|---|
| hits | 13324 | 291 | 0.325 | 0.16708 | 0.16676 | 0.16562 | 0.16776 |
| hr | 7184 | 291 | 0.084 | 0.07406 | 0.07402 | 0.07317 | 0.07458 |
| k | 3930 | 287 | 0.453 | 0.16767 | 0.16330 | 0.15629 | 0.18209 |
| tb | 12922 | 291 | 0.299 | 0.21509 | 0.21496 | 0.20607 | 0.21431 |
| all | 37360 | 292 | 0.283 | 0.16586 | 0.16523 | 0.16085 | 0.16745 |

### August (BAS-70) vs July (BAS-93), hits and pooled

BAS-70's published hits lead is its **second half**; BAS-93 scores July whole, so the like-for-like August comparator is *August whole*.

| window | stat | rule | n bets | ROI fee-waived | 95% CI | ROI as-quoted |
|---|---|---|---|---|---|---|
| August whole | hits | threshold @ 2 pts | 9004 | +3.7% | (-0.5%, +7.9%) | -1.8% |
| August whole | hits | posterior @ 0.65 | 15351 | +7.1% | (+2.4%, +12.2%) | -0.2% |
| August whole | hits | league-rate @ 2 pts | 10121 | +2.1% | (-1.8%, +6.1%) | -3.4% |
| August whole | all | threshold @ 2 pts | 33493 | -0.8% | (-3.1%, +1.6%) | -5.5% |
| August whole | all | posterior @ 0.65 | 48605 | +1.3% | (-1.2%, +3.9%) | -4.1% |
| August whole | all | league-rate @ 2 pts | 38365 | -0.5% | (-3.0%, +2.0%) | -5.6% |
| August first half | hits | threshold @ 2 pts | 4767 | -1.6% | (-7.2%, +4.2%) | -7.2% |
| August first half | hits | posterior @ 0.65 | 8273 | +2.8% | (-3.8%, +9.9%) | -5.0% |
| August first half | hits | league-rate @ 2 pts | 5335 | -0.9% | (-6.5%, +4.7%) | -6.5% |
| August first half | all | threshold @ 2 pts | 16737 | -2.2% | (-5.3%, +0.9%) | -7.0% |
| August first half | all | posterior @ 0.65 | 24693 | +0.2% | (-3.1%, +3.8%) | -5.3% |
| August first half | all | league-rate @ 2 pts | 19297 | -0.9% | (-4.4%, +2.7%) | -6.1% |
| August second half (BAS-70's headline) | hits | threshold @ 2 pts | 4237 | +9.5% | (+3.5%, +15.4%) | +4.3% |
| August second half (BAS-70's headline) | hits | posterior @ 0.65 | 7078 | +12.1% | (+5.8%, +19.1%) | +5.5% |
| August second half (BAS-70's headline) | hits | league-rate @ 2 pts | 4786 | +5.4% | (-0.2%, +11.0%) | +0.0% |
| August second half (BAS-70's headline) | all | threshold @ 2 pts | 16756 | +0.7% | (-2.8%, +4.0%) | -4.0% |
| August second half (BAS-70's headline) | all | posterior @ 0.65 | 23912 | +2.5% | (-1.0%, +6.2%) | -2.7% |
| August second half (BAS-70's headline) | all | league-rate @ 2 pts | 19068 | -0.0% | (-3.5%, +3.6%) | -5.1% |
| **July whole (BAS-93)** | hits | threshold @ 2 pts | 4336 | +0.6% | (-4.4%, +6.1%) | -4.7% |
| **July whole (BAS-93)** | all | threshold @ 2 pts | 17537 | -1.4% | (-4.2%, +1.4%) | -5.8% |

### Money, August (BAS-70's archive), whole window, same rule

| stat | model | n bets | hit | ROI fee-waived | 95% CI | ROI as-quoted | 95% CI |
|---|---|---|---|---|---|---|---|
| all | marcel_partial + matchup | 33493 | 0.532 | -0.8% | (-3.1%, +1.6%) | -5.5% | (-7.7%, -3.1%) |
| all | marcel_partial | 35043 | 0.514 | -0.6% | (-3.0%, +1.8%) | -5.6% | (-8.0%, -3.1%) |
| all | league_rate (control) | 38365 | 0.513 | -0.5% | (-3.0%, +2.0%) | -5.6% | (-8.1%, -3.1%) |
| all | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| all | random_edge (control) | 22720 | 0.442 | +4.3% | (+0.3%, +8.6%) | -5.5% | (-9.4%, -1.0%) |
| hits | marcel_partial + matchup | 9004 | 0.446 | +3.7% | (-0.5%, +7.9%) | -1.8% | (-6.0%, +2.5%) |
| hits | marcel_partial | 9711 | 0.438 | +2.5% | (-1.5%, +6.6%) | -3.2% | (-7.1%, +0.9%) |
| hits | league_rate (control) | 10121 | 0.439 | +2.1% | (-1.8%, +6.1%) | -3.4% | (-7.3%, +0.6%) |
| hits | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| hits | random_edge (control) | 7758 | 0.436 | +3.5% | (-2.0%, +9.5%) | -5.1% | (-10.5%, +0.9%) |
| hr | marcel_partial + matchup | 2600 | 0.255 | -0.6% | (-11.5%, +10.7%) | -11.0% | (-22.0%, +0.3%) |
| hr | marcel_partial | 2954 | 0.236 | -2.0% | (-12.2%, +8.9%) | -12.5% | (-22.6%, -1.7%) |
| hr | league_rate (control) | 4983 | 0.337 | -3.4% | (-10.8%, +4.3%) | -13.2% | (-20.7%, -5.5%) |
| hr | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| hr | random_edge (control) | 4225 | 0.315 | -17.4% | (-27.7%, -5.5%) | -40.4% | (-50.7%, -28.6%) |
| k | marcel_partial + matchup | 4358 | 0.564 | -7.1% | (-13.0%, -1.1%) | -11.5% | (-17.5%, -5.6%) |
| k | marcel_partial | 4748 | 0.532 | -5.1% | (-12.1%, +1.6%) | -10.0% | (-16.9%, -3.3%) |
| k | league_rate (control) | 5270 | 0.504 | -5.4% | (-12.5%, +1.3%) | -10.7% | (-17.8%, -3.9%) |
| k | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| k | random_edge (control) | 2457 | 0.485 | -6.8% | (-13.1%, -0.3%) | -12.4% | (-18.7%, -5.9%) |
| tb | marcel_partial + matchup | 17531 | 0.609 | -1.5% | (-3.3%, +0.3%) | -5.0% | (-6.8%, -3.2%) |
| tb | marcel_partial | 17630 | 0.597 | -0.8% | (-2.9%, +1.1%) | -4.5% | (-6.6%, -2.5%) |
| tb | league_rate (control) | 17991 | 0.605 | +0.4% | (-1.7%, +2.3%) | -3.3% | (-5.3%, -1.3%) |
| tb | market (control) | 0 | — | +0.0% | (+0.0%, +0.0%) | +0.0% | (+0.0%, +0.0%) |
| tb | random_edge (control) | 8460 | 0.471 | +11.5% | (+6.5%, +16.9%) | +5.9% | (+0.9%, +11.2%) |

### Brier per stat, August

| stat | n | games | over rate | current | matchup | market | league-rate |
|---|---|---|---|---|---|---|---|
| hits | 21088 | 455 | 0.325 | 0.16431 | 0.16389 | 0.16324 | 0.16503 |
| hr | 11892 | 455 | 0.078 | 0.06878 | 0.06866 | 0.06829 | 0.06935 |
| k | 6156 | 452 | 0.425 | 0.16767 | 0.16201 | 0.15191 | 0.17894 |
| tb | 22602 | 455 | 0.261 | 0.19403 | 0.19366 | 0.18579 | 0.19389 |
| all | 61738 | 455 | 0.264 | 0.15713 | 0.15626 | 0.15207 | 0.15855 |
