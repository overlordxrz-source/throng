#!/usr/bin/env python3
import re

# Withdrawal test
with open('runs_withdrawal/run_20260518_233539/science.log') as f:
    text = f.read()
steps_wd = re.findall(r'step=\s+([\d,]+)\s+blue=\d+\s+red=\d+\s+brain=\dL\s+ppo=\d+\s+surv=([\d.]+)', text)
steps_wd = [(int(s.replace(',','')), float(v)) for s,v in steps_wd if int(s.replace(',','')) >= 50000]

print('=== WITHDRAWAL TEST (signals zeroed) ===')
for s,v in steps_wd:
    print(f'  step {s:>6}: surv={v:.2f}')
avg_wd = sum(v for _,v in steps_wd) / len(steps_wd) if steps_wd else 0
print(f'  Average survival: {avg_wd:.3f}')

# Hardmax test
with open('runs_large/run_20260518_233539/hardmax_mps_log.txt') as f:
    text = f.read()
steps_hm = re.findall(r'step=\s+([\d,]+)\s+blue=\d+\s+red=\d+\s+brain=\dL\s+ppo=\d+\s+surv=([\d.]+)', text)
steps_hm = [(int(s.replace(',','')), float(v)) for s,v in steps_hm if int(s.replace(',','')) >= 50000]

print('\n=== HARDMAX TEST (discrete tokens) ===')
for s,v in steps_hm:
    print(f'  step {s:>6}: surv={v:.2f}')
avg_hm = sum(v for _,v in steps_hm) / len(steps_hm) if steps_hm else 0
print(f'  Average survival: {avg_hm:.3f}')

print('\n=== COMPARISON ===')
print(f'  Withdrawal avg: {avg_wd:.3f}')
print(f'  Hardmax avg:    {avg_hm:.3f}')
print(f'  Difference:      {avg_hm - avg_wd:+.3f}')
