#!/usr/bin/env python3
import psutil
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
    try:
        cmd = ' '.join(p.info['cmdline'] or [])
        if 'main.py' in cmd:
            print(f'PID={p.info["pid"]} MEM={p.info["memory_info"].rss/1024/1024:.1f}MB CPU={p.info["cpu_percent"]:.1f}%')
            print(f'  CMD: {cmd}')
    except Exception:
        pass
