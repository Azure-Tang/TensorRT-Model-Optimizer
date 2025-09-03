#!/bin/bash
# 用法: ./run_with_monitor.sh python your_script.py --args

LOGFILE="monitor_$(date +%Y%m%d_%H%M%S).log"

# 每 5 秒记录一次 CPU/GPU 占用
(
  while true; do
    echo "=== $(date) ===" >> "$LOGFILE"
    free -h >> "$LOGFILE"            # 系统内存
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv >> "$LOGFILE" 2>/dev/null
    echo "" >> "$LOGFILE"
    sleep 5
  done
) &

MONITOR_PID=$!

# 跑你的命令
"$@"
EXIT_CODE=$?

# 停止监控
kill $MONITOR_PID 2>/dev/null

echo "=== Job finished ===" >> "$LOGFILE"
echo "Exit code: $EXIT_CODE" >> "$LOGFILE"
date >> "$LOGFILE"

echo "exit code:  $EXIT_CODE"

