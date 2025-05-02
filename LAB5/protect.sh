#!/bin/bash

# 參數設定
SWAP_THRESHOLD=95    # 當Swap使用超過70%，就殺python
CHECK_INTERVAL=5     # 每隔5秒檢查一次

while true; do
    # 讀取swap使用情況
    swap_total=$(free | awk '/Swap:/ {print $2}')
    swap_used=$(free | awk '/Swap:/ {print $3}')

    # 如果沒有swap區（swap_total == 0），直接跳過
    if [ "$swap_total" -eq 0 ]; then
        echo "No swap configured."
        sleep $CHECK_INTERVAL
        continue
    fi

    # 計算swap使用百分比
    swap_percent=$(awk "BEGIN {printf \"%.2f\", ($swap_used/$swap_total)*100}")

    echo "Current SWAP Usage: $swap_percent%"

    # 如果超過門檻，開始處理
    if (( $(echo "$swap_percent > $SWAP_THRESHOLD" | bc -l) )); then
        echo "!!! Swap usage critical ($swap_percent%). Killing Python processes..."

        # 只殺 "python" or "python3" 的process
        pids=$(pgrep -f "python")
        
        if [ -z "$pids" ]; then
            echo "No python process found."
        else
            for pid in $pids; do
                echo "Killing PID $pid"
                kill -9 $pid
            done
        fi

        # 等待一下讓系統釋放資源
        sleep 10

        # optional: 再清一下pagecache（有時Linux不會馬上釋放）
        echo 3 | sudo tee /proc/sys/vm/drop_caches
        echo "Dropped caches to free memory."

        # 殺完休息久一點再回圈
        sleep 30
    else
        sleep $CHECK_INTERVAL
    fi
done
