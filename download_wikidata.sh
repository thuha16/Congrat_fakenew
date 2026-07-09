#!/usr/bin/env bash
# Downloads the OpenKE Wikidata pretrained embeddings archive, resuming on failure.
set -uo pipefail

URL="https://thunlp.oss-cn-qingdao.aliyuncs.com/openke/Wikidata.zip"
DEST_DIR="$(dirname "$0")/data/pretrained"
OUT="$DEST_DIR/Wikidata.zip"
MAX_RETRIES=10
POLL_INTERVAL=5 # seconds between speed samples

mkdir -p "$DEST_DIR"

# Polls $OUT's size every POLL_INTERVAL seconds and prints a plain (non-carriage-return)
# timestamped speed line, so progress is visible even when stdout isn't a tty (e.g. logged
# via tee, run in background) where curl's own progress meter renders poorly.
monitor_speed() {
    local prev_size=0 cur_size delta
    while true; do
        sleep "$POLL_INTERVAL"
        cur_size=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
        delta=$(( cur_size - prev_size ))
        printf '[%s] downloaded: %.1f MB total, %.2f MB/s\n' \
            "$(date +%H:%M:%S)" \
            "$(echo "$cur_size / 1048576" | bc -l)" \
            "$(echo "$delta / $POLL_INTERVAL / 1048576" | bc -l)"
        prev_size=$cur_size
    done
}

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo "Attempt $attempt/$MAX_RETRIES..."

    monitor_speed &
    monitor_pid=$!

    # --speed-limit/--speed-time: abort (so the retry loop below can restart the
    # connection) if the transfer stalls below 50KB/s for 30s straight, since a stalled
    # but still-open connection never triggers curl's own --retry.
    curl -L -C - --retry 5 --retry-delay 5 --speed-limit 51200 --speed-time 30 -o "$OUT" "$URL"
    status=$?

    kill "$monitor_pid" 2>/dev/null
    wait "$monitor_pid" 2>/dev/null

    if [ $status -eq 0 ]; then
        echo "Download finished successfully."
        break
    fi
    echo "curl exited with status $status, retrying..."
    sleep 5
done

if [ $status -ne 0 ]; then
    echo "Failed after $MAX_RETRIES attempts." >&2
    exit 1
fi

echo "Extracting..."
python3 -m zipfile -e "$OUT" "$DEST_DIR/"
echo "Contents:"
find "$DEST_DIR" -maxdepth 3 -type f | sort
