#!/usr/bin/env bash
# Downloads the OpenKE TransR pretrained embeddings (.npy), resuming on failure.
# Downloads into a tmp staging dir first, then moves the finished file into
# data/pretrained/ only after a successful, complete download.
set -uo pipefail

URL="https://thunlp.oss-cn-qingdao.aliyuncs.com/zzy/transr.npy"
TMP_DIR="${TMPDIR:-/tmp}/congrat_download_staging"
FINAL_DIR="$(dirname "$0")/data/pretrained"
TMP_OUT="$TMP_DIR/transr.npy"
FINAL_OUT="$FINAL_DIR/transr.npy"
MAX_RETRIES=10
POLL_INTERVAL=5 # seconds between speed samples

mkdir -p "$TMP_DIR" "$FINAL_DIR"

# Polls $TMP_OUT's size every POLL_INTERVAL seconds and prints a plain (non-carriage-return)
# timestamped speed line, so progress is visible even when stdout isn't a tty (e.g. logged
# via tee, run in background) where curl's own progress meter renders poorly.
monitor_speed() {
    local prev_size=0 cur_size delta
    while true; do
        sleep "$POLL_INTERVAL"
        cur_size=$(stat -c%s "$TMP_OUT" 2>/dev/null || echo 0)
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
    curl -L -C - --retry 5 --retry-delay 5 --speed-limit 51200 --speed-time 30 -o "$TMP_OUT" "$URL"
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

echo "Moving finished download from tmp staging into $FINAL_DIR ..."
mv "$TMP_OUT" "$FINAL_OUT"
echo "Done. File is at: $FINAL_OUT"
ls -lh "$FINAL_OUT"
