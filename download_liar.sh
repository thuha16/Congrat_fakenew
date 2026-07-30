#!/usr/bin/env bash
# Downloads the LIAR fake-news dataset, resuming on failure and stalls.
# Downloads into a tmp staging dir first, then moves/extracts the finished
# file into data/Liar/ only after a successful, complete download.
set -uo pipefail

URL="https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
TMP_DIR="${TMPDIR:-/tmp}/congrat_download_staging"
FINAL_DIR="$(dirname "$0")/data/Liar"
TMP_OUT="$TMP_DIR/liar_dataset.zip"
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
        printf '[%s] downloaded: %.2f MB total, %.2f MB/s\n' \
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

    # --speed-limit/--speed-time: abort (so this retry loop can restart the connection)
    # if the transfer stalls below 50KB/s for 30s straight, since a stalled but
    # still-open connection never triggers curl's own --retry.
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

echo "Extracting into $FINAL_DIR ..."
python3 -m zipfile -e "$TMP_OUT" "$FINAL_DIR/"
echo "Contents:"
find "$FINAL_DIR" -maxdepth 3 -type f | sort

echo "Cleaning up tmp staging file..."
rm -f "$TMP_OUT"
