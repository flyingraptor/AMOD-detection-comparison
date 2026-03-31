#!/bin/bash
# Extracts the AMOD split archive while deleting each part
# as soon as 7z finishes reading it, to save disk space.

set -e
cd /home/flyingalien/Repositories/AMOD/data/

ARCHIVE="AMOD_For_Distribution.zip"
OUTDIR="/home/flyingalien/Repositories/AMOD/data"

echo "=== AMOD Extract + Free script ==="
echo "Output dir: $OUTDIR"
echo "Starting 7z extraction in background..."

7z x "$ARCHIVE" -o"$OUTDIR" &
PID=$!

echo "7z PID: $PID"
echo "Waiting for 7z to start reading parts..."
sleep 15

PREV_PART=""
CURRENT_PART=""

while kill -0 $PID 2>/dev/null; do
    # Find which .z* part (not .zip) 7z currently has open
    CURRENT_PART=$(lsof -p $PID 2>/dev/null \
        | grep -o 'AMOD_For_Distribution\.z[0-9]*[^i]' \
        | grep -v '\.zip' \
        | head -1 || true)

    # When 7z moves to a new part, delete the previous one
    if [ -n "$CURRENT_PART" ] && \
       [ "$CURRENT_PART" != "$PREV_PART" ] && \
       [ -n "$PREV_PART" ] && \
       [ -f "$PREV_PART" ]; then
        echo "[$(date +%H:%M:%S)] 7z moved to $CURRENT_PART — deleting $PREV_PART ($(du -sh "$PREV_PART" | cut -f1))"
        rm -f "$PREV_PART"
        df -h "$OUTDIR" | tail -1 | awk '{print "  Free space: " $4}'
    fi

    PREV_PART="$CURRENT_PART"
    sleep 5
done

wait $PID
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo ""
    echo "=== Extraction complete! Cleaning up remaining archive files... ==="
    rm -f AMOD_For_Distribution.z* AMOD_For_Distribution.zip
    echo "All archive parts deleted."
    echo "Final free space: $(df -h $OUTDIR | tail -1 | awk '{print $4}')"
else
    echo "=== 7z exited with error code $EXIT ==="
fi
