#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# start.sh — launch Regent model server + Nuxt UI
#
# Usage:
#   ./start.sh [options]
#
# Options:
#   --config    <path>   Model config YAML       (default: configs/regent_370m.yaml)
#   --model     <path>   Weights .safetensors    (default: none — random init)
#   --tokenizer <path>   SentencePiece .model    (default: data/tokenizer/regent.model)
#   --port      <int>    API server port         (default: 8400)
#   --ui-port   <int>    Nuxt dev server port    (default: 3000)
#   --help               Show this message
# ---------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
CONFIG="$ROOT/configs/regent_370m.yaml"
MODEL=""
TOKENIZER="$ROOT/data/tokenizer/regent.model"
PORT=8400
UI_PORT=3000

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)     CONFIG="$2";    shift 2 ;;
    --model)      MODEL="$2";     shift 2 ;;
    --tokenizer)  TOKENIZER="$2"; shift 2 ;;
    --port)       PORT="$2";      shift 2 ;;
    --ui-port)    UI_PORT="$2";   shift 2 ;;
    --help)
      sed -n '4,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi

# Warn (don't abort) if tokenizer is missing — server will load without it
if [[ ! -f "$TOKENIZER" ]]; then
  echo "WARN: tokenizer not found at $TOKENIZER — server will start without it"
  TOKENIZER=""
fi

if [[ ! -d "$ROOT/ui/node_modules" ]]; then
  echo "── Installing UI dependencies…"
  (cd "$ROOT/ui" && npm install --silent)
fi

# ---------------------------------------------------------------------------
# PIDs tracked for cleanup
# ---------------------------------------------------------------------------

SERVER_PID=""
UI_PID=""

cleanup() {
  echo ""
  echo "── Shutting down…"
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && echo "   server stopped (pid $SERVER_PID)"
  [[ -n "$UI_PID"     ]] && kill "$UI_PID"     2>/dev/null && echo "   UI stopped     (pid $UI_PID)"
  exit 0
}

trap cleanup SIGINT SIGTERM

# ---------------------------------------------------------------------------
# Build server command
# ---------------------------------------------------------------------------

SERVER_CMD=(python3 -m serve.server --config "$CONFIG" --port "$PORT")
[[ -n "$MODEL"     ]] && SERVER_CMD+=(--model     "$MODEL")
[[ -n "$TOKENIZER" ]] && SERVER_CMD+=(--tokenizer "$TOKENIZER")

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------

mkdir -p "$ROOT/logs"

echo ""
echo "════════════════════════════════════════"
echo "  Regent Model Studio"
echo "════════════════════════════════════════"
echo "  Config    : $CONFIG"
echo "  Weights   : ${MODEL:-none (random init)}"
echo "  Tokenizer : ${TOKENIZER:-none}"
echo "  API port  : $PORT"
echo "  UI port   : $UI_PORT"
echo "════════════════════════════════════════"
echo ""

echo "── Starting model server…"
PYTHONPATH="$ROOT" "${SERVER_CMD[@]}" \
  > "$ROOT/logs/server.log" 2>&1 &
SERVER_PID=$!
echo "   server pid: $SERVER_PID  (logs: logs/server.log)"

# Give the server a moment to bind before starting the UI
sleep 1

# ---------------------------------------------------------------------------
# Start UI
# ---------------------------------------------------------------------------

echo "── Starting Nuxt UI…"
(cd "$ROOT/ui" && API_URL="http://localhost:$PORT" PORT=$UI_PORT npm run dev) \
  > "$ROOT/logs/ui.log" 2>&1 &
UI_PID=$!
echo "   UI pid: $UI_PID  (logs: logs/ui.log)"

echo ""
echo "── Ready"
echo "   API  → http://localhost:$PORT/health"
echo "   UI   → http://localhost:$UI_PORT"
echo ""
echo "   Press Ctrl+C to stop both processes."
echo ""

# ---------------------------------------------------------------------------
# Tail both logs to stdout so the terminal is useful
# ---------------------------------------------------------------------------

# Re-open logs now that processes are running
tail -f "$ROOT/logs/server.log" "$ROOT/logs/ui.log" &
TAIL_PID=$!

# Wait for either child to exit unexpectedly
wait $SERVER_PID || true
wait $UI_PID     || true

kill $TAIL_PID 2>/dev/null || true
cleanup
