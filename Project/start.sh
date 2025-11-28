#!/bin/sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- CONFIGURATION ---
FRONTEND_PORT=8080
TUNNEL_LOG="/tmp/localhost-run.log"
# ---------------------

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "Neither 'docker compose' nor 'docker-compose' is available. Please install Docker Compose." >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "'ollama' CLI is required on the host. Install it from https://ollama.ai and try again." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "'curl' is required for health checks. Please install it and retry." >&2
  exit 1
fi

# --- OLLAMA SETUP ---
OLLAMA_LISTEN_ADDR="${OLLAMA_LISTEN_ADDR:-0.0.0.0:11434}"
OLLAMA_CLIENT_ADDR="${OLLAMA_CLIENT_ADDR:-127.0.0.1:11434}"
OLLAMA_HEALTH_URL="http://${OLLAMA_CLIENT_ADDR}/api/version"
OLLAMA_LOG="${OLLAMA_LOG:-/tmp/ollama-serve.log}"
OLLAMA_PID=""
TUNNEL_PID=""

cleanup() {
  echo ""
  echo "🛑 Shutting down services..."
  
  if [ -n "${TUNNEL_PID}" ]; then
    echo "Killing Tunnel (pid ${TUNNEL_PID})..."
    kill "${TUNNEL_PID}" 2>/dev/null || true
  fi

  if [ -n "${OLLAMA_PID}" ]; then
    echo "Stopping local Ollama (pid ${OLLAMA_PID})..."
    kill "${OLLAMA_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

ensure_ollama_ready() {
  i=0
  while [ $i -lt 30 ]; do
    if curl -sf "${OLLAMA_HEALTH_URL}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}

# --- 1. START OLLAMA ---
if ! curl -sf "${OLLAMA_HEALTH_URL}" >/dev/null 2>&1; then
  echo "🧠 Starting local Ollama server..."
  OLLAMA_HOST="${OLLAMA_LISTEN_ADDR}" nohup ollama serve >"${OLLAMA_LOG}" 2>&1 &
  OLLAMA_PID=$!
  if ! ensure_ollama_ready; then
    echo "Ollama did not become ready. Check logs at ${OLLAMA_LOG}." >&2
    exit 1
  fi
  echo "✅ Ollama is running (pid: ${OLLAMA_PID})."
else
  echo "✅ Detected existing Ollama server at ${OLLAMA_CLIENT_ADDR}."
fi


# --- 2. PRIME MODELS (First) ---
# 先熱機，這樣 Tunnel 開好時系統已經是 Ready 的
prime_model() {
  model="$1"
  echo "⚙️  Priming model ${model}..."
  if ! printf "%s" "Warmup request." | OLLAMA_HOST="${OLLAMA_CLIENT_ADDR}" ollama run "${model}" >/dev/null 2>&1; then
    echo "Failed to prime ${model}. Ensure the model name is correct." >&2
    exit 1
  fi
}

prime_model "hf.co/unsloth/medgemma-27b-it-GGUF:Q4_K_M"
prime_model "llama3.1:latest"


# --- 3. START TUNNEL (Then Tunnel) ---
start_tunnel() {
  echo "🚀 Starting localhost.run tunnel for port ${FRONTEND_PORT}..."
  rm -f "$TUNNEL_LOG"
  
  # 使用 -tt 強制輸出，並忽略 Host Key 檢查
  ssh -tt -R 80:localhost:${FRONTEND_PORT} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes localhost.run > "$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!

  echo "⏳ Waiting for public URL..."
  
  count=0
  while [ $count -lt 30 ]; do
    if [ -s "$TUNNEL_LOG" ]; then
        # 抓取 https 網址，並過濾掉無關的關鍵字
        URL=$(grep -o "https://[^ ]*" "$TUNNEL_LOG" | grep -v "admin" | grep -v "twitter" | grep -v "facebook" | grep -v "docs" | head -n 1 || true)
        
        if [ -n "$URL" ]; then
          # 清除可能存在的換行符號
          URL=$(echo "$URL" | tr -d '\r')
          echo ""
          echo "======================================================"
          echo "🌍 \033[1;32mYour Public URL: $URL\033[0m"
          echo "======================================================"
          echo "   (Docker logs will start in 5 seconds...)"
          echo ""
          return 0
        fi
    fi
    
    if ! kill -0 $TUNNEL_PID 2>/dev/null; then
        echo "❌ Tunnel process died. Log:"
        cat "$TUNNEL_LOG"
        return 1
    fi

    sleep 1
    count=$((count + 1))
    printf "."
  done
  
  echo ""
  echo "⚠️  Timeout. Could not grab URL. Raw log content:"
  cat "$TUNNEL_LOG"
}


start_tunnel

# --- 4. START DOCKER (Last) ---
echo "🐳 Starting Docker services..."
${COMPOSE_CMD} up --build "$@"