#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  paper/trace_run.sh --intent TEXT --setup TEXT (--expect-artifact PATH | --expect-log PATH | --config PATH) [...] -- COMMAND [ARG ...]

Environment:
  EVENTPDF_USE_TRACE=0       Local-debug opt-out; do not use for paper experiments.
  EVENTPDF_TRACE_REQUIRED=0  Local-debug fallback; do not use for paper experiments.
  EVENTPDF_TRACE_CLI="..."   Optional command prefix, e.g. "uvx --from origami-trace otrace".
  EVENTPDF_ENV_FILE=PATH     Optional dotenv file path; defaults to repo-root .env.
  EVENTPDF_TRACE_ALLOW_BINARY_ARTIFACTS=1  Explicitly allow binary expected artifacts.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${EVENTPDF_ENV_FILE:-$repo_root/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$env_file"
  set +a
fi

intent=""
setup=""
declare -a expected_args=()
declare -a binary_artifacts=()
evidence_arg_count=0

while (($#)); do
  case "$1" in
    --intent)
      shift
      [[ $# -gt 0 ]] || { usage; exit 2; }
      intent="$1"
      ;;
    --setup)
      shift
      [[ $# -gt 0 ]] || { usage; exit 2; }
      setup="$1"
      ;;
    --expect-artifact|--expect-log|--config|--tag|--title)
      key="$1"
      shift
      [[ $# -gt 0 ]] || { usage; exit 2; }
      expected_args+=("$key" "$1")
      case "$key" in
        --expect-artifact|--expect-log|--config)
          evidence_arg_count=$((evidence_arg_count + 1))
          ;;
      esac
      if [[ "$key" == "--expect-artifact" ]]; then
        artifact_lower="${1,,}"
        case "$artifact_lower" in
          *.pdf|*.png|*.jpg|*.jpeg|*.gif|*.webp|*.bmp|*.tif|*.tiff)
            binary_artifacts+=("$1")
            ;;
        esac
      fi
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown trace_run.sh argument: $1" >&2
      usage
      exit 2
      ;;
  esac
  shift
done

[[ -n "$intent" && -n "$setup" && $# -gt 0 ]] || { usage; exit 2; }

trace_fail_or_fallback() {
  local reason="$1"
  shift
  if [[ "${EVENTPDF_TRACE_REQUIRED:-1}" == "1" ]]; then
    echo "Trace required but unavailable: ${reason}" >&2
    exit 1
  fi
  echo "Trace disabled: ${reason}; running command directly." >&2
  exec "$@"
}

if [[ "${EVENTPDF_USE_TRACE:-1}" != "1" ]]; then
  exec "$@"
fi

if ((evidence_arg_count == 0)); then
  echo "Trace requires at least one --expect-artifact, --expect-log, or --config path." >&2
  exit 2
fi

if ((${#binary_artifacts[@]} > 0)) && [[ "${EVENTPDF_TRACE_ALLOW_BINARY_ARTIFACTS:-0}" != "1" ]]; then
  echo "Trace binary expected artifacts are disabled by default after hosted upload warnings." >&2
  printf 'Use source/text artifacts instead, or set EVENTPDF_TRACE_ALLOW_BINARY_ARTIFACTS=1 for: %s\n' "${binary_artifacts[*]}" >&2
  exit 2
fi

declare -a trace_cmd=()
if [[ -n "${EVENTPDF_TRACE_CLI:-}" ]]; then
  # shellcheck disable=SC2206
  trace_cmd=(${EVENTPDF_TRACE_CLI})
elif command -v uvx >/dev/null 2>&1; then
  trace_cmd=(uvx --from origami-trace otrace)
elif command -v otrace >/dev/null 2>&1; then
  trace_cmd=(otrace)
else
  trace_cmd=(uvx --from origami-trace otrace)
fi

trace_help_ok() {
  local help_text required
  help_text="$("${trace_cmd[@]}" --help 2>&1 || true)"
  for required in init whoami run; do
    [[ "$help_text" == *"$required"* ]] || return 1
  done
  return 0
}

if ! trace_help_ok; then
  trace_cmd=(uvx --from origami-trace otrace)
  trace_help_ok || trace_fail_or_fallback "hosted Trace CLI is unavailable" "$@"
fi

if [[ -z "${TRACE_API_KEY:-}" ]]; then
  trace_fail_or_fallback "TRACE_API_KEY is unset" "$@"
fi

exec "${trace_cmd[@]}" run \
  --intent "$intent" \
  --setup "$setup" \
  "${expected_args[@]}" \
  -- "$@"
