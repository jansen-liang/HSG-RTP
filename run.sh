#!/bin/bash
set -e

# Backward-compatible entry point. New users should call run_stream.sh directly.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/run_stream.sh" "$@"
