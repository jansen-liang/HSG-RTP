#!/bin/bash
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_DIR="${HSG_RTP_OUTPUT_DIR:-${HLR_OUTPUT_DIR:-$PROJECT_ROOT/pipeline/output}}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/hsg_rtp_dataset_${TIMESTAMP}.json"

cd "$PROJECT_ROOT"

# hotel supermarket allensville

python -m pipeline.main \
    --scenes  hotel supermarket allensville office pudu \
    --task-types delivery tidying guidance \
    --difficulties easy medium hard \
    --max-tasks 70 \
    --easy-prop 0.5 \
    --medium-prop 0.3 \
    --hard-prop 0.2 \
    --llm-agent gpt-4o-mini \
    --output "$OUTPUT_FILE"

echo "✅ Dataset saved to: $OUTPUT_FILE"
