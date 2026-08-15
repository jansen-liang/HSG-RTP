#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NO_HSG_SUMMARY="${HSG_RTP_NO_HSG_SUMMARY:-evaluation_results/no_hsg_trained_bounded_final70_20260809/summary.json}"
while [[ ! -f "$NO_HSG_SUMMARY" ]]; do
    if ! tmux has-session -t no_hsg_train 2>/dev/null \
        && ! tmux has-session -t no_hsg_eval_wait 2>/dev/null; then
        echo "No-HSG training/evaluation exited before producing $NO_HSG_SUMMARY" >&2
        exit 1
    fi
    sleep 60
done

FULL_SUMMARY="${HSG_RTP_FULL_SUMMARY:-evaluation_results/full_hsg_bounded_final70_20260809/summary.json}"
if [[ ! -f "$FULL_SUMMARY" ]]; then
    ./scripts/run_routed_ablation_evaluation.sh \
        checkpoints/full_protocol_matched/checkpoints.env \
        full \
        evaluation_results/full_hsg_bounded_final70_20260809
fi

for ablation in \
    no_global_topology \
    no_object_tokens \
    no_graph_updates_history
do
    ./scripts/run_component_ablation_training.sh "$ablation" smoke
    ./scripts/run_component_ablation_training.sh "$ablation" full
    ./scripts/run_routed_ablation_evaluation.sh \
        "checkpoints/${ablation}_protocol_matched/checkpoints.env" \
        "$ablation" \
        "evaluation_results/${ablation}_bounded_final70_20260809"
done

touch evaluation_results/component_ablations_complete_20260809
