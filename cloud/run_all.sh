#!/usr/bin/env bash
# Run every vLLM-based experiment against the Modal servers from cloud/covert_modal.py.
# Usage: [COVERT=0] bash cloud/run_all.sh <qwen7b_url> <llama8b_url> <qwen32b_url>   (each ending in /v1)
# Three pipelines run in parallel, one per server. Responses are cached under cache/, so re-running is cheap.
set -euo pipefail
Q=$1 L=$2 B=$3
QWEN=Qwen/Qwen2.5-7B-Instruct LLAMA=NousResearch/Meta-Llama-3.1-8B-Instruct BIG=Qwen/Qwen2.5-32B-Instruct-AWQ
OUT=results/llm_rps_vllm
PY=.venv/Scripts/python
CONDS="independent channel channel_shared instructed"

games() {  # url model runs conditions...
  local url=$1 model=$2 runs=$3; shift 3
  for c in "$@"; do
    NETINF_BASE_URL=$url NETINF_MODEL=$model $PY experiments/llm_rps.py --condition "$c" --runs "$runs" \
      --rounds 100 --workers 16 --out $OUT
  done
}
judge() {  # url judge_model dir
  NETINF_BASE_URL=$1 NETINF_MODEL=$2 $PY experiments/judge_chat.py "$OUT/$3"
}

(  # Qwen-7B server: covert fine-tune, then fp16 rerun with 30 games
  if [ "${COVERT:-1}" = 1 ]; then games "$Q" covert 10 channel independent; judge "$Q" $QWEN covert; fi
  games "$Q" $QWEN 30 $CONDS
  judge "$Q" $QWEN Qwen2.5-7B-Instruct
) > $OUT.qwen.log 2>&1 &

(  # Llama-8B server: fp16 rerun, then mixed-model independent games (needs both servers)
  games "$L" $LLAMA 30 $CONDS
  judge "$L" $LLAMA Meta-Llama-3.1-8B-Instruct
  $PY experiments/llm_rps.py --condition independent --runs 30 --rounds 100 --workers 16 --out $OUT \
    --agents "$QWEN@$Q,$LLAMA@$L,$QWEN@$Q"
) > $OUT.llama.log 2>&1 &

(  # Qwen-32B server: does scale change emergence?
  games "$B" $BIG 20 $CONDS
  judge "$B" $BIG Qwen2.5-32B-Instruct-AWQ
) > $OUT.big.log 2>&1 &

wait
$PY experiments/analyze_llm.py $OUT
echo "all done"
