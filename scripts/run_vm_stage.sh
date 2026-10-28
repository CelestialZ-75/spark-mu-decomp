#!/usr/bin/env bash
# Run from Ubuntu; each stage stops on error and preserves its terminal log.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-}"
case "$stage" in
  smoke|cpu|spark|rms-features|mu-features|rms-ridge|mu-gbt|mu-ridge) ;;
  *) echo 'Usage: bash scripts/run_vm_stage.sh {smoke|cpu|spark|rms-features|mu-features|rms-ridge|mu-gbt|mu-ridge}' >&2; exit 2 ;;
esac
mkdir -p outputs/docker-logs
exec > >(tee "outputs/docker-logs/${stage}.log") 2>&1
dc=(docker compose -f environment/compose.yaml -f environment/compose.limits.yaml)
"${dc[@]}" config --quiet
"${dc[@]}" config > outputs/docker-logs/compose.resolved.yaml
submit=(spark-submit --master spark://master:7077 --deploy-mode client
  --driver-memory 768m --executor-memory 512m
  --conf spark.driver.host=driver --conf spark.driver.bindAddress=0.0.0.0
  --conf spark.cores.max=2 --conf spark.executor.cores=1
  --conf spark.scheduler.minRegisteredResourcesRatio=1.0
  --conf spark.scheduler.maxRegisteredResourcesWaitingTime=30s
  --conf spark.sql.shuffle.partitions=2)
case "$stage" in
  cpu|rms-features|mu-features)
    # CPU baseline owns the same total budget; avoid competing cluster daemons.
    "${dc[@]}" stop worker-1 worker-2 master
    if [[ "$stage" == cpu ]]; then
      "${dc[@]}" run --rm --no-deps cpu
      "${dc[@]}" run --rm --no-deps cpu python -m mu_course.plot_results /workspace/outputs/docker-cpu
    else
      kind="${stage%-features}"
      "${dc[@]}" run --rm --no-deps cpu python -m mu_course.prepare_prediction \
        --features "$kind" --output "/workspace/outputs/prediction-${kind}"
    fi
    ;;
  *)
    "${dc[@]}" up -d master worker-1 worker-2
    ids=()
    while IFS= read -r id; do ids+=("$id"); done < <("${dc[@]}" ps -q master worker-1 worker-2)
    docker inspect --format '{{.Name}} cpu={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}}' \
      "${ids[@]}" | tee "outputs/docker-logs/${stage}-limits.txt"
    case "$stage" in
      smoke)
        "${dc[@]}" run --rm --use-aliases driver "${submit[@]}" scripts/environment_smoke.py \
          --backend spark --master spark://master:7077 --require-two-hosts
        ;;
      spark)
        "${dc[@]}" run --rm --use-aliases driver "${submit[@]}" scripts/submit.py \
          --backend spark --master spark://master:7077 --partitions 2 --threads 1 \
          --output /workspace/outputs/docker-spark
        "${dc[@]}" run --rm --no-deps driver python -m mu_course.plot_results /workspace/outputs/docker-spark
        python3 - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('outputs/docker-spark/parameters.json').read_text())
assert len(report['worker_hosts']) == 2, report['worker_hosts']
assert report['mu_count'] > 0, 'No MU found'
print('Two workers participated:', report['worker_hosts'])
PY
        ;;
      *)
        kind="${stage%-*}"
        model="${stage#*-}"
        "${dc[@]}" run --rm --use-aliases driver "${submit[@]}" scripts/predict_submit.py \
          --features "/workspace/outputs/prediction-${kind}" --model "$model" \
          --master spark://master:7077 --partitions 2 --require-two-hosts \
          --output "/workspace/outputs/${stage}"
        ;;
    esac
    ;;
esac
echo "Completed stage: $stage"
