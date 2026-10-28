#!/usr/bin/env bash
# Run from the project root: bash scripts/run_formal_group.sh outputs/formal-... mu cpu
set -euo pipefail
root=$(cd "${1:?Specify formal batch directory}" && pwd)
shift
cd "$(dirname "$0")/.."
tag=$(basename "$root")
kind=${1:?Specify mu or ml}
profile=${2:?Specify profile}
pair=${3:-}
base=(docker compose -f environment/compose.yaml -f environment/compose.limits.yaml)
override="$root/config/default.yaml"
workers=(worker-1 worker-2)
slots=2
parts=2
case "$kind:$profile" in
  mu:cpu|ml:local1) workers=() ;;
  mu:one-half) workers=(worker-1); slots=1 ;;
  mu:one-full) workers=(worker-1); override="$root/config/one-full.yaml" ;;
  mu:two-p1) parts=1 ;;
  mu:two-p2|ml:cluster2) ;;
  mu:two-p4) parts=4 ;;
  *) echo 'Unknown experiment profile' >&2; exit 2 ;;
esac
if [[ "$kind" == ml ]]; then
  case "$pair" in rms-ridge|mu-gbt|mu-ridge) ;; *) echo 'Unknown feature/model pair' >&2; exit 2 ;; esac
fi
test "$(docker image inspect spark-mu-course:dev --format '{{.Id}}')" = "$(cat "$root/image-id.txt")"
sha256sum --check "$root/input-sha256.txt"
group="$kind-$profile${pair:+-$pair}"
test ! -e "$root/$group" || { echo "Group already exists: $group; preserve it and use a new batch" >&2; exit 1; }
mkdir "$root/$group"
dc=("${base[@]}" -f "$override")
"${dc[@]}" config > "$root/$group/compose.resolved.yaml"
"${base[@]}" stop worker-1 worker-2 master
if ((${#workers[@]})); then
  "${dc[@]}" up -d master "${workers[@]}"
  ids=()
  while IFS= read -r id; do ids+=("$id"); done < <("${dc[@]}" ps -q master "${workers[@]}")
  docker inspect --format '{{.Name}} cpu={{.HostConfig.NanoCpus}} memory={{.HostConfig.Memory}} swap={{.HostConfig.MemorySwap}}' "${ids[@]}" > "$root/$group/limits.txt"
fi
submit=(spark-submit --master spark://master:7077 --deploy-mode client
  --driver-memory 768m --executor-memory 512m
  --conf spark.driver.host=driver --conf spark.driver.bindAddress=0.0.0.0
  --conf "spark.cores.max=$slots" --conf spark.executor.cores=1
  --conf spark.scheduler.minRegisteredResourcesRatio=1.0
  --conf spark.scheduler.maxRegisteredResourcesWaitingTime=30s
  --conf spark.sql.shuffle.partitions=2)
for run in 0 1 2 3; do
  dest="/workspace/outputs/$tag/$group/run-$run"
  if [[ "$kind" == mu ]]; then
    args=(--start 3 --duration 2 --ext-factor 10 --n-mu-max 5 --n-iter 30 --cost-fcn 1 --sil-threshold 0.8 --seed 42 --output "$dest")
    if [[ "$profile" == cpu ]]; then
      cmd=("${dc[@]}" run --rm --no-deps cpu python -m mu_course --backend cpu --threads 2 "${args[@]}")
    else
      cmd=("${dc[@]}" run --rm --no-deps --use-aliases driver "${submit[@]}" scripts/submit.py --backend spark --master spark://master:7077 --threads 1 --partitions "$parts" "${args[@]}")
    fi
  else
    feature=${pair%-*}
    model=${pair#*-}
    args=(--features "/workspace/outputs/$tag/features-$feature" --model "$model" --partitions 2 --seed 42 --reg-param 0.1 --max-iter 20 --max-depth 3 --output "$dest")
    if [[ "$profile" == local1 ]]; then
      cmd=("${dc[@]}" run --rm --no-deps cpu spark-submit --master 'local[1]' --driver-memory 768m --conf spark.sql.shuffle.partitions=2 scripts/predict_submit.py --master 'local[1]' "${args[@]}")
    else
      cmd=("${dc[@]}" run --rm --no-deps --use-aliases driver "${submit[@]}" scripts/predict_submit.py --master spark://master:7077 --require-two-hosts "${args[@]}")
    fi
  fi
  echo "Running $group / run-$run (0=warm-up)"
  start=$(date +%s)
  "${cmd[@]}" 2>&1 | tee "$root/$group/run-$run.log"
  printf '%s\n' "$(( $(date +%s) - start ))" > "$root/$group/run-$run.wall-seconds.txt"
done
touch "$root/$group/SUCCESS"
echo "Completed group: $group"
