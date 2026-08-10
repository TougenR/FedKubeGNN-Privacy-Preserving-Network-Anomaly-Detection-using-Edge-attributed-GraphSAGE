#!/usr/bin/env bash
set -euo pipefail

readonly project_id="fedlearning-20260729-hn"
readonly zone="asia-southeast1-b"
readonly jenkins_instance="fedkube-jenkins"
readonly traffic_instance="fedkube-traffic-generator"

usage() {
  echo "Usage: $0 {status|demo|ci}" >&2
  exit 2
}

instance_status() {
  gcloud compute instances describe "$1" \
    --project="$project_id" \
    --zone="$zone" \
    --format='value(status)' 2>/dev/null || true
}

require_instance() {
  if [[ -z "$(instance_status "$1")" ]]; then
    echo "Required instance does not exist: $1" >&2
    exit 1
  fi
}

wait_for_status() {
  local instance="$1"
  local expected="$2"
  gcloud compute instances describe "$instance" \
    --project="$project_id" \
    --zone="$zone" \
    --format='value(status)' >/dev/null
  for _ in {1..60}; do
    if [[ "$(instance_status "$instance")" == "$expected" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "Timed out waiting for $instance to become $expected." >&2
  exit 1
}

stop_instance() {
  local instance="$1"
  if [[ "$(instance_status "$instance")" != "TERMINATED" ]]; then
    gcloud compute instances stop "$instance" \
      --project="$project_id" \
      --zone="$zone" \
      --quiet
    wait_for_status "$instance" TERMINATED
  fi
}

start_instance() {
  local instance="$1"
  if [[ "$(instance_status "$instance")" != "RUNNING" ]]; then
    gcloud compute instances start "$instance" \
      --project="$project_id" \
      --zone="$zone" \
      --quiet
    wait_for_status "$instance" RUNNING
  fi
}

show_status() {
  gcloud compute instances list \
    --project="$project_id" \
    --filter="name=($jenkins_instance OR $traffic_instance)" \
    --format='table(name,status,zone.basename(),machineType.basename(),networkInterfaces[0].networkIP)'
}

mode="${1:-}"
case "$mode" in
  status)
    show_status
    ;;
  demo)
    require_instance "$jenkins_instance"
    require_instance "$traffic_instance"
    stop_instance "$jenkins_instance"
    start_instance "$traffic_instance"
    if [[ "$(instance_status "$jenkins_instance")" != "TERMINATED" ]]; then
      echo "Fail closed: Jenkins is not stopped." >&2
      exit 1
    fi
    show_status
    ;;
  ci)
    require_instance "$jenkins_instance"
    require_instance "$traffic_instance"
    stop_instance "$traffic_instance"
    start_instance "$jenkins_instance"
    if [[ "$(instance_status "$traffic_instance")" != "TERMINATED" ]]; then
      echo "Fail closed: traffic generator is not stopped." >&2
      exit 1
    fi
    show_status
    ;;
  *)
    usage
    ;;
esac
