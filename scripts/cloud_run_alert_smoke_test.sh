#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/cloud_run_alert_smoke_test.sh --marker <roc_auc_low|drift_detected|job_failure> [options]

Options:
  --project <id>        GCP project ID (default: active gcloud project)
  --region <region>     Cloud Run region (default: us-central1)
  --repo <name>         Artifact Registry repo (default: foresight)
  --job-name <name>     Cloud Run Job name (default: alert-log-smoke-<epoch>)
  --roc-auc <float>     Value for roc_auc_low marker (default: 0.8425)
  --count <int>         Value for drift_detected marker (default: 3)
  --keep-job            Do not delete the Cloud Run job after execution
  --no-execute          Deploy only; do not execute the job
  -h, --help            Show this help

Examples:
  scripts/cloud_run_alert_smoke_test.sh --marker roc_auc_low
  scripts/cloud_run_alert_smoke_test.sh --marker drift_detected --count 6
  scripts/cloud_run_alert_smoke_test.sh --marker job_failure --keep-job
EOF
}

MARKER=""
PROJECT_ID=""
REGION="us-central1"
REPO_NAME="foresight"
JOB_NAME="alert-log-smoke-$(date +%s)"
ROC_AUC="0.8425"
DRIFT_COUNT="3"
KEEP_JOB="false"
EXECUTE_JOB="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --marker)
      MARKER="$2"
      shift 2
      ;;
    --project)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --repo)
      REPO_NAME="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --roc-auc)
      ROC_AUC="$2"
      shift 2
      ;;
    --count)
      DRIFT_COUNT="$2"
      shift 2
      ;;
    --keep-job)
      KEEP_JOB="true"
      shift
      ;;
    --no-execute)
      EXECUTE_JOB="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$MARKER" ]]; then
  echo "--marker is required" >&2
  usage
  exit 1
fi

case "$MARKER" in
  roc_auc_low|drift_detected|job_failure)
    ;;
  *)
    echo "Invalid --marker: $MARKER" >&2
    usage
    exit 1
    ;;
esac

if [[ -z "$PROJECT_ID" ]]; then
  PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
fi

if [[ -z "$PROJECT_ID" ]]; then
  echo "No GCP project configured. Pass --project or run: gcloud config set project <id>" >&2
  exit 1
fi

TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/alert-log-smoke:${TAG}"

TMP_DIR="$(mktemp -d)"
cleanup_tmp() {
  rm -rf "$TMP_DIR"
}
trap cleanup_tmp EXIT

cat > "${TMP_DIR}/runner.py" <<'PY'
import argparse
import logging
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True, choices=["roc_auc_low", "drift_detected", "job_failure"])
    parser.add_argument("--roc-auc", type=float, default=0.8425)
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("alert-smoke")

    logger.info("Starting Cloud Run alert smoke test marker=%s", args.marker)

    if args.marker == "roc_auc_low":
        logger.warning("TEST_ROC_AUC_LOW test_roc_auc=%.4f threshold=0.85", args.roc_auc)
        logger.info("Finished successfully")
        return 0

    if args.marker == "drift_detected":
        logger.warning("DRIFT_DETECTED count=%d", args.count)
        logger.info("Finished successfully")
        return 0

    logger.error("CLOUD_RUN_JOB_FAILURE_TEST simulated error for alert validation")
    return 1


if __name__ == "__main__":
    sys.exit(main())
PY

cat > "${TMP_DIR}/Dockerfile" <<'DOCKER'
FROM python:3.12-slim
WORKDIR /app
COPY runner.py /app/runner.py
ENTRYPOINT ["python", "/app/runner.py"]
DOCKER

echo "[1/4] Building throwaway smoke image: ${IMAGE_URI}"
gcloud builds submit "$TMP_DIR" --tag "$IMAGE_URI" --project "$PROJECT_ID" --quiet

echo "[2/4] Deploying Cloud Run job: ${JOB_NAME}"
gcloud run jobs deploy "$JOB_NAME" \
  --image "$IMAGE_URI" \
  --region "$REGION" \
  --project "$PROJECT_ID" \
  --max-retries 0 \
  --args="--marker=${MARKER},--roc-auc=${ROC_AUC},--count=${DRIFT_COUNT}" \
  --quiet

if [[ "$EXECUTE_JOB" == "true" ]]; then
  echo "[3/4] Executing Cloud Run job (wait=true)"
  set +e
  gcloud run jobs execute "$JOB_NAME" --region "$REGION" --project "$PROJECT_ID" --wait --quiet
  EXEC_RC=$?
  set -e

  if [[ "$MARKER" == "job_failure" ]]; then
    if [[ "$EXEC_RC" -eq 0 ]]; then
      echo "Expected a failed execution for marker=job_failure, but execution succeeded." >&2
      exit 1
    fi
    echo "Job failure marker test behaved as expected (non-zero execution)."
  else
    if [[ "$EXEC_RC" -ne 0 ]]; then
      echo "Execution failed unexpectedly for marker=${MARKER}" >&2
      exit "$EXEC_RC"
    fi
    echo "Execution succeeded for marker=${MARKER}."
  fi

  case "$MARKER" in
    roc_auc_low)
      echo "Check alert policy: Model Test ROC-AUC Below 0.85"
      ;;
    drift_detected)
      echo "Check alert policy: Drift Detected"
      ;;
    job_failure)
      echo "Check alert policy: Cloud Run Job Failure"
      ;;
  esac
fi

if [[ "$KEEP_JOB" == "false" ]]; then
  echo "[4/4] Deleting temporary Cloud Run job: ${JOB_NAME}"
  gcloud run jobs delete "$JOB_NAME" --region "$REGION" --project "$PROJECT_ID" --quiet
else
  echo "[4/4] Keeping Cloud Run job: ${JOB_NAME}"
fi

echo "Done. Image pushed: ${IMAGE_URI}"
