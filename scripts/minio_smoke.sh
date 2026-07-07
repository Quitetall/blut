#!/usr/bin/env bash
# MinIO smoke for the S3 blob store (ADR 0067 T4.2).
#
# Stands up a throwaway MinIO (via docker), points the AWS_* env at it, creates
# a bucket, and runs the `#[ignore]`d `s3_blob_store_round_trip` test against it.
# CPU-only, no cloud account — CI-able on any runner with docker.
#
#   ./scripts/minio_smoke.sh
#
# Exit non-zero if the round-trip fails. Cleans up the container on exit.
set -euo pipefail

CONTAINER="blut-minio-smoke-$$"
PORT="${MINIO_PORT:-19000}"
BUCKET="blut-smoke"
ACCESS="minioadmin"
SECRET="minioadmin"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo ">> starting MinIO ($CONTAINER) on :$PORT"
docker run -d --name "$CONTAINER" -p "$PORT:9000" \
  -e "MINIO_ROOT_USER=$ACCESS" -e "MINIO_ROOT_PASSWORD=$SECRET" \
  quay.io/minio/minio server /data >/dev/null

# Wait for the endpoint to answer — fail loudly if it never does, rather than
# falling through to a confusing connection-refused in `cargo test`.
ready=0
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/minio/health/live" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo ">> MinIO failed to become healthy after 30s" >&2
  docker logs "$CONTAINER" 2>&1 | tail -20 >&2 || true
  exit 1
fi

export AWS_ACCESS_KEY_ID="$ACCESS"
export AWS_SECRET_ACCESS_KEY="$SECRET"
export AWS_ENDPOINT="http://127.0.0.1:$PORT"
export AWS_ALLOW_HTTP=true
export AWS_REGION="us-east-1"
export BLUT_S3_TEST_BUCKET="$BUCKET"

# Create the bucket via the AWS CLI if present, else via a mc sidecar.
if command -v aws >/dev/null 2>&1; then
  aws --endpoint-url "$AWS_ENDPOINT" s3 mb "s3://$BUCKET" >/dev/null 2>&1 || true
else
  docker run --rm --network host --entrypoint sh quay.io/minio/mc -c \
    "mc alias set s $AWS_ENDPOINT $ACCESS $SECRET && mc mb -p s/$BUCKET" >/dev/null 2>&1 || true
fi

echo ">> running s3_blob_store_round_trip"
cargo test --features s3 --lib object_store::tests::s3_blob_store_round_trip \
  -- --ignored --exact --nocapture

echo ">> MinIO smoke PASSED"
