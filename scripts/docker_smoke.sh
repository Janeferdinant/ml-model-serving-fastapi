#!/bin/sh
# Run the image locally (never pushes it anywhere) and smoke-test the running container.
# Usage: sh scripts/docker_smoke.sh <image[:tag]>
set -eu

IMAGE="${1:?usage: docker_smoke.sh <image>}"
NAME="mlserving-smoke-$$"
PORT="${SMOKE_PORT:-18000}"
KEY="smoke-test-key"

cleanup() {
  docker logs --tail 30 "$NAME" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "image size (bytes): $(docker image inspect "$IMAGE" --format '{{.Size}}')"

docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:8000" \
  -e MLS_API_KEY="$KEY" \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges:true \
  --health-interval 5s --health-start-period 5s \
  "$IMAGE" >/dev/null

python3 scripts/smoke_test.py --base-url "http://127.0.0.1:${PORT}" --api-key "$KEY" --wait 90

uid="$(docker exec "$NAME" id -u)"
echo "container user id: $uid"
[ "$uid" != "0" ] || { echo "FAIL: container runs as root"; exit 1; }

i=0
while [ "$i" -lt 30 ]; do
  status="$(docker inspect --format '{{.State.Health.Status}}' "$NAME")"
  [ "$status" = "healthy" ] && break
  i=$((i + 1))
  sleep 2
done
echo "docker health status: $status"
[ "$status" = "healthy" ] || { echo "FAIL: container did not become healthy"; exit 1; }

echo "DOCKER SMOKE OK"
