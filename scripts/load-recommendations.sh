#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root regardless of where this script is invoked.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_REPO="$ROOT_DIR/.datasets/recommendations"
DUMP_RELATIVE="data/recommendations-embeddings-aligned-5.26.dump"
DUMP_FILE="$DATASET_REPO/$DUMP_RELATIVE"
COMPOSE_FILE="$ROOT_DIR/compose.db.yml"

compose() {
    docker compose -f "$COMPOSE_FILE" --profile migration "$@"
}

if [[ "${1:-}" != "--replace" ]]; then
    echo "Usage: $0 --replace"
    echo "WARNING: This replaces the existing Neo4j database used as the migration source."
    exit 1
fi

for command in docker git; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "Missing required command: $command" >&2
        exit 1
    fi
done

if ! git lfs version >/dev/null 2>&1; then
    echo "Git LFS is required. Install it and try again." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose is required." >&2
    exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "compose.db.yml not found in the repository root." >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "Missing .env. Copy .env.example to .env and set NEO4J_PASSWORD first." >&2
    exit 1
fi

echo "==> Downloading the Recommendations dataset"

mkdir -p "$ROOT_DIR/.datasets"

if [[ ! -d "$DATASET_REPO/.git" ]]; then
    if [[ -e "$DATASET_REPO" ]]; then
        echo "Dataset directory exists but is not a Git repository: $DATASET_REPO" >&2
        exit 1
    fi

    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
        https://github.com/neo4j-graph-examples/recommendations.git \
        "$DATASET_REPO"
fi

# Fetch only the Neo4j 5.26 dump needed by this project.
git -C "$DATASET_REPO" lfs pull \
    --include="$DUMP_RELATIVE" \
    --exclude=""

if [[ ! -s "$DUMP_FILE" ]]; then
    echo "Dataset dump not found: $DUMP_FILE" >&2
    exit 1
fi

# A failed LFS download leaves a small text pointer in place of the dump.
if head -n 1 "$DUMP_FILE" | grep -q \
    '^version https://git-lfs.github.com/spec/v1'; then
    echo "Git LFS did not download the actual dataset dump." >&2
    exit 1
fi

echo "==> Stopping Neo4j before the offline restore"

compose stop neo4j

echo "==> Restoring the dataset into this project's Neo4j volume"

# The one-off container mounts the same Compose-managed /data volume as the
# normal Neo4j service. Standard input carries the binary dump into neo4j-admin.
compose run \
    --rm \
    --no-deps \
    -T \
    --user neo4j \
    neo4j \
    neo4j-admin database load neo4j \
    --from-stdin \
    --overwrite-destination=true < "$DUMP_FILE"

echo "==> Starting Neo4j"

compose up -d neo4j

cypher() {
    compose exec -T neo4j sh -c '
        NEO4J_USERNAME=neo4j \
        NEO4J_PASSWORD="${NEO4J_AUTH#neo4j/}" \
        cypher-shell -d neo4j
    '
}

echo "==> Waiting for Neo4j"

ready=false

for ((attempt = 1; attempt <= 60; attempt++)); do
    if echo "RETURN 1;" | cypher >/dev/null 2>&1; then
        ready=true
        break
    fi

    sleep 2
done

if [[ "$ready" != true ]]; then
    echo "Neo4j did not become ready. Check the container logs." >&2
    exit 1
fi

echo "==> Verifying the imported source data"

cypher <<'CYPHER'
MATCH (movie:Movie)
RETURN
    count(movie) AS totalMovies,
    count(movie.plot) AS moviesWithPlots,
    count(movie.embedding) AS moviesWithSourceEmbeddings,
    count(movie.plotEmbeddingNemotron) AS moviesWithNemotronEmbeddings;
CYPHER

echo "==> Neo4j source dataset is ready"
echo "Next: start Memgraph and run the migration commands documented in README.md."
