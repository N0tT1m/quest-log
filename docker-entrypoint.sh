#!/bin/sh
set -e

echo "Initializing database..."

# Run seed scripts which handle migrations via ALTER TABLE
npx tsx scripts/seed-projects.ts
npx tsx scripts/seed-ai-paths.ts

echo "Starting application..."
exec node build
