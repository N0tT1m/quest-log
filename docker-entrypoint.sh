#\!/bin/sh
set -e

echo "Initializing database..."

# Run seed scripts which handle migrations via ALTER TABLE
npx tsx scripts/seed-projects.ts
npx tsx scripts/seed-ai-paths.ts
npx tsx scripts/seed-network-redteam.ts
npx tsx scripts/seed-redteam-learning.ts
npx tsx scripts/seed-hacking-expanded.ts
npx tsx scripts/seed-learn-dl-nocourses.ts
npx tsx scripts/seed-deep-learning-part2.ts
npx tsx scripts/seed-transformers-expanded.ts
npx tsx scripts/seed-ml-pipeline-expanded.ts

echo "Starting application..."
exec node build
