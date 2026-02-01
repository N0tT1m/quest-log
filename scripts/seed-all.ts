#!/usr/bin/env npx tsx
/**
 * Run all seed scripts to populate the database
 * Usage: npx tsx scripts/seed-all.ts
 */

import { execSync } from 'child_process';
import { readdirSync } from 'fs';
import { join } from 'path';

const scriptsDir = join(process.cwd(), 'scripts');

// Get all seed files except this one and seed.ts (base schema)
const seedFiles = readdirSync(scriptsDir)
	.filter((f) => f.startsWith('seed-') && f.endsWith('.ts') && f !== 'seed-all.ts' && f !== 'seed.ts')
	.sort();

console.log('=== Running All Seed Scripts ===\n');
console.log('Found seed files:');
seedFiles.forEach((f) => console.log(`  - ${f}`));
console.log();

// Check if base seed needs to run first
console.log('[1/2] Running base seed (schema + initial data)...');
try {
	execSync('npx tsx scripts/seed.ts', { stdio: 'inherit' });
	console.log('✓ Base seed complete\n');
} catch (e) {
	console.error('✗ Base seed failed');
	process.exit(1);
}

// Run all other seeds
console.log('[2/2] Running additional seeds...\n');

let success = 0;
let failed = 0;

for (const file of seedFiles) {
	console.log(`Running ${file}...`);
	try {
		execSync(`npx tsx scripts/${file}`, { stdio: 'pipe' });
		console.log(`  ✓ ${file}`);
		success++;
	} catch (e) {
		console.log(`  ✗ ${file} (may have duplicate data, continuing...)`);
		failed++;
	}
}

console.log('\n=== Seeding Complete ===');
console.log(`Success: ${success}`);
console.log(`Skipped/Failed: ${failed}`);
console.log('\nYour database should now have all learning paths!');
