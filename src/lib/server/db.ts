import Database from 'better-sqlite3';
import { drizzle } from 'drizzle-orm/better-sqlite3';
import * as schema from './schema';
import { existsSync, mkdirSync } from 'fs';
import { dirname } from 'path';

const dbPath = process.env.DATABASE_URL?.replace('file:', '') || './data/quest-log.db';

// Ensure data directory exists
const dir = dirname(dbPath);
if (!existsSync(dir)) {
	mkdirSync(dir, { recursive: true });
}

const sqlite = new Database(dbPath);
sqlite.pragma('journal_mode = WAL');
sqlite.pragma('foreign_keys = ON');

export const db = drizzle(sqlite, { schema });

// Initialize tables
sqlite.exec(`
	CREATE TABLE IF NOT EXISTS paths (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL,
		description TEXT,
		icon TEXT DEFAULT 'book',
		color TEXT DEFAULT 'emerald',
		language TEXT,
		skills TEXT,
		start_hint TEXT,
		difficulty TEXT DEFAULT 'intermediate',
		estimated_weeks INTEGER,
		created_at INTEGER
	);

	CREATE TABLE IF NOT EXISTS modules (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		path_id INTEGER NOT NULL REFERENCES paths(id) ON DELETE CASCADE,
		name TEXT NOT NULL,
		description TEXT,
		order_index INTEGER NOT NULL DEFAULT 0,
		target_date TEXT,
		created_at INTEGER
	);

	CREATE TABLE IF NOT EXISTS tasks (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		module_id INTEGER NOT NULL REFERENCES modules(id) ON DELETE CASCADE,
		title TEXT NOT NULL,
		description TEXT,
		order_index INTEGER NOT NULL DEFAULT 0,
		completed INTEGER NOT NULL DEFAULT 0,
		completed_at INTEGER,
		notes TEXT,
		created_at INTEGER
	);

	CREATE TABLE IF NOT EXISTS activity (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		date TEXT NOT NULL UNIQUE,
		tasks_completed INTEGER NOT NULL DEFAULT 0
	);
`);

// Add new columns if they don't exist (for existing databases)
try {
	sqlite.exec(`ALTER TABLE paths ADD COLUMN language TEXT`);
} catch {}
try {
	sqlite.exec(`ALTER TABLE paths ADD COLUMN skills TEXT`);
} catch {}
try {
	sqlite.exec(`ALTER TABLE paths ADD COLUMN start_hint TEXT`);
} catch {}
try {
	sqlite.exec(`ALTER TABLE paths ADD COLUMN difficulty TEXT DEFAULT 'intermediate'`);
} catch {}
try {
	sqlite.exec(`ALTER TABLE paths ADD COLUMN estimated_weeks INTEGER`);
} catch {}

export { schema };
