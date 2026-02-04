import Database from 'better-sqlite3';
import { getAllTaskDetails } from './task-details';

const db = new Database('data/quest-log.db');

const updateTask = db.prepare('UPDATE tasks SET details = ? WHERE id = ?');

const taskDetails = getAllTaskDetails();

console.log(`Updating ${taskDetails.size} tasks with details...`);

const transaction = db.transaction(() => {
	let updated = 0;
	for (const [id, details] of taskDetails) {
		const result = updateTask.run(details, id);
		if (result.changes === 0) {
			console.warn(`Task ${id} not found in database`);
		} else {
			updated++;
		}
	}
	return updated;
});

try {
	const updated = transaction();
	console.log(`Successfully updated ${updated} tasks`);

	// Verification
	const remaining = db
		.prepare(
			`
    SELECT COUNT(*) as count FROM tasks
    WHERE details IS NULL OR length(details) < 100
  `
		)
		.get() as { count: number };

	console.log(`Tasks still needing details: ${remaining.count}`);

	if (remaining.count === 0) {
		console.log('All tasks now have complete details!');
	}
} catch (error) {
	console.error('Update failed:', error);
	process.exit(1);
}
