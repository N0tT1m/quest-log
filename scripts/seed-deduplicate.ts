import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

// Find duplicate path names
const duplicates = db.prepare(`
  SELECT name, COUNT(*) as cnt FROM paths
  GROUP BY name
  HAVING COUNT(*) > 1
`).all() as { name: string; cnt: number }[];

console.log(`Found ${duplicates.length} duplicate path names`);

duplicates.forEach((dup) => {
	// Get all paths with this name, ordered by task count descending
	const paths = db.prepare(`
    SELECT p.id, p.name,
      (SELECT COUNT(*) FROM modules m JOIN tasks t ON t.module_id = m.id WHERE m.path_id = p.id) as task_count,
      LENGTH(p.schedule) as schedule_len
    FROM paths p
    WHERE p.name = ?
    ORDER BY task_count DESC, schedule_len DESC
  `).all(dup.name) as { id: number; name: string; task_count: number; schedule_len: number }[];

	console.log(`\n${dup.name}:`);
	paths.forEach((p, i) => {
		console.log(`  ID ${p.id}: ${p.task_count} tasks, schedule: ${p.schedule_len || 0} chars ${i === 0 ? '← KEEP' : '← DELETE'}`);
	});

	// Delete all but the one with the most tasks
	const toDelete = paths.slice(1).map((p) => p.id);
	if (toDelete.length > 0) {
		toDelete.forEach((id) => {
			db.prepare('DELETE FROM tasks WHERE module_id IN (SELECT id FROM modules WHERE path_id = ?)').run(id);
			db.prepare('DELETE FROM modules WHERE path_id = ?').run(id);
			db.prepare('DELETE FROM paths WHERE id = ?').run(id);
		});
		console.log(`  Deleted IDs: ${toDelete.join(', ')}`);
	}
});

const finalCount = db.prepare('SELECT COUNT(*) as count FROM paths').get() as { count: number };
console.log(`\nFinal path count: ${finalCount.count}`);

db.close();
