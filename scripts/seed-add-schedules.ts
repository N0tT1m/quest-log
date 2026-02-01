import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const updateSchedule = db.prepare('UPDATE paths SET schedule = ? WHERE id = ?');

// Get paths without schedules and their modules/tasks
const paths = db.prepare(`
  SELECT p.id, p.name FROM paths p
  WHERE p.schedule IS NULL OR p.schedule = ''
`).all() as { id: number; name: string }[];

function getModulesWithTasks(pathId: number) {
	return db.prepare(`
    SELECT m.id, m.name, m.description,
      (SELECT GROUP_CONCAT(t.title, '|') FROM tasks t WHERE t.module_id = m.id ORDER BY t.order_index) as tasks
    FROM modules m WHERE m.path_id = ?
    ORDER BY m.order_index
  `).all(pathId) as { id: number; name: string; description: string; tasks: string }[];
}

function generateSchedule(name: string, modules: { id: number; name: string; description: string; tasks: string }[]) {
	const allTasks: { module: string; task: string }[] = [];

	modules.forEach((m) => {
		if (m.tasks) {
			m.tasks.split('|').forEach((t) => allTasks.push({ module: m.name, task: t }));
		}
	});

	if (allTasks.length === 0) {
		return 'Week 1:\n  - Get started with the learning path\n  - Review prerequisites\n  - Set up development environment';
	}

	const weeks = Math.max(2, Math.ceil(allTasks.length / 5)); // ~5 tasks per week
	const tasksPerWeek = Math.ceil(allTasks.length / weeks);

	let schedule = '';

	for (let w = 1; w <= weeks; w++) {
		schedule += `Week ${w}:\n`;
		const weekTasks = allTasks.slice((w - 1) * tasksPerWeek, w * tasksPerWeek);

		// Group by module
		const byModule: Record<string, string[]> = {};
		weekTasks.forEach((t) => {
			if (!byModule[t.module]) byModule[t.module] = [];
			byModule[t.module].push(t.task);
		});

		Object.entries(byModule).forEach(([mod, tasks]) => {
			schedule += `  ${mod}:\n`;
			tasks.forEach((t) => {
				schedule += `    - ${t}\n`;
			});
		});
		schedule += '\n';
	}

	return schedule.trim();
}

console.log(`Found ${paths.length} paths without schedules`);

paths.forEach((p) => {
	const modules = getModulesWithTasks(p.id);
	const schedule = generateSchedule(p.name, modules);
	updateSchedule.run(schedule, p.id);
	console.log(`  Added schedule to: ${p.name}`);
});

// Verify
const remaining = db
	.prepare("SELECT COUNT(*) as c FROM paths WHERE schedule IS NULL OR schedule = ''")
	.get() as { c: number };
console.log(`\nPaths still without schedules: ${remaining.c}`);

db.close();
