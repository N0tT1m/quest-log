import { db, schema } from './db';
import { eq, sql, desc, asc } from 'drizzle-orm';

const { paths, modules, tasks, activity } = schema;

// Paths
export async function getAllPaths() {
	return db.select().from(paths).orderBy(asc(paths.createdAt));
}

export async function getPathById(id: number) {
	const result = await db.select().from(paths).where(eq(paths.id, id)).limit(1);
	return result[0] || null;
}

export async function getPathWithModulesAndTasks(pathId: number) {
	const path = await getPathById(pathId);
	if (!path) return null;

	const pathModules = await db
		.select()
		.from(modules)
		.where(eq(modules.pathId, pathId))
		.orderBy(asc(modules.orderIndex));

	const modulesWithTasks = await Promise.all(
		pathModules.map(async (mod) => {
			const moduleTasks = await db
				.select()
				.from(tasks)
				.where(eq(tasks.moduleId, mod.id))
				.orderBy(asc(tasks.orderIndex));
			return { ...mod, tasks: moduleTasks };
		})
	);

	return { ...path, modules: modulesWithTasks };
}

// Tasks
export async function toggleTask(taskId: number) {
	const task = await db.select().from(tasks).where(eq(tasks.id, taskId)).limit(1);
	if (!task[0]) return null;

	const newCompleted = !task[0].completed;
	const now = new Date();
	const today = now.toISOString().split('T')[0];

	await db
		.update(tasks)
		.set({
			completed: newCompleted,
			completedAt: newCompleted ? now : null
		})
		.where(eq(tasks.id, taskId));

	// Update activity
	if (newCompleted) {
		await db
			.insert(activity)
			.values({ date: today, tasksCompleted: 1 })
			.onConflictDoUpdate({
				target: activity.date,
				set: { tasksCompleted: sql`${activity.tasksCompleted} + 1` }
			});
	} else {
		await db
			.update(activity)
			.set({ tasksCompleted: sql`MAX(0, ${activity.tasksCompleted} - 1)` })
			.where(eq(activity.date, today));
	}

	return db.select().from(tasks).where(eq(tasks.id, taskId)).limit(1);
}

export async function updateTaskNotes(taskId: number, notes: string) {
	await db.update(tasks).set({ notes }).where(eq(tasks.id, taskId));
	return db.select().from(tasks).where(eq(tasks.id, taskId)).limit(1);
}

// Stats
export async function getStats() {
	const allTasks = await db.select().from(tasks);
	const completedTasks = allTasks.filter((t) => t.completed);
	const totalPaths = await db.select({ count: sql<number>`count(*)` }).from(paths);

	// Calculate streak
	const recentActivity = await db
		.select()
		.from(activity)
		.where(sql`${activity.tasksCompleted} > 0`)
		.orderBy(desc(activity.date))
		.limit(30);

	let streak = 0;
	const today = new Date().toISOString().split('T')[0];
	const dates = recentActivity.map((a) => a.date);

	// Check if today or yesterday has activity
	const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
	if (dates.includes(today) || dates.includes(yesterday)) {
		let checkDate = dates.includes(today) ? new Date() : new Date(Date.now() - 86400000);

		while (true) {
			const dateStr = checkDate.toISOString().split('T')[0];
			if (dates.includes(dateStr)) {
				streak++;
				checkDate = new Date(checkDate.getTime() - 86400000);
			} else {
				break;
			}
		}
	}

	return {
		totalTasks: allTasks.length,
		completedTasks: completedTasks.length,
		totalPaths: totalPaths[0]?.count || 0,
		streak,
		completionRate: allTasks.length > 0
			? Math.round((completedTasks.length / allTasks.length) * 100)
			: 0
	};
}

export async function getRecentActivity() {
	return db
		.select()
		.from(activity)
		.orderBy(desc(activity.date))
		.limit(7);
}

// Get upcoming tasks (next 5 incomplete tasks)
export async function getUpcomingTasks() {
	return db
		.select({
			id: tasks.id,
			title: tasks.title,
			moduleName: modules.name,
			pathName: paths.name,
			pathId: paths.id
		})
		.from(tasks)
		.innerJoin(modules, eq(tasks.moduleId, modules.id))
		.innerJoin(paths, eq(modules.pathId, paths.id))
		.where(eq(tasks.completed, false))
		.orderBy(asc(modules.orderIndex), asc(tasks.orderIndex))
		.limit(5);
}
