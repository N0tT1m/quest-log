import type { PageServerLoad } from './$types';
import { getStats, getUpcomingTasks, getAllPaths, getRecentActivity } from '$lib/server/queries';
import { db, schema } from '$lib/server/db';
import { eq, sql } from 'drizzle-orm';

export const load: PageServerLoad = async () => {
	const stats = await getStats();
	const upcomingTasks = await getUpcomingTasks();
	const paths = await getAllPaths();
	const recentActivity = await getRecentActivity();

	// Get progress for each path
	const pathsWithProgress = await Promise.all(
		paths.map(async (path) => {
			const pathTasks = await db
				.select({
					total: sql<number>`count(*)`,
					completed: sql<number>`sum(case when ${schema.tasks.completed} = 1 then 1 else 0 end)`
				})
				.from(schema.tasks)
				.innerJoin(schema.modules, eq(schema.tasks.moduleId, schema.modules.id))
				.where(eq(schema.modules.pathId, path.id));

			const total = pathTasks[0]?.total || 0;
			const completed = pathTasks[0]?.completed || 0;

			return {
				...path,
				totalTasks: total,
				completedTasks: completed,
				progress: total > 0 ? Math.round((completed / total) * 100) : 0
			};
		})
	);

	return {
		stats,
		upcomingTasks,
		paths: pathsWithProgress,
		recentActivity
	};
};
