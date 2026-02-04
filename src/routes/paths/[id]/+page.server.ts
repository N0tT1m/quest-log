import type { PageServerLoad, Actions } from './$types';
import { getPathWithModulesAndTasks, toggleTask, updateTaskNotes } from '$lib/server/queries';
import { error } from '@sveltejs/kit';

export const load: PageServerLoad = async ({ params }) => {
	const pathId = parseInt(params.id);
	if (isNaN(pathId)) {
		throw error(400, 'Invalid path ID');
	}

	const path = await getPathWithModulesAndTasks(pathId);
	if (!path) {
		throw error(404, 'Path not found');
	}

	// Calculate stats
	const allTasks = path.modules.flatMap((m) => m.tasks);
	const completedTasks = allTasks.filter((t) => t.completed);

	return {
		path,
		stats: {
			totalTasks: allTasks.length,
			completedTasks: completedTasks.length,
			progress: allTasks.length > 0
				? Math.round((completedTasks.length / allTasks.length) * 100)
				: 0
		}
	};
};

export const actions: Actions = {
	toggle: async ({ request }) => {
		const formData = await request.formData();
		const taskId = parseInt(formData.get('taskId') as string);

		if (isNaN(taskId)) {
			return { success: false, error: 'Invalid task ID' };
		}

		const result = await toggleTask(taskId);
		return { success: true, task: result?.[0] };
	},

	updateNotes: async ({ request }) => {
		const formData = await request.formData();
		const taskId = parseInt(formData.get('taskId') as string);
		const notes = formData.get('notes') as string;

		if (isNaN(taskId)) {
			return { success: false, error: 'Invalid task ID' };
		}

		const result = await updateTaskNotes(taskId, notes);
		return { success: true, task: result?.[0] };
	}
};
