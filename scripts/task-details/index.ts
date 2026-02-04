export interface TaskDetail {
	id: number;
	title: string;
	details: string;
}

export interface PathDetails {
	pathName: string;
	pathId: number;
	tasks: TaskDetail[];
}

import aiMl from './ai-ml.json' assert { type: 'json' };
import securityRedteam from './security-redteam.json' assert { type: 'json' };
import systems from './systems.json' assert { type: 'json' };
import networking from './networking.json' assert { type: 'json' };
import databases from './databases.json' assert { type: 'json' };
import projects from './projects.json' assert { type: 'json' };
import specialized from './specialized.json' assert { type: 'json' };

export const allDetails: PathDetails[] = [
	...(aiMl as unknown as PathDetails[]),
	...(securityRedteam as unknown as PathDetails[]),
	...(systems as unknown as PathDetails[]),
	...(networking as unknown as PathDetails[]),
	...(databases as unknown as PathDetails[]),
	...(projects as unknown as PathDetails[]),
	...(specialized as unknown as PathDetails[])
];

export function getAllTaskDetails(): Map<number, string> {
	const map = new Map<number, string>();
	for (const path of allDetails) {
		for (const task of path.tasks) {
			map.set(task.id, task.details);
		}
	}
	return map;
}
