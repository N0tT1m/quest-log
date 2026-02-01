import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);
const now = Date.now();

function addTasksToModule(moduleId: number, tasks: [string, string][]) {
	const existing = db.prepare('SELECT MAX(order_index) as max FROM tasks WHERE module_id = ?').get(moduleId) as { max: number };
	let idx = (existing.max || 0) + 1;
	tasks.forEach(([title, desc]) => {
		insertTask.run(moduleId, title, desc, '', idx++, now);
	});
}

function getOrCreateModule(pathId: number, name: string, desc: string, orderIdx: number): number {
	const existing = db.prepare('SELECT id FROM modules WHERE path_id = ? AND name = ?').get(pathId, name) as { id: number } | undefined;
	if (existing) return existing.id;
	const result = insertModule.run(pathId, name, desc, orderIdx, now);
	return Number(result.lastInsertRowid);
}

// Find paths with schedules that have more content than tasks
const paths = db.prepare(`
  SELECT p.id, p.name, p.schedule,
    (SELECT COUNT(*) FROM modules m JOIN tasks t ON t.module_id = m.id WHERE m.path_id = p.id) as task_count
  FROM paths p
  WHERE p.schedule IS NOT NULL AND p.schedule != ''
`).all() as { id: number; name: string; schedule: string; task_count: number }[];

console.log('Checking paths for schedule/task mismatch...');

let expanded = 0;

paths.forEach((p) => {
	const schedule = p.schedule || '';
	const weeks = (schedule.match(/week\s*\d+/gi) || []).length;
	const bullets = (schedule.match(/^\s*[-•*]/gm) || []).length;
	const lines = schedule.split('\n').filter((l) => l.trim().length > 5).length;

	const scheduleItems = Math.max(weeks * 3, bullets, Math.floor(lines / 2));
	const ratio = scheduleItems > 0 ? p.task_count / scheduleItems : 1;

	// If tasks are less than 50% of schedule items and schedule is substantial
	if (ratio < 0.5 && scheduleItems > 15) {
		console.log(`\nExpanding: ${p.name} (${p.task_count} tasks vs ~${scheduleItems} schedule items)`);
		expanded++;

		// Add tasks based on path name patterns
		if (p.name.includes('Transformer') || p.name.includes('LLM') || p.name.includes('Deep Learning')) {
			let modId = getOrCreateModule(p.id, 'Advanced Architecture', 'Deep dive into model architecture', 20);
			addTasksToModule(modId, [
				['Implement multi-head attention', 'Parallel attention heads'],
				['Build positional encoding', 'Sinusoidal and learned'],
				['Add layer normalization', 'Pre-LN vs Post-LN'],
				['Implement feed-forward network', 'MLP layers'],
				['Build residual connections', 'Skip connections'],
			]);
			modId = getOrCreateModule(p.id, 'Training Techniques', 'Modern training methods', 21);
			addTasksToModule(modId, [
				['Implement gradient accumulation', 'Simulate larger batches'],
				['Add learning rate warmup', 'Gradual LR increase'],
				['Build mixed precision training', 'FP16/BF16'],
				['Implement gradient clipping', 'Prevent exploding gradients'],
				['Add checkpoint saving', 'Resume training'],
			]);
			modId = getOrCreateModule(p.id, 'Fine-tuning Methods', 'Adaptation techniques', 22);
			addTasksToModule(modId, [
				['Implement LoRA', 'Low-rank adaptation'],
				['Build QLoRA', 'Quantized LoRA'],
				['Add prompt tuning', 'Soft prompts'],
				['Implement adapter layers', 'Bottleneck adapters'],
				['Build instruction tuning pipeline', 'Following instructions'],
			]);
		} else if (p.name.includes('ML Pipeline') || p.name.includes('MLOps')) {
			let modId = getOrCreateModule(p.id, 'Data Pipeline', 'Data engineering', 20);
			addTasksToModule(modId, [
				['Build data ingestion', 'Multiple sources'],
				['Implement data validation', 'Great Expectations'],
				['Add feature engineering', 'Transform features'],
				['Build feature store', 'Feast or similar'],
				['Implement data versioning', 'DVC'],
			]);
			modId = getOrCreateModule(p.id, 'Model Management', 'MLOps workflow', 21);
			addTasksToModule(modId, [
				['Set up experiment tracking', 'MLflow'],
				['Build model registry', 'Version models'],
				['Implement CI/CD for ML', 'Automated pipelines'],
				['Add model monitoring', 'Drift detection'],
				['Build A/B testing framework', 'Compare models'],
			]);
		} else if (p.name.includes('Metasploit') || p.name.includes('C2') || p.name.includes('Cobalt')) {
			let modId = getOrCreateModule(p.id, 'Core Framework', 'Base implementation', 20);
			addTasksToModule(modId, [
				['Build module loader', 'Dynamic loading'],
				['Implement exploit interface', 'Standard API'],
				['Add payload generation', 'Multiple formats'],
				['Build encoder system', 'Payload encoding'],
				['Implement session handler', 'Manage connections'],
			]);
			modId = getOrCreateModule(p.id, 'Post-Exploitation', 'After initial access', 21);
			addTasksToModule(modId, [
				['Build process injection', 'Multiple techniques'],
				['Implement credential harvesting', 'Memory extraction'],
				['Add lateral movement', 'Network traversal'],
				['Build persistence mechanisms', 'Maintain access'],
				['Implement privilege escalation', 'Elevate permissions'],
			]);
		} else if (p.name.includes('Impacket') || p.name.includes('SMB') || p.name.includes('Kerberos')) {
			let modId = getOrCreateModule(p.id, 'Protocol Implementation', 'Core protocols', 20);
			addTasksToModule(modId, [
				['Implement SMB client', 'File sharing'],
				['Build DCE/RPC', 'Remote procedure calls'],
				['Add NTLM authentication', 'Windows auth'],
				['Implement Kerberos client', 'Ticket requests'],
				['Build LDAP queries', 'Directory access'],
			]);
			modId = getOrCreateModule(p.id, 'Attack Tools', 'Offensive capabilities', 21);
			addTasksToModule(modId, [
				['Build secretsdump', 'Extract credentials'],
				['Implement psexec', 'Remote execution'],
				['Add wmiexec', 'WMI execution'],
				['Build smbexec', 'SMB execution'],
				['Implement GetNPUsers', 'AS-REP roasting'],
			]);
		} else if (p.name.includes('Terraform') || p.name.includes('Infrastructure')) {
			let modId = getOrCreateModule(p.id, 'Core Engine', 'Main functionality', 20);
			addTasksToModule(modId, [
				['Build resource graph', 'Dependency management'],
				['Implement plan phase', 'Preview changes'],
				['Add apply phase', 'Execute changes'],
				['Build destroy phase', 'Remove resources'],
				['Implement refresh', 'Sync state'],
			]);
			modId = getOrCreateModule(p.id, 'Provider System', 'Cloud integrations', 21);
			addTasksToModule(modId, [
				['Design provider interface', 'CRUD operations'],
				['Implement resource schema', 'Type definitions'],
				['Add data sources', 'Read-only resources'],
				['Build provisioners', 'Local/remote exec'],
				['Implement backends', 'State storage'],
			]);
		} else if (p.name.includes('Red Team') || p.name.includes('Malware') || p.name.includes('AD')) {
			let modId = getOrCreateModule(p.id, 'Offensive Techniques', 'Attack methods', 20);
			addTasksToModule(modId, [
				['Implement process hollowing', 'Code injection'],
				['Build DLL injection', 'Load into process'],
				['Add syscall evasion', 'Direct syscalls'],
				['Implement AMSI bypass', 'Avoid detection'],
				['Build ETW patching', 'Disable logging'],
			]);
			modId = getOrCreateModule(p.id, 'Domain Attacks', 'Active Directory', 21);
			addTasksToModule(modId, [
				['Implement DCSync', 'Credential replication'],
				['Build golden ticket', 'Forge TGT'],
				['Add silver ticket', 'Forge service ticket'],
				['Implement delegation abuse', 'Constrained delegation'],
				['Build trust attacks', 'Cross-domain'],
			]);
		}
	}
});

console.log(`\nExpanded ${expanded} paths`);

const counts = db.prepare(`
  SELECT
    (SELECT COUNT(*) FROM paths) as paths,
    (SELECT COUNT(*) FROM modules) as modules,
    (SELECT COUNT(*) FROM tasks) as tasks
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final: ${counts.paths} paths, ${counts.modules} modules, ${counts.tasks} tasks`);

db.close();
