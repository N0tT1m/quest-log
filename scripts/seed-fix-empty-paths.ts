import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

function addTasks(pathId: number, modules: { name: string; desc: string; tasks: [string, string][] }[]) {
	modules.forEach((mod, i) => {
		const m = insertModule.run(pathId, mod.name, mod.desc, i, now);
		mod.tasks.forEach(([title, desc], j) => {
			insertTask.run(m.lastInsertRowid, title, desc, '', j, now);
		});
	});
}

// Find all paths with 0 tasks
const emptyPaths = db.prepare(`
	SELECT p.id, p.name FROM paths p
	LEFT JOIN modules m ON m.path_id = p.id
	LEFT JOIN tasks t ON t.module_id = m.id
	GROUP BY p.id
	HAVING COUNT(t.id) = 0
`).all() as { id: number; name: string }[];

console.log(`Found ${emptyPaths.length} paths with no tasks`);

for (const path of emptyPaths) {
	console.log(`Adding tasks to: ${path.name}`);

	// Add generic but relevant tasks based on path name
	if (path.name.includes('Compiler') || path.name.includes('Programming Language')) {
		addTasks(path.id, [
			{ name: 'Lexical Analysis', desc: 'Build the lexer', tasks: [
				['Design token types', 'Define all tokens your language needs'],
				['Implement lexer', 'Convert source code to tokens'],
				['Handle whitespace and comments', 'Skip non-significant characters'],
				['Add error reporting', 'Report invalid tokens with line numbers'],
				['Write lexer tests', 'Test all token types'],
			]},
			{ name: 'Parsing', desc: 'Build the parser', tasks: [
				['Define grammar', 'Write BNF/EBNF grammar'],
				['Implement recursive descent parser', 'Parse expressions and statements'],
				['Build AST nodes', 'Create abstract syntax tree'],
				['Handle operator precedence', 'Correct parsing of expressions'],
				['Add parser error recovery', 'Continue parsing after errors'],
			]},
			{ name: 'Code Generation', desc: 'Generate output', tasks: [
				['Design IR or output format', 'Choose compilation target'],
				['Implement code generator', 'Walk AST and emit code'],
				['Handle control flow', 'Generate if/while/for'],
				['Implement functions', 'Function calls and returns'],
				['Add optimizations', 'Basic optimization passes'],
			]},
		]);
	} else if (path.name.includes('Deep Learning') || path.name.includes('ML') || path.name.includes('Transformer') || path.name.includes('AI')) {
		addTasks(path.id, [
			{ name: 'Foundations', desc: 'Core concepts', tasks: [
				['Understand neural network basics', 'Layers, activations, backprop'],
				['Implement gradient descent', 'Optimization from scratch'],
				['Build simple neural network', 'Forward and backward pass'],
				['Learn PyTorch/TensorFlow basics', 'Framework fundamentals'],
				['Understand loss functions', 'MSE, cross-entropy, etc.'],
			]},
			{ name: 'Architecture', desc: 'Model architectures', tasks: [
				['Implement attention mechanism', 'Self-attention from scratch'],
				['Build transformer block', 'Full transformer architecture'],
				['Add positional encoding', 'Position information'],
				['Implement layer normalization', 'Stabilize training'],
				['Build full model', 'Stack components together'],
			]},
			{ name: 'Training & Deployment', desc: 'Production ML', tasks: [
				['Set up training loop', 'Batch processing, logging'],
				['Implement learning rate scheduling', 'Warmup, decay'],
				['Add model checkpointing', 'Save and load models'],
				['Build inference pipeline', 'Efficient prediction'],
				['Deploy model', 'Serve predictions'],
			]},
		]);
	} else if (path.name.includes('Redis') || path.name.includes('SQLite') || path.name.includes('LSM') || path.name.includes('Key-Value')) {
		addTasks(path.id, [
			{ name: 'Data Structures', desc: 'Core structures', tasks: [
				['Implement hash table', 'Key-value storage'],
				['Build skip list or B-tree', 'Ordered data structure'],
				['Add expiration support', 'TTL handling'],
				['Implement persistence', 'Write to disk'],
				['Build WAL', 'Write-ahead logging'],
			]},
			{ name: 'Protocol & API', desc: 'Client interface', tasks: [
				['Design wire protocol', 'RESP or custom'],
				['Implement parser', 'Parse commands'],
				['Add command handlers', 'GET, SET, DELETE, etc.'],
				['Build client library', 'Easy-to-use API'],
				['Add pipelining', 'Batch commands'],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement transactions', 'MULTI/EXEC or similar'],
				['Add pub/sub', 'Message passing'],
				['Build replication', 'Master-replica sync'],
				['Add clustering', 'Distributed storage'],
				['Implement monitoring', 'Stats and metrics'],
			]},
		]);
	} else if (path.name.includes('Packet') || path.name.includes('DNS') || path.name.includes('Load Balancer') || path.name.includes('HTTP Server') || path.name.includes('TLS')) {
		addTasks(path.id, [
			{ name: 'Network Basics', desc: 'Foundation', tasks: [
				['Understand the protocol', 'Read RFC or specification'],
				['Set up raw sockets or library', 'Low-level networking'],
				['Implement packet parsing', 'Decode protocol messages'],
				['Build packet construction', 'Create valid packets'],
				['Add error handling', 'Handle malformed input'],
			]},
			{ name: 'Core Implementation', desc: 'Main functionality', tasks: [
				['Implement server/client', 'Basic communication'],
				['Add request handling', 'Process incoming requests'],
				['Build response generation', 'Create proper responses'],
				['Implement connection management', 'Handle multiple clients'],
				['Add logging', 'Debug and monitor'],
			]},
			{ name: 'Advanced Features', desc: 'Production ready', tasks: [
				['Add TLS support', 'Encryption'],
				['Implement caching', 'Performance optimization'],
				['Build health checks', 'Monitoring'],
				['Add configuration', 'Runtime settings'],
				['Performance testing', 'Benchmark and optimize'],
			]},
		]);
	} else if (path.name.includes('Ray Tracer')) {
		addTasks(path.id, [
			{ name: 'Ray Tracing Basics', desc: 'Core algorithm', tasks: [
				['Implement ray-sphere intersection', 'Basic geometry'],
				['Build camera model', 'Generate rays'],
				['Add basic shading', 'Diffuse lighting'],
				['Implement shadows', 'Shadow rays'],
				['Add reflections', 'Recursive ray tracing'],
			]},
			{ name: 'Materials & Lighting', desc: 'Realistic rendering', tasks: [
				['Implement materials', 'Diffuse, specular, etc.'],
				['Add multiple light sources', 'Point, directional'],
				['Build BVH acceleration', 'Faster intersection'],
				['Implement textures', 'Image mapping'],
				['Add anti-aliasing', 'Supersampling'],
			]},
		]);
	} else if (path.name.includes('Terraform') || path.name.includes('DevOps')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Basics', desc: 'Core concepts', tasks: [
				['Understand IaC principles', 'Infrastructure as code'],
				['Design resource model', 'State management'],
				['Implement provider interface', 'Cloud API abstraction'],
				['Build plan/apply workflow', 'Preview and execute'],
				['Add state management', 'Track resources'],
			]},
			{ name: 'Advanced Features', desc: 'Production features', tasks: [
				['Implement modules', 'Reusable components'],
				['Add variables and outputs', 'Parameterization'],
				['Build dependency graph', 'Execution order'],
				['Implement import', 'Adopt existing resources'],
				['Add remote state', 'Team collaboration'],
			]},
		]);
	} else if (path.name.includes('Password Manager')) {
		addTasks(path.id, [
			{ name: 'Cryptography', desc: 'Security foundation', tasks: [
				['Implement key derivation', 'PBKDF2 or Argon2'],
				['Add encryption', 'AES-256-GCM'],
				['Build secure storage', 'Encrypted vault'],
				['Implement master password', 'Single password entry'],
				['Add password generation', 'Random strong passwords'],
			]},
			{ name: 'User Interface', desc: 'Usability', tasks: [
				['Build CLI interface', 'Command-line access'],
				['Add search and filter', 'Find entries'],
				['Implement clipboard integration', 'Copy passwords'],
				['Add import/export', 'Backup and migrate'],
				['Build browser extension', 'Auto-fill support'],
			]},
		]);
	} else if (path.name.includes('Hacking') || path.name.includes('Security') || path.name.includes('Red Team') || path.name.includes('CTF')) {
		addTasks(path.id, [
			{ name: 'Reconnaissance', desc: 'Information gathering', tasks: [
				['Learn network scanning', 'Discover hosts and services'],
				['Understand enumeration', 'Extract information'],
				['Practice OSINT', 'Open source intelligence'],
				['Study web recon', 'Subdomain, directory discovery'],
				['Build target profile', 'Comprehensive information'],
			]},
			{ name: 'Exploitation', desc: 'Attack techniques', tasks: [
				['Learn common vulnerabilities', 'OWASP Top 10'],
				['Practice privilege escalation', 'Linux and Windows'],
				['Understand Active Directory attacks', 'Kerberos, NTLM'],
				['Study web exploitation', 'XSS, SQLi, SSRF'],
				['Practice post-exploitation', 'Persistence, pivoting'],
			]},
			{ name: 'Tooling', desc: 'Build your own', tasks: [
				['Build a scanner', 'Port or vulnerability scanner'],
				['Create an exploit', 'PoC for known CVE'],
				['Develop a payload', 'Reverse shell or similar'],
				['Write a parser', 'Parse exploit output'],
				['Automate workflow', 'Chain tools together'],
			]},
		]);
	} else if (path.name.includes('Homelab')) {
		addTasks(path.id, [
			{ name: 'Infrastructure Setup', desc: 'Build foundation', tasks: [
				['Set up hypervisor', 'Proxmox, ESXi, or similar'],
				['Configure networking', 'VLANs, firewall rules'],
				['Deploy domain controller', 'Active Directory'],
				['Add monitoring', 'Prometheus, Grafana'],
				['Set up logging', 'ELK stack or similar'],
			]},
			{ name: 'Security Lab', desc: 'Practice environment', tasks: [
				['Deploy vulnerable machines', 'Metasploitable, DVWA'],
				['Set up attack machine', 'Kali or Parrot'],
				['Configure C2 framework', 'Sliver, Mythic'],
				['Add detection tools', 'Elastic Security, Wazuh'],
				['Practice attack/defense', 'Red vs blue exercises'],
			]},
		]);
	} else {
		// Generic development tasks
		addTasks(path.id, [
			{ name: 'Foundation', desc: 'Core implementation', tasks: [
				['Research the domain', 'Understand requirements'],
				['Design architecture', 'Plan components'],
				['Set up project structure', 'Organize code'],
				['Implement core logic', 'Main functionality'],
				['Add error handling', 'Robust error management'],
			]},
			{ name: 'Features', desc: 'Build out functionality', tasks: [
				['Implement main features', 'Core use cases'],
				['Add configuration', 'Runtime settings'],
				['Build CLI or API', 'User interface'],
				['Add logging', 'Debug and monitor'],
				['Write tests', 'Verify correctness'],
			]},
			{ name: 'Polish', desc: 'Production ready', tasks: [
				['Optimize performance', 'Profile and improve'],
				['Add documentation', 'Usage and API docs'],
				['Handle edge cases', 'Robust handling'],
				['Package for distribution', 'Easy installation'],
				['Final testing', 'End-to-end verification'],
			]},
		]);
	}
}

console.log('Done adding tasks to empty paths!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
