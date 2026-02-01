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
				['Implement multi-head attention', 'Split Q, K, V into h heads (e.g., 8 or 12), compute attention in parallel, concatenate outputs. Each head learns different relationships. Example: head 1 focuses on syntax, head 2 on semantics.'],
				['Build positional encoding', 'Add position information since attention is permutation-invariant. Sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d)). Or learned embeddings. RoPE for modern models.'],
				['Add layer normalization', 'Normalize activations across features. Pre-LN (before attention/FFN) is more stable for training. Post-LN (after residual) was original design. LayerNorm(x) = (x - μ) / σ * γ + β.'],
				['Implement feed-forward network', 'Two-layer MLP after attention: FFN(x) = GELU(xW₁)W₂. Hidden dim typically 4x model dim (e.g., 768 → 3072 → 768). Processes each position independently.'],
				['Build residual connections', 'Add input to output of each sublayer: x + Sublayer(x). Enables gradient flow through deep networks. Essential for training 100+ layer transformers.'],
			]);
			modId = getOrCreateModule(p.id, 'Training Techniques', 'Modern training methods', 21);
			addTasksToModule(modId, [
				['Implement gradient accumulation', 'Accumulate gradients over N mini-batches before optimizer step. Simulates batch_size * N with less memory. Example: 4 steps of batch=8 equals batch=32.'],
				['Add learning rate warmup', 'Start with small LR, linearly increase to target over warmup steps (e.g., 2000 steps). Prevents early instability. Then decay (cosine, linear, or constant).'],
				['Build mixed precision training', 'Use FP16/BF16 for forward/backward pass, FP32 for optimizer. 2x memory savings, faster compute on modern GPUs. torch.cuda.amp.autocast() context manager.'],
				['Implement gradient clipping', 'Clip gradient norm to max value (e.g., 1.0) to prevent exploding gradients. torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).'],
				['Add checkpoint saving', 'Save model state_dict, optimizer state, epoch, step periodically. Enable resume from failures. Save best model based on validation loss.'],
			]);
			modId = getOrCreateModule(p.id, 'Fine-tuning Methods', 'Adaptation techniques', 22);
			addTasksToModule(modId, [
				['Implement LoRA', 'Low-Rank Adaptation: freeze base model, add trainable low-rank matrices A (d×r) and B (r×d) where r<<d (e.g., 8). W\' = W + BA. 100x fewer trainable parameters.'],
				['Build QLoRA', 'Quantize base model to 4-bit NormalFloat, apply LoRA adapters in FP16. Enables fine-tuning 65B models on single GPU. Uses double quantization for memory efficiency.'],
				['Add prompt tuning', 'Prepend learnable soft prompt embeddings (e.g., 20 virtual tokens) to input. Only train prompt parameters, freeze model. Task-specific prompts for multi-task.'],
				['Implement adapter layers', 'Insert small bottleneck layers (down-project → nonlinearity → up-project) between transformer layers. Train only adapters. ~3% additional parameters per task.'],
				['Build instruction tuning pipeline', 'Fine-tune on instruction-response pairs. Format: "### Instruction:\\n{task}\\n### Response:\\n{answer}". Teaches model to follow diverse instructions.'],
			]);
		} else if (p.name.includes('ML Pipeline') || p.name.includes('MLOps')) {
			let modId = getOrCreateModule(p.id, 'Data Pipeline', 'Data engineering', 20);
			addTasksToModule(modId, [
				['Build data ingestion', 'Pull data from multiple sources: S3, databases, APIs, streaming (Kafka). Handle batching, retries, schema validation. Apache Airflow or Prefect for orchestration.'],
				['Implement data validation', 'Use Great Expectations or similar: define expectations (column not null, values in range), run validation, fail pipeline on violations. Generate data quality reports.'],
				['Add feature engineering', 'Transform raw data into ML features: scaling, encoding categoricals, datetime extraction, aggregations. Use pandas, Spark, or dbt. Document feature logic.'],
				['Build feature store', 'Centralized feature repository (Feast, Tecton). Store feature definitions, compute and serve features consistently for training and inference. Avoid training-serving skew.'],
				['Implement data versioning', 'Track dataset versions with DVC or Delta Lake. Link model version to exact training data. Enable reproducibility: checkout data version, retrain, get same model.'],
			]);
			modId = getOrCreateModule(p.id, 'Model Management', 'MLOps workflow', 21);
			addTasksToModule(modId, [
				['Set up experiment tracking', 'Log hyperparameters, metrics, artifacts with MLflow. Compare runs, visualize learning curves. Tag best runs. Example: mlflow.log_param("lr", 0.001), mlflow.log_metric("accuracy", 0.95).'],
				['Build model registry', 'Version models with metadata (metrics, training data version). Stages: None → Staging → Production → Archived. Approval workflow before production promotion.'],
				['Implement CI/CD for ML', 'Trigger training on data/code changes. Run validation tests. Build container images. Deploy to staging, run integration tests, promote to production. GitHub Actions or Jenkins.'],
				['Add model monitoring', 'Track prediction distribution, input data drift (KS test, PSI), model performance on labeled data. Alert on drift threshold. Retrain trigger. Use Evidently AI or custom dashboards.'],
				['Build A/B testing framework', 'Split traffic between model versions (e.g., 90/10). Track business metrics per variant. Statistical significance testing. Gradual rollout: 10% → 50% → 100%.'],
			]);
		} else if (p.name.includes('Metasploit') || p.name.includes('C2') || p.name.includes('Cobalt')) {
			let modId = getOrCreateModule(p.id, 'Core Framework', 'Base implementation', 20);
			addTasksToModule(modId, [
				['Build module loader', 'Dynamically load exploit/payload modules at runtime. Plugin architecture with common interface. Module metadata: name, author, targets, options. Hot-reload during operation.'],
				['Implement exploit interface', 'Standard API for exploits: check() to test vulnerability, exploit() to execute, options for target/port/payload. Return session on success. Handle common patterns: connect, send, receive.'],
				['Add payload generation', 'Generate payloads in multiple formats: raw shellcode, EXE, DLL, PowerShell, Python, VBA macro. Embed configuration (C2 address, sleep time). Cross-platform support.'],
				['Build encoder system', 'Encode payloads to evade detection: XOR, shikata_ga_nai (polymorphic), base64. Chain encoders. Avoid bad characters (null bytes). Prepend decoder stub.'],
				['Implement session handler', 'Manage active connections: track sessions by ID, route commands to correct session, handle disconnects gracefully, support multiple session types (shell, meterpreter).'],
			]);
			modId = getOrCreateModule(p.id, 'Post-Exploitation', 'After initial access', 21);
			addTasksToModule(modId, [
				['Build process injection', 'Multiple techniques: CreateRemoteThread classic, QueueUserAPC for early-bird, process hollowing, module stomping. Target selection: find suitable host process.'],
				['Implement credential harvesting', 'Extract from memory: LSASS dump (MiniDumpWriteDump), SAM database, browser passwords, cached credentials. Parse for NTLM hashes, Kerberos tickets, plaintext.'],
				['Add lateral movement', 'Move through network: WMI execution, SMB PsExec, WinRM, DCOM. Pass credentials or use current token. Target selection based on recon data.'],
				['Build persistence mechanisms', 'Survive reboots: Registry run keys, scheduled tasks, services, WMI subscriptions. Choose based on privileges (user vs admin). Avoid common detection signatures.'],
				['Implement privilege escalation', 'Elevate permissions: check for misconfigured services, unquoted paths, AlwaysInstallElevated, token impersonation (Potato attacks). Automated privesc checking.'],
			]);
		} else if (p.name.includes('Impacket') || p.name.includes('SMB') || p.name.includes('Kerberos')) {
			let modId = getOrCreateModule(p.id, 'Protocol Implementation', 'Core protocols', 20);
			addTasksToModule(modId, [
				['Implement SMB client', 'Build SMB2/3 client: negotiate dialect, session setup with NTLM/Kerberos, tree connect to shares. File operations: list, read, write, delete. Handle signing and encryption.'],
				['Build DCE/RPC', 'Implement DCE/RPC over SMB named pipes. Bind to interfaces (SAMR, LSARPC, SVCCTL) by UUID. Marshal/unmarshal NDR data. Call remote procedures.'],
				['Add NTLM authentication', 'Implement NTLM challenge-response: Type1 (negotiate), Type2 (challenge from server), Type3 (response with hash). Support NTLMv1, NTLMv2, pass-the-hash.'],
				['Implement Kerberos client', 'AS-REQ for TGT (with password/hash/cert), TGS-REQ for service tickets. Handle encryption types (RC4, AES128, AES256). Parse tickets, extract PAC.'],
				['Build LDAP queries', 'LDAP client: bind (simple/SASL), search with filters (objectClass=user), paged results for large datasets. Query users, groups, computers, GPOs.'],
			]);
			modId = getOrCreateModule(p.id, 'Attack Tools', 'Offensive capabilities', 21);
			addTasksToModule(modId, [
				['Build secretsdump', 'Extract credentials remotely: SAM via registry (local accounts), LSA secrets (service passwords), cached domain creds, NTDS.dit via DCSync. No local code execution needed.'],
				['Implement psexec', 'Remote execution: copy service binary to ADMIN$, create service via SVCCTL RPC, start service, read output via named pipe. Clean up afterward.'],
				['Add wmiexec', 'Execute via WMI Win32_Process.Create(). Semi-interactive shell. Redirect output to file on C$ share, read back, delete. Stealthier than psexec (no service creation).'],
				['Build smbexec', 'Create service with cmd.exe command, output to share. No binary upload. Most stealthy of exec methods. Service name randomization.'],
				['Implement GetNPUsers', 'AS-REP roasting: find users without Kerberos preauth (DONT_REQUIRE_PREAUTH flag), request AS-REP, extract encrypted timestamp for offline cracking.'],
			]);
		} else if (p.name.includes('Terraform') || p.name.includes('Infrastructure')) {
			let modId = getOrCreateModule(p.id, 'Core Engine', 'Main functionality', 20);
			addTasksToModule(modId, [
				['Build resource graph', 'Parse HCL configuration, build DAG of resources based on dependencies (references, depends_on). Topological sort for execution order. Detect cycles.'],
				['Implement plan phase', 'Compare desired state (config) with current state (state file). Generate diff: resources to create, update, or destroy. Show human-readable plan output.'],
				['Add apply phase', 'Execute plan: create/update/destroy resources via provider APIs. Update state file after each operation. Handle errors with partial state. Parallelism where possible.'],
				['Build destroy phase', 'Reverse dependency order. Destroy all managed resources. Update state to empty. Handle destroy-time provisioners. Confirm destructive action.'],
				['Implement refresh', 'Read current state of all resources from provider APIs. Update state file to match reality. Detect drift from desired configuration.'],
			]);
			modId = getOrCreateModule(p.id, 'Provider System', 'Cloud integrations', 21);
			addTasksToModule(modId, [
				['Design provider interface', 'Define CRUD operations: Create, Read, Update, Delete, Exists. Provider initializes with credentials. Returns resource data or errors. Plugin architecture.'],
				['Implement resource schema', 'Define resource attributes: name, type (string, int, list, map), required/optional, computed, sensitive. Validation rules. Document for user reference.'],
				['Add data sources', 'Read-only resources that fetch existing infrastructure data. Example: aws_ami data source to find latest AMI ID. Use in other resources.'],
				['Build provisioners', 'Run scripts after resource creation: local-exec (on Terraform host), remote-exec (SSH to resource). File provisioner for copying files. Use sparingly.'],
				['Implement backends', 'Store state remotely: S3 + DynamoDB (locking), Terraform Cloud, Consul, PostgreSQL. State locking prevents concurrent modifications. Encryption at rest.'],
			]);
		} else if (p.name.includes('Red Team') || p.name.includes('Malware') || p.name.includes('AD')) {
			let modId = getOrCreateModule(p.id, 'Offensive Techniques', 'Attack methods', 20);
			addTasksToModule(modId, [
				['Implement process hollowing', 'Create suspended process, unmap original code (NtUnmapViewOfSection), write payload, set thread context to new entry point, resume. Process runs payload but looks legitimate.'],
				['Build DLL injection', 'Classic: VirtualAllocEx + WriteProcessMemory + CreateRemoteThread with LoadLibrary. Manual mapping: copy PE, resolve imports, call DllMain. Reflective: DLL loads itself from memory.'],
				['Add syscall evasion', 'Bypass usermode hooks by calling kernel directly. Get syscall numbers from ntdll.dll. Build stubs: mov r10,rcx; mov eax,SSN; syscall; ret. SysWhispers, HellsGate tools.'],
				['Implement AMSI bypass', 'Patch AmsiScanBuffer to return clean. Or patch amsi.dll in memory. Or use obfuscation to avoid signatures. PowerShell: [Ref].Assembly... to access AMSI context.'],
				['Build ETW patching', 'Disable Event Tracing: patch EtwEventWrite to ret. Or remove provider registrations. Prevents security tools from receiving telemetry. Also patch NtTraceEvent.'],
			]);
			modId = getOrCreateModule(p.id, 'Domain Attacks', 'Active Directory', 21);
			addTasksToModule(modId, [
				['Implement DCSync', 'Replicate credentials using DRSUAPI (what DCs use). Requires Replicating Directory Changes rights. lsadump::dcsync extracts all NTLM hashes without touching DC disk.'],
				['Build golden ticket', 'Forge TGT with krbtgt hash: any user, any groups, 10-year validity. kerberos::golden /user:admin /domain:corp.com /sid:S-1-5-21-... /krbtgt:hash. Complete domain compromise.'],
				['Add silver ticket', 'Forge TGS with service account hash. Target specific service (CIFS, HTTP, MSSQL). No DC contact needed. Service account hash from Kerberoast or local extraction.'],
				['Implement delegation abuse', 'Unconstrained: extract TGTs from memory when admin connects. Constrained: S4U2Self + S4U2Proxy to impersonate users. RBCD: modify msDS-AllowedToActOnBehalfOfOtherIdentity.'],
				['Build trust attacks', 'Cross-domain: inter-realm TGT with SID history. Forest: krbtgt trust key. Child-to-parent: escalate from child domain to forest root. ExtraSIDs to add Enterprise Admins SID.'],
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
