import Database from 'better-sqlite3';

const db = new Database('data/quest-log.db');

const insertModule = db.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = db.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

function addModuleWithTasks(pathId: number, moduleName: string, moduleDesc: string, orderIndex: number, tasks: [string, string, string][]) {
	const mod = insertModule.run(pathId, moduleName, moduleDesc, orderIndex, now);
	tasks.forEach(([title, desc, details], i) => {
		insertTask.run(mod.lastInsertRowid, title, desc, details, i, now);
	});
	return mod.lastInsertRowid;
}

// Get all paths with few tasks
const pathsNeedingTasks = db.prepare(`
	SELECT p.id, p.name FROM paths p
	LEFT JOIN modules m ON m.path_id = p.id
	LEFT JOIN tasks t ON t.module_id = m.id
	GROUP BY p.id
	HAVING COUNT(t.id) < 10
`).all() as { id: number; name: string }[];

console.log(`Found ${pathsNeedingTasks.length} paths needing tasks`);

// ============================================================================
// Build Your Own Container Runtime
// ============================================================================
const containerPath = pathsNeedingTasks.find(p => p.name.includes('Container Runtime'));
if (containerPath) {
	addModuleWithTasks(containerPath.id, 'Week 1-2: Namespaces', 'Linux namespace isolation', 0, [
		['Understand Linux namespaces', 'Process isolation primitives', `## Namespaces Overview

Types:
- PID: process ID isolation
- NET: network isolation
- MNT: mount point isolation
- UTS: hostname isolation
- IPC: inter-process communication
- USER: user ID mapping`],
		['Implement PID namespace', 'Isolate process tree', `## PID Namespace

\`\`\`c
clone(child_fn, stack, CLONE_NEWPID | SIGCHLD, NULL);
\`\`\`

Child sees itself as PID 1`],
		['Add network namespace', 'Isolated networking', `## Network Namespace

Create isolated network stack:
- Own interfaces
- Own routing table
- Own iptables rules`],
		['Implement mount namespace', 'Filesystem isolation', `## Mount Namespace

New mount table
pivot_root to new root filesystem
Unmount host filesystems`],
		['Add UTS namespace', 'Hostname isolation', `## UTS Namespace

Container has own hostname:
\`\`\`c
sethostname("container", 9);
\`\`\``],
		['Implement user namespace', 'UID/GID mapping', `## User Namespace

Map container root to unprivileged host user:
\`\`\`
uid_map: 0 1000 1
gid_map: 0 1000 1
\`\`\``],
	]);

	addModuleWithTasks(containerPath.id, 'Week 3-4: Cgroups & Filesystem', 'Resource limits and rootfs', 1, [
		['Understand cgroups v2', 'Resource control', `## Cgroups v2

Unified hierarchy at /sys/fs/cgroup
Controllers:
- cpu: CPU time limits
- memory: memory limits
- io: disk I/O limits
- pids: process count limits`],
		['Implement memory limits', 'Restrict container memory', `## Memory Limits

\`\`\`bash
echo 100M > /sys/fs/cgroup/container/memory.max
echo 1 > /sys/fs/cgroup/container/memory.swap.max
\`\`\``],
		['Add CPU limits', 'Control CPU usage', `## CPU Limits

\`\`\`bash
echo "100000 100000" > cpu.max  # 100% of one CPU
echo "50000 100000" > cpu.max   # 50% of one CPU
\`\`\``],
		['Build rootfs from scratch', 'Create minimal filesystem', `## Rootfs Creation

1. Create directory structure
2. Copy busybox binary
3. Create device nodes
4. Set up /etc files`],
		['Implement overlay filesystem', 'Layered images', `## OverlayFS

\`\`\`bash
mount -t overlay overlay \\
  -o lowerdir=base,upperdir=changes,workdir=work \\
  merged/
\`\`\``],
		['Add image layer support', 'Docker-style layers', `## Image Layers

Each layer is a tarball
Stack layers with overlayfs
Copy-on-write for changes`],
	]);

	addModuleWithTasks(containerPath.id, 'Week 5-6: Networking & CLI', 'Container networking and interface', 2, [
		['Create virtual ethernet pairs', 'veth for networking', `## Veth Pairs

\`\`\`bash
ip link add veth0 type veth peer name veth1
ip link set veth1 netns container
\`\`\``],
		['Set up bridge networking', 'Connect containers', `## Bridge Network

\`\`\`bash
ip link add br0 type bridge
ip link set veth0 master br0
ip addr add 10.0.0.1/24 dev br0
\`\`\``],
		['Implement port forwarding', 'Expose container ports', `## Port Forward

\`\`\`bash
iptables -t nat -A PREROUTING \\
  -p tcp --dport 8080 \\
  -j DNAT --to-destination 10.0.0.2:80
\`\`\``],
		['Build CLI interface', 'docker-like commands', `## CLI Design

\`\`\`
mycontainer run alpine /bin/sh
mycontainer ps
mycontainer stop <id>
mycontainer images
\`\`\``],
		['Add image pull support', 'Download from registry', `## Image Pull

HTTP API to registry:
GET /v2/library/alpine/manifests/latest
GET /v2/library/alpine/blobs/<digest>`],
		['Implement exec command', 'Run in existing container', `## Exec

Enter running container's namespaces:
\`\`\`c
setns(fd, CLONE_NEWPID);
setns(fd, CLONE_NEWNET);
// ...
\`\`\``],
	]);
}

// ============================================================================
// Build Your Own Debugger
// ============================================================================
const debuggerPath = pathsNeedingTasks.find(p => p.name.includes('Debugger'));
if (debuggerPath) {
	addModuleWithTasks(debuggerPath.id, 'Week 1-2: ptrace Basics', 'Process tracing fundamentals', 0, [
		['Understand ptrace system call', 'Debugger foundation', `## ptrace Overview

Operations:
- PTRACE_TRACEME: allow parent to trace
- PTRACE_ATTACH: attach to process
- PTRACE_PEEKDATA: read memory
- PTRACE_POKEDATA: write memory
- PTRACE_SINGLESTEP: execute one instruction
- PTRACE_CONT: continue execution`],
		['Implement attach/detach', 'Connect to process', `## Attach

\`\`\`c
ptrace(PTRACE_ATTACH, pid, NULL, NULL);
waitpid(pid, &status, 0);  // wait for stop
// ... debug ...
ptrace(PTRACE_DETACH, pid, NULL, NULL);
\`\`\``],
		['Read process memory', 'Examine memory contents', `## Memory Reading

\`\`\`c
long data = ptrace(PTRACE_PEEKDATA, pid, addr, NULL);
\`\`\`

Read in word-sized chunks`],
		['Write process memory', 'Modify memory contents', `## Memory Writing

\`\`\`c
ptrace(PTRACE_POKEDATA, pid, addr, new_value);
\`\`\`

Used for breakpoints and patches`],
		['Get/set registers', 'Access CPU state', `## Registers

\`\`\`c
struct user_regs_struct regs;
ptrace(PTRACE_GETREGS, pid, NULL, &regs);
printf("RIP: %llx\\n", regs.rip);
\`\`\``],
		['Implement single-stepping', 'Execute one instruction', `## Single Step

\`\`\`c
ptrace(PTRACE_SINGLESTEP, pid, NULL, NULL);
waitpid(pid, &status, 0);
\`\`\``],
	]);

	addModuleWithTasks(debuggerPath.id, 'Week 3-4: Breakpoints', 'Software breakpoints', 1, [
		['Understand INT3 instruction', 'x86 breakpoint mechanism', `## INT3

Opcode 0xCC triggers SIGTRAP
CPU stops execution
Debugger regains control`],
		['Implement software breakpoints', 'Set/clear breakpoints', `## Setting Breakpoint

1. Read original byte at address
2. Write 0xCC (INT3)
3. On hit: restore original, step back, re-set`],
		['Build breakpoint manager', 'Track active breakpoints', `## Breakpoint Manager

\`\`\`c
struct breakpoint {
    void* addr;
    uint8_t original_byte;
    bool enabled;
};
\`\`\``],
		['Handle breakpoint hits', 'Process SIGTRAP', `## Hit Handler

1. Get RIP (points past INT3)
2. Find breakpoint at RIP-1
3. Restore original byte
4. Decrement RIP
5. Single-step
6. Re-enable breakpoint`],
		['Add conditional breakpoints', 'Break on condition', `## Conditional

When breakpoint hits:
1. Evaluate condition (register value, memory)
2. If false, continue automatically`],
		['Implement watchpoints', 'Break on memory access', `## Watchpoints

Use hardware debug registers (DR0-DR3)
Break on read, write, or execute`],
	]);

	addModuleWithTasks(debuggerPath.id, 'Week 5-6: Symbols & UI', 'Debug information and interface', 2, [
		['Parse ELF symbol tables', 'Load debug symbols', `## Symbol Loading

Read .symtab and .dynsym sections
Map addresses to function names
Support stripped binaries (partial)`],
		['Parse DWARF debug info', 'Source-level debugging', `## DWARF

Sections:
- .debug_info: types, variables
- .debug_line: line numbers
- .debug_frame: stack unwinding`],
		['Implement stack unwinding', 'Backtrace support', `## Stack Trace

Using frame pointers:
\`\`\`c
while (rbp != 0) {
    rip = *(rbp + 8);
    rbp = *rbp;
    print_frame(rip);
}
\`\`\``],
		['Build command-line interface', 'gdb-like commands', `## Commands

- break <addr/func>
- continue
- step / next
- print <expr>
- backtrace
- info registers`],
		['Add source code display', 'Show source at PC', `## Source Display

Using DWARF line info:
1. Find file/line for current RIP
2. Read source file
3. Display context around current line`],
		['Implement expression evaluation', 'Evaluate debug expressions', `## Expressions

Parse and evaluate:
- Variable names
- Register names ($rax)
- Memory dereference (*ptr)
- Arithmetic`],
	]);
}

// ============================================================================
// Build Your Own Memory Allocator
// ============================================================================
const allocatorPath = pathsNeedingTasks.find(p => p.name.includes('Memory Allocator'));
if (allocatorPath) {
	addModuleWithTasks(allocatorPath.id, 'Week 1-2: Basic Allocator', 'Simple memory management', 0, [
		['Understand sbrk/mmap', 'Getting memory from OS', `## System Calls

sbrk(n): extend heap by n bytes
mmap: map anonymous memory pages
Both return raw memory to manage`],
		['Implement simple bump allocator', 'Simplest allocator', `## Bump Allocator

\`\`\`c
void* alloc(size_t size) {
    void* ptr = heap_end;
    heap_end += size;
    return ptr;
}
\`\`\`

No free support, but very fast`],
		['Build free list allocator', 'Track free blocks', `## Free List

\`\`\`c
struct block {
    size_t size;
    struct block* next;
};
\`\`\`

Search list for fitting block on alloc`],
		['Implement first-fit strategy', 'Simple allocation strategy', `## First Fit

Return first block >= requested size
Fast but causes fragmentation`],
		['Add best-fit strategy', 'Minimize waste', `## Best Fit

Search entire list
Return smallest fitting block
Less fragmentation, slower`],
		['Implement block coalescing', 'Merge adjacent free blocks', `## Coalescing

When freeing:
- Check if prev block is free -> merge
- Check if next block is free -> merge
- Reduces fragmentation`],
	]);

	addModuleWithTasks(allocatorPath.id, 'Week 3-4: Advanced Allocator', 'Size classes and segregated lists', 1, [
		['Design size classes', 'Bucket similar sizes', `## Size Classes

Classes: 16, 32, 64, 128, 256, 512, 1024, ...
Round up requests to nearest class
Eliminates fragmentation within class`],
		['Build segregated free lists', 'Per-class free lists', `## Segregated Lists

\`\`\`c
struct free_list* lists[NUM_CLASSES];
// lists[0] for 16-byte blocks
// lists[1] for 32-byte blocks
// ...
\`\`\``],
		['Implement splitting', 'Split large blocks', `## Block Splitting

If block too large for request:
1. Allocate requested size
2. Put remainder in free list`],
		['Add block headers', 'Track metadata', `## Block Header

\`\`\`c
struct header {
    size_t size;     // block size
    bool allocated;  // in use?
    // maybe: size class, magic number
};
\`\`\``],
		['Implement boundary tags', 'Footer for coalescing', `## Boundary Tags

Header at start AND footer at end
Footer allows finding prev block's size
Enables bidirectional coalescing`],
		['Build explicit free list', 'Doubly-linked free blocks', `## Explicit List

Free blocks contain:
- Size (header)
- Next free pointer
- Prev free pointer
- Size (footer)

Faster than searching all blocks`],
	]);

	addModuleWithTasks(allocatorPath.id, 'Week 5-6: Production Features', 'Thread safety and debugging', 2, [
		['Add thread safety', 'Handle concurrent access', `## Thread Safety

Options:
1. Global mutex (simple, slow)
2. Per-size-class locks
3. Thread-local caches (tcmalloc style)`],
		['Implement thread-local caches', 'Per-thread allocation', `## Thread Local

Each thread has small cache
Most allocs/frees hit cache
Only lock when cache empty/full`],
		['Build memory pool', 'Fixed-size allocations', `## Memory Pool

Pre-allocate N objects of same size
Very fast for specific object types
Used in games, networking`],
		['Add debugging features', 'Detect memory errors', `## Debug Mode

- Canaries: detect buffer overflow
- Red zones: padding between blocks
- Track allocation sites (caller address)
- Detect double-free`],
		['Implement memory statistics', 'Track usage', `## Statistics

Track:
- Total allocated
- Peak usage
- Allocation count
- Free count
- Fragmentation ratio`],
		['Match malloc/free interface', 'Drop-in replacement', `## Interface

\`\`\`c
void* my_malloc(size_t size);
void my_free(void* ptr);
void* my_realloc(void* ptr, size_t size);
void* my_calloc(size_t n, size_t size);
\`\`\``],
	]);
}

// ============================================================================
// Build Your Own Thread Pool
// ============================================================================
const threadPoolPath = pathsNeedingTasks.find(p => p.name.includes('Thread Pool'));
if (threadPoolPath) {
	addModuleWithTasks(threadPoolPath.id, 'Week 1-2: Core Thread Pool', 'Basic work queue', 0, [
		['Create worker threads', 'Spawn thread pool', `## Worker Creation

\`\`\`c
for (int i = 0; i < num_threads; i++) {
    pthread_create(&threads[i], NULL, worker_fn, pool);
}
\`\`\``],
		['Build work queue', 'Thread-safe task queue', `## Work Queue

\`\`\`c
struct task {
    void (*fn)(void*);
    void* arg;
};
struct queue {
    task* tasks;
    int head, tail;
    pthread_mutex_t lock;
    pthread_cond_t not_empty;
};
\`\`\``],
		['Implement task submission', 'Add work to queue', `## Submit

\`\`\`c
void submit(pool, fn, arg) {
    lock(&queue.lock);
    enqueue(&queue, {fn, arg});
    signal(&queue.not_empty);
    unlock(&queue.lock);
}
\`\`\``],
		['Build worker loop', 'Process tasks', `## Worker Loop

\`\`\`c
while (running) {
    lock(&queue.lock);
    while (queue_empty()) wait(&not_empty);
    task = dequeue(&queue);
    unlock(&queue.lock);
    task.fn(task.arg);
}
\`\`\``],
		['Add graceful shutdown', 'Stop workers cleanly', `## Shutdown

1. Set shutdown flag
2. Broadcast to all waiters
3. Join all threads
4. Drain remaining tasks`],
		['Handle task results', 'Future/promise pattern', `## Futures

\`\`\`c
struct future {
    void* result;
    bool completed;
    pthread_cond_t done;
};
// submit returns future
// wait blocks until done
\`\`\``],
	]);

	addModuleWithTasks(threadPoolPath.id, 'Week 3-4: Advanced Features', 'Work stealing and priorities', 1, [
		['Implement work stealing', 'Balance load across threads', `## Work Stealing

Each thread has own deque
Steal from other threads when idle
Reduces contention on shared queue`],
		['Add priority queues', 'Task priorities', `## Priority Queue

High/medium/low priority queues
Workers check high first
Prevent priority inversion`],
		['Implement task dependencies', 'DAG scheduling', `## Dependencies

\`\`\`c
task_t* t1 = submit(fn1);
task_t* t2 = submit(fn2);
task_t* t3 = submit_after(fn3, {t1, t2});
\`\`\``],
		['Build fork-join model', 'Recursive parallelism', `## Fork-Join

\`\`\`c
result parallel_sum(arr, n) {
    if (n < THRESHOLD) return sequential_sum(arr, n);
    mid = n/2;
    future left = fork(parallel_sum, arr, mid);
    right = parallel_sum(arr+mid, n-mid);
    return join(left) + right;
}
\`\`\``],
		['Add thread affinity', 'Pin threads to CPUs', `## CPU Affinity

\`\`\`c
cpu_set_t cpuset;
CPU_SET(core_id, &cpuset);
pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
\`\`\``],
		['Implement adaptive sizing', 'Dynamic thread count', `## Adaptive Pool

Monitor queue length
Add threads when queue grows
Remove threads when idle`],
	]);
}

// ============================================================================
// Build Your Own Async Runtime
// ============================================================================
const asyncPath = pathsNeedingTasks.find(p => p.name.includes('Async Runtime'));
if (asyncPath) {
	addModuleWithTasks(asyncPath.id, 'Week 1-2: Event Loop', 'Core async foundation', 0, [
		['Understand async/await model', 'Cooperative concurrency', `## Async Model

Tasks yield at await points
Event loop schedules ready tasks
No OS threads per task`],
		['Build basic event loop', 'Task scheduling', `## Event Loop

\`\`\`rust
loop {
    // Poll ready tasks
    for task in ready_queue {
        task.poll();
    }
    // Wait for I/O events
    poll_events();
}
\`\`\``],
		['Implement Future trait', 'Pollable async unit', `## Future Trait

\`\`\`rust
trait Future {
    type Output;
    fn poll(&mut self, cx: &Context) -> Poll<Self::Output>;
}
enum Poll<T> { Ready(T), Pending }
\`\`\``],
		['Build Waker mechanism', 'Wake sleeping tasks', `## Waker

\`\`\`rust
// When I/O ready, wake task
waker.wake();
// Task re-added to ready queue
\`\`\``],
		['Create executor', 'Run futures to completion', `## Executor

\`\`\`rust
fn block_on<F: Future>(future: F) -> F::Output {
    let waker = create_waker();
    loop {
        match future.poll(&waker) {
            Poll::Ready(result) => return result,
            Poll::Pending => wait_for_wake(),
        }
    }
}
\`\`\``],
		['Add task spawning', 'Concurrent tasks', `## Spawn

\`\`\`rust
spawn(async {
    let a = fetch("url1").await;
    let b = fetch("url2").await;
    a + b
});
\`\`\``],
	]);

	addModuleWithTasks(asyncPath.id, 'Week 3-4: I/O Integration', 'Async I/O operations', 1, [
		['Integrate epoll/kqueue', 'OS event notification', `## Event Source

\`\`\`c
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
epoll_wait(epfd, events, max_events, timeout);
\`\`\``],
		['Build async TCP', 'Non-blocking sockets', `## Async TCP

\`\`\`rust
let stream = TcpStream::connect("addr").await;
stream.write(data).await;
let n = stream.read(&mut buf).await;
\`\`\``],
		['Implement async timers', 'Sleep and timeout', `## Timers

\`\`\`rust
sleep(Duration::from_secs(1)).await;
timeout(Duration::from_secs(5), slow_op()).await;
\`\`\``],
		['Add async file I/O', 'File operations', `## Async Files

Options:
- Thread pool for blocking ops
- io_uring on Linux
- IOCP on Windows`],
		['Build async channels', 'Task communication', `## Channels

\`\`\`rust
let (tx, rx) = channel();
spawn(async move { tx.send(42).await; });
let val = rx.recv().await;
\`\`\``],
		['Implement select/join', 'Concurrent operations', `## Select

\`\`\`rust
select! {
    result = future1 => handle(result),
    result = future2 => handle(result),
    _ = timeout => return Err(Timeout),
}
\`\`\``],
	]);

	addModuleWithTasks(asyncPath.id, 'Week 5-6: Multi-threading', 'Multi-threaded runtime', 2, [
		['Build work-stealing scheduler', 'Parallel task execution', `## Work Stealing

Each thread has local queue
Steal from others when empty
Good load balancing`],
		['Add thread-safe wakers', 'Cross-thread wake', `## Thread-Safe Waker

Waker must be Send + Sync
Use atomic operations
Handle thread parking`],
		['Implement task pinning', '!Send futures', `## Local Tasks

Some futures can't move threads
Pin to spawning thread
spawn_local vs spawn`],
		['Build blocking task pool', 'Run sync code', `## Blocking

\`\`\`rust
let result = spawn_blocking(|| {
    std::fs::read_to_string("file")
}).await;
\`\`\``],
		['Add graceful shutdown', 'Clean termination', `## Shutdown

1. Stop accepting new tasks
2. Cancel/drain pending tasks
3. Wait for running tasks
4. Clean up resources`],
		['Benchmark and optimize', 'Performance tuning', `## Optimization

- Minimize allocations
- Reduce lock contention
- Batch wake operations
- Profile with perf`],
	]);
}

// ============================================================================
// Terminal-based Task Scheduler
// ============================================================================
const schedulerPath = pathsNeedingTasks.find(p => p.name.includes('Task Scheduler'));
if (schedulerPath) {
	addModuleWithTasks(schedulerPath.id, 'Week 1-2: Core Scheduler', 'Job scheduling fundamentals', 0, [
		['Design job data structure', 'Define job properties', `## Job Structure

\`\`\`go
type Job struct {
    ID        string
    Name      string
    Command   string
    Schedule  string  // cron expression
    NextRun   time.Time
    LastRun   time.Time
    Status    string
}
\`\`\``],
		['Parse cron expressions', 'Standard cron format', `## Cron Parsing

Format: min hour day month weekday
* * * * * = every minute
0 */2 * * * = every 2 hours
0 0 * * 0 = weekly on Sunday`],
		['Calculate next run time', 'Schedule computation', `## Next Run

Given current time and cron expression:
1. Parse cron fields
2. Find next matching time
3. Handle edge cases (DST, etc.)`],
		['Implement job queue', 'Priority by next run', `## Job Queue

Use min-heap ordered by NextRun
Pop jobs when NextRun <= now
Re-insert with new NextRun after execution`],
		['Build execution engine', 'Run jobs', `## Execution

\`\`\`go
cmd := exec.Command("sh", "-c", job.Command)
output, err := cmd.CombinedOutput()
// Log result, update job status
\`\`\``],
		['Add job persistence', 'Save jobs to disk', `## Persistence

Store in SQLite:
- Jobs table
- Execution history
- Load on startup`],
	]);

	addModuleWithTasks(schedulerPath.id, 'Week 3-4: Features', 'Advanced scheduling', 1, [
		['Implement job dependencies', 'Run after other jobs', `## Dependencies

Job B depends on Job A:
- Wait for A to complete
- Only run B if A succeeded
- Handle circular dependencies`],
		['Add retries with backoff', 'Handle failures', `## Retries

\`\`\`go
type RetryConfig struct {
    MaxRetries int
    Backoff    time.Duration
    MaxBackoff time.Duration
}
// Exponential backoff: 1s, 2s, 4s, 8s...
\`\`\``],
		['Build job timeout', 'Kill long-running jobs', `## Timeout

\`\`\`go
ctx, cancel := context.WithTimeout(ctx, job.Timeout)
cmd := exec.CommandContext(ctx, ...)
\`\`\``],
		['Implement concurrent limits', 'Control parallelism', `## Concurrency

Global limit: max N jobs running
Per-resource limits: only 1 job using database
Use semaphores/tokens`],
		['Add notifications', 'Alert on failure', `## Notifications

On failure:
- Send email
- Slack webhook
- Custom webhook`],
		['Build TUI dashboard', 'Terminal interface', `## TUI

Show:
- Running jobs
- Next scheduled jobs
- Recent failures
- Job logs`],
	]);
}

// ============================================================================
// Simple Key-Value Store
// ============================================================================
const kvPath = pathsNeedingTasks.find(p => p.name.includes('Key-Value Store'));
if (kvPath) {
	addModuleWithTasks(kvPath.id, 'Week 1-2: In-Memory Store', 'Hash map based storage', 0, [
		['Implement basic hash map', 'Core data structure', `## Hash Map

\`\`\`go
type Store struct {
    data map[string][]byte
    mu   sync.RWMutex
}
\`\`\``],
		['Add GET/SET/DELETE', 'Basic operations', `## Operations

GET key -> value or nil
SET key value [EX seconds]
DEL key -> deleted count`],
		['Implement TTL support', 'Key expiration', `## TTL

Store expiration time with value
Background goroutine cleans expired
Lazy deletion on access`],
		['Add atomic operations', 'INCR, DECR, etc.', `## Atomic Ops

INCR key: increment integer
DECR key: decrement integer
SETNX key value: set if not exists`],
		['Build pub/sub', 'Message channels', `## Pub/Sub

SUBSCRIBE channel
PUBLISH channel message
Multiple subscribers per channel`],
		['Add data types', 'Lists, sets, hashes', `## Data Types

LIST: LPUSH, RPUSH, LPOP, RPOP
SET: SADD, SREM, SMEMBERS
HASH: HSET, HGET, HGETALL`],
	]);

	addModuleWithTasks(kvPath.id, 'Week 3-4: Persistence', 'Disk storage', 1, [
		['Implement RDB snapshots', 'Point-in-time dumps', `## RDB

Periodically dump entire dataset
Fork process, write to temp file
Rename atomically when done`],
		['Build AOF logging', 'Append-only file', `## AOF

Log every write operation
Replay on startup
Rewrite to compact`],
		['Add fsync policies', 'Durability options', `## Fsync

- always: after each write (slow, safe)
- everysec: batch per second
- no: let OS decide`],
		['Implement log compaction', 'Reduce AOF size', `## Compaction

Background rewrite:
1. Fork process
2. Write current state
3. Append buffered writes
4. Atomic replace`],
		['Build recovery', 'Load from disk', `## Recovery

On startup:
1. Check for RDB file
2. Check for AOF file
3. Prefer AOF (more recent)
4. Replay to memory`],
		['Add background saves', 'Non-blocking persistence', `## Background Save

BGSAVE: fork and save
Don't block main thread
Copy-on-write for efficiency`],
	]);
}

// ============================================================================
// Continue with remaining paths...
// ============================================================================

// Simple Message Queue
const mqPath = pathsNeedingTasks.find(p => p.name.includes('Message Queue'));
if (mqPath) {
	addModuleWithTasks(mqPath.id, 'Week 1-2: Queue Basics', 'Message handling', 0, [
		['Design message format', 'Message structure', `## Message

\`\`\`go
type Message struct {
    ID        string
    Topic     string
    Body      []byte
    Timestamp time.Time
    Headers   map[string]string
}
\`\`\``],
		['Implement producer API', 'Send messages', `## Producer

\`\`\`go
producer.Send(topic, body)
producer.SendBatch(topic, messages)
\`\`\``],
		['Build consumer API', 'Receive messages', `## Consumer

\`\`\`go
consumer.Subscribe(topic)
msg := consumer.Receive()
consumer.Ack(msg.ID)
\`\`\``],
		['Add topic management', 'Create/delete topics', `## Topics

CREATE TOPIC name
DELETE TOPIC name
LIST TOPICS`],
		['Implement message persistence', 'Durability', `## Persistence

Write messages to disk
Index by offset
Retain for configurable period`],
		['Build consumer groups', 'Parallel consumption', `## Consumer Groups

Multiple consumers share topic
Each message delivered once per group
Track offsets per group`],
	]);

	addModuleWithTasks(mqPath.id, 'Week 3-4: Advanced Features', 'Reliability and performance', 1, [
		['Add acknowledgment modes', 'Delivery guarantees', `## Ack Modes

- Auto: ack on receive
- Manual: explicit ack
- Batch: ack multiple`],
		['Implement dead letter queue', 'Failed message handling', `## DLQ

After N retries, move to DLQ
Separate topic for inspection
Manual reprocessing`],
		['Build message ordering', 'Preserve order', `## Ordering

Per-partition ordering
Partition by key
Single consumer per partition`],
		['Add batching', 'Efficient transfers', `## Batching

Batch messages for network efficiency
Configurable batch size and linger
Compress batches`],
		['Implement backpressure', 'Flow control', `## Backpressure

Producer: block when queue full
Consumer: pause when overwhelmed
Configurable high/low watermarks`],
		['Build monitoring', 'Metrics and health', `## Monitoring

- Queue depth
- Message rate
- Consumer lag
- Error rates`],
	]);
}

console.log('Done expanding more tasks!');

const finalCount = db.prepare(`
	SELECT COUNT(*) as paths,
	(SELECT COUNT(*) FROM modules) as modules,
	(SELECT COUNT(*) FROM tasks) as tasks
	FROM paths
`).get() as { paths: number; modules: number; tasks: number };

console.log(`Final counts: ${finalCount.paths} paths, ${finalCount.modules} modules, ${finalCount.tasks} tasks`);

db.close();
