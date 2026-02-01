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
		['Understand Linux namespaces', 'Learn the 7 Linux namespace types that provide process isolation: PID (separate process ID space), NET (isolated network stack), MNT (private mount points), UTS (separate hostname), IPC (isolated shared memory/semaphores), USER (UID/GID remapping), and CGROUP (resource limits). Each creates an isolated view of system resources.', `## Namespaces Overview

Types:
- PID: process ID isolation
- NET: network isolation
- MNT: mount point isolation
- UTS: hostname isolation
- IPC: inter-process communication
- USER: user ID mapping`],
		['Implement PID namespace', 'Create isolated process tree using clone() with CLONE_NEWPID flag. The first process in new namespace becomes PID 1 (init), responsible for reaping orphaned children. Processes inside cannot see or signal processes outside. Use unshare() for existing processes.', `## PID Namespace

\`\`\`c
clone(child_fn, stack, CLONE_NEWPID | SIGCHLD, NULL);
\`\`\`

Child sees itself as PID 1`],
		['Add network namespace', 'Create isolated network stack with its own interfaces, routing tables, iptables rules, and port space. Initially empty (only loopback). Connect to host via veth pairs or macvlan. Example: ip netns add myns, then ip link set eth0 netns myns.', `## Network Namespace

Create isolated network stack:
- Own interfaces
- Own routing table
- Own iptables rules`],
		['Implement mount namespace', 'Give container private filesystem view: clone with CLONE_NEWNS, then pivot_root to container rootfs (not chroot which is escapable). Mark host mounts as slave/private to prevent propagation. Bind mount specific paths (/dev, /proc) as needed.', `## Mount Namespace

New mount table
pivot_root to new root filesystem
Unmount host filesystems`],
		['Add UTS namespace', 'Isolate hostname and domain name so each container can have its own identity. Use CLONE_NEWUTS flag, then sethostname() to set container hostname. Commonly set to container ID or user-specified name. Simple but important for multi-container environments.', `## UTS Namespace

Container has own hostname:
\`\`\`c
sethostname("container", 9);
\`\`\``],
		['Implement user namespace', 'Map container UIDs to unprivileged host UIDs: container root (UID 0) maps to host user (e.g., UID 100000). Write to /proc/[pid]/uid_map and gid_map. Enables rootless containers where container root has no host privileges. Requires CLONE_NEWUSER.', `## User Namespace

Map container root to unprivileged host user:
\`\`\`
uid_map: 0 1000 1
gid_map: 0 1000 1
\`\`\``],
	]);

	addModuleWithTasks(containerPath.id, 'Week 3-4: Cgroups & Filesystem', 'Resource limits and rootfs', 1, [
		['Understand cgroups v2', 'Learn the unified cgroups v2 hierarchy at /sys/fs/cgroup for resource control. Controllers: cpu (limit CPU time via cpu.max), memory (cap RAM usage via memory.max), io (throttle disk via io.max), pids (limit process count). Enable controllers via cgroup.subtree_control.', `## Cgroups v2

Unified hierarchy at /sys/fs/cgroup
Controllers:
- cpu: CPU time limits
- memory: memory limits
- io: disk I/O limits
- pids: process count limits`],
		['Implement memory limits', 'Restrict container memory by writing to memory.max (e.g., "100M"). Set memory.swap.max to 0 to prevent swap usage. The OOM killer terminates processes when limit exceeded. Monitor usage via memory.current. Handle memory pressure events via memory.events.', `## Memory Limits

\`\`\`bash
echo 100M > /sys/fs/cgroup/container/memory.max
echo 1 > /sys/fs/cgroup/container/memory.swap.max
\`\`\``],
		['Add CPU limits', 'Control CPU time via cpu.max: format is "quota period" in microseconds. "50000 100000" = 50% of one CPU (50ms every 100ms). "200000 100000" = 2 CPUs max. Use cpu.weight (1-10000) for relative priority between containers sharing CPUs.', `## CPU Limits

\`\`\`bash
echo "100000 100000" > cpu.max  # 100% of one CPU
echo "50000 100000" > cpu.max   # 50% of one CPU
\`\`\``],
		['Build rootfs from scratch', 'Create minimal container filesystem: /bin (busybox symlinks), /dev (null, zero, random, urandom, tty), /etc (passwd, group, resolv.conf), /proc and /sys (mount points). Use busybox for most utilities. Under 5MB total for minimal container.', `## Rootfs Creation

1. Create directory structure
2. Copy busybox binary
3. Create device nodes
4. Set up /etc files`],
		['Implement overlay filesystem', 'Use OverlayFS to layer filesystem changes on read-only base: lowerdir is base image (read-only), upperdir captures writes, workdir is scratch space. Mount merged view. Enables efficient image sharing - multiple containers share same lowerdir.', `## OverlayFS

\`\`\`bash
mount -t overlay overlay \\
  -o lowerdir=base,upperdir=changes,workdir=work \\
  merged/
\`\`\``],
		['Add image layer support', 'Implement Docker-style layered images: each layer is a tarball of filesystem diff. Stack multiple lowerdirs in overlay mount (separated by colons). Download layers from registry, extract to layer directories. Copy-on-write means unchanged files share disk space.', `## Image Layers

Each layer is a tarball
Stack layers with overlayfs
Copy-on-write for changes`],
	]);

	addModuleWithTasks(containerPath.id, 'Week 5-6: Networking & CLI', 'Container networking and interface', 2, [
		['Create virtual ethernet pairs', 'Create veth pairs - virtual cables connecting network namespaces. One end (veth0) stays in host namespace, other end (veth1) moves into container namespace via "ip link set veth1 netns <container>". Assign IP addresses to each end to enable communication.', `## Veth Pairs

\`\`\`bash
ip link add veth0 type veth peer name veth1
ip link set veth1 netns container
\`\`\``],
		['Set up bridge networking', 'Create Linux bridge to connect multiple containers: "ip link add br0 type bridge", attach container veth endpoints to bridge, assign bridge an IP (gateway for containers). Enable IP forwarding and MASQUERADE for outbound traffic from containers.', `## Bridge Network

\`\`\`bash
ip link add br0 type bridge
ip link set veth0 master br0
ip addr add 10.0.0.1/24 dev br0
\`\`\``],
		['Implement port forwarding', 'Expose container ports to host using iptables NAT: DNAT rule in PREROUTING chain redirects host:8080 to container:80. Also add MASQUERADE in POSTROUTING for return traffic. Track published ports per container for cleanup on stop.', `## Port Forward

\`\`\`bash
iptables -t nat -A PREROUTING \\
  -p tcp --dport 8080 \\
  -j DNAT --to-destination 10.0.0.2:80
\`\`\``],
		['Build CLI interface', 'Create docker-like CLI: "run" creates and starts container with image and command, "ps" lists running containers with ID/name/status, "stop" sends SIGTERM then SIGKILL, "exec" runs command in existing container, "images" lists local images.', `## CLI Design

\`\`\`
mycontainer run alpine /bin/sh
mycontainer ps
mycontainer stop <id>
mycontainer images
\`\`\``],
		['Add image pull support', 'Implement Docker Registry HTTP API v2: GET /v2/library/alpine/manifests/latest returns manifest with layer digests, GET /v2/.../blobs/<digest> downloads each layer tarball. Verify SHA256 digest of downloaded content. Handle authentication tokens.', `## Image Pull

HTTP API to registry:
GET /v2/library/alpine/manifests/latest
GET /v2/library/alpine/blobs/<digest>`],
		['Implement exec command', 'Enter running container namespaces: open /proc/<pid>/ns/* files for target container, use setns() to enter each namespace (pid, net, mnt, uts, ipc). Then fork and exec the requested command. Return exit code to caller.', `## Exec

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
		['Understand ptrace system call', 'Learn ptrace - the Linux system call powering all debuggers. Key operations: PTRACE_TRACEME (child allows parent to trace it), PTRACE_ATTACH (attach to running process), PTRACE_PEEKDATA/POKEDATA (read/write memory), PTRACE_GETREGS (read CPU registers), PTRACE_SINGLESTEP (execute one instruction).', `## ptrace Overview

Operations:
- PTRACE_TRACEME: allow parent to trace
- PTRACE_ATTACH: attach to process
- PTRACE_PEEKDATA: read memory
- PTRACE_POKEDATA: write memory
- PTRACE_SINGLESTEP: execute one instruction
- PTRACE_CONT: continue execution`],
		['Implement attach/detach', 'Attach to running process: PTRACE_ATTACH sends SIGSTOP to target, then waitpid() until it stops. Now you can read memory, set breakpoints, etc. PTRACE_DETACH resumes process normally. Handle permission errors (can only attach to own processes or with CAP_SYS_PTRACE).', `## Attach

\`\`\`c
ptrace(PTRACE_ATTACH, pid, NULL, NULL);
waitpid(pid, &status, 0);  // wait for stop
// ... debug ...
ptrace(PTRACE_DETACH, pid, NULL, NULL);
\`\`\``],
		['Read process memory', 'Examine target memory with PTRACE_PEEKDATA: reads one word (8 bytes on x64) at specified address. Build read_memory(pid, addr, buf, len) by reading words and copying to buffer. Use /proc/<pid>/mem for larger reads (more efficient for bulk reads).', `## Memory Reading

\`\`\`c
long data = ptrace(PTRACE_PEEKDATA, pid, addr, NULL);
\`\`\`

Read in word-sized chunks`],
		['Write process memory', 'Modify target memory with PTRACE_POKEDATA: writes one word at address. Used to set breakpoints (write 0xCC), inject code, or patch values. Build write_memory(pid, addr, buf, len) helper. Preserve and restore original bytes when needed.', `## Memory Writing

\`\`\`c
ptrace(PTRACE_POKEDATA, pid, addr, new_value);
\`\`\`

Used for breakpoints and patches`],
		['Get/set registers', 'Read/write CPU registers: PTRACE_GETREGS fills user_regs_struct with all general-purpose registers (rax, rbx, rip, rsp, etc.). PTRACE_SETREGS writes them back. Essential for examining state at breakpoints, modifying return values, or changing instruction pointer.', `## Registers

\`\`\`c
struct user_regs_struct regs;
ptrace(PTRACE_GETREGS, pid, NULL, &regs);
printf("RIP: %llx\\n", regs.rip);
\`\`\``],
		['Implement single-stepping', 'Execute one instruction at a time: PTRACE_SINGLESTEP resumes and stops after one instruction. CPU raises SIGTRAP caught via waitpid(). Essential for stepping through code. Handle calls/returns by tracking stack depth.', `## Single Step

\`\`\`c
ptrace(PTRACE_SINGLESTEP, pid, NULL, NULL);
waitpid(pid, &status, 0);
\`\`\``],
	]);

	addModuleWithTasks(debuggerPath.id, 'Week 3-4: Breakpoints', 'Software breakpoints', 1, [
		['Understand INT3 instruction', 'Learn x86 breakpoint mechanism: the INT3 instruction (opcode 0xCC, single byte) triggers a software interrupt. CPU stops execution and raises SIGTRAP signal. Debugger catches signal and gains control. Single-byte opcode critical - can replace any instruction start.', `## INT3

Opcode 0xCC triggers SIGTRAP
CPU stops execution
Debugger regains control`],
		['Implement software breakpoints', 'Set breakpoint: save original byte at target address, write 0xCC (INT3). On hit: RIP points past the 0xCC, so decrement RIP, restore original byte, single-step to execute it, then re-write 0xCC. This cycle enables persistent breakpoints.', `## Setting Breakpoint

1. Read original byte at address
2. Write 0xCC (INT3)
3. On hit: restore original, step back, re-set`],
		['Build breakpoint manager', 'Track all breakpoints in a data structure: address, original byte value, enabled state, optional condition, hit count. Support enable/disable without removing. Handle overlapping breakpoints. Provide list/delete commands.', `## Breakpoint Manager

\`\`\`c
struct breakpoint {
    void* addr;
    uint8_t original_byte;
    bool enabled;
};
\`\`\``],
		['Handle breakpoint hits', 'Process SIGTRAP from breakpoint: 1) get RIP (points one byte past INT3), 2) find breakpoint at RIP-1, 3) restore original instruction byte, 4) set RIP back one byte, 5) single-step to execute original instruction, 6) re-enable breakpoint by writing 0xCC again.', `## Hit Handler

1. Get RIP (points past INT3)
2. Find breakpoint at RIP-1
3. Restore original byte
4. Decrement RIP
5. Single-step
6. Re-enable breakpoint`],
		['Add conditional breakpoints', 'Extend breakpoints with conditions: when hit, evaluate expression (check register value, compare memory contents). If condition is false, automatically continue execution. Useful for breaking only when variable equals specific value or loop counter reaches N.', `## Conditional

When breakpoint hits:
1. Evaluate condition (register value, memory)
2. If false, continue automatically`],
		['Implement watchpoints', 'Hardware watchpoints using x86 debug registers DR0-DR3: set address to watch, configure DR7 for read/write/execute trigger and size (1/2/4/8 bytes). CPU traps on matching access. Only 4 watchpoints available but very fast - no performance impact until triggered.', `## Watchpoints

Use hardware debug registers (DR0-DR3)
Break on read, write, or execute`],
	]);

	addModuleWithTasks(debuggerPath.id, 'Week 5-6: Symbols & UI', 'Debug information and interface', 2, [
		['Parse ELF symbol tables', 'Load symbols from ELF binaries: .symtab section contains all symbols (function names, variables), .dynsym has dynamic symbols for shared libraries. Parse ELF header to find sections, read symbol entries (name offset, address, size, type). Build address-to-name lookup table.', `## Symbol Loading

Read .symtab and .dynsym sections
Map addresses to function names
Support stripped binaries (partial)`],
		['Parse DWARF debug info', 'Parse DWARF sections for source-level debugging: .debug_info contains type definitions and variable locations, .debug_line maps addresses to source file/line numbers, .debug_frame describes how to unwind the stack. Use libdwarf or parse manually.', `## DWARF

Sections:
- .debug_info: types, variables
- .debug_line: line numbers
- .debug_frame: stack unwinding`],
		['Implement stack unwinding', 'Generate backtrace by walking the stack: with frame pointers (RBP chain), read return address at RBP+8, previous RBP at [RBP], repeat. Without frame pointers, use DWARF .debug_frame CFI (Call Frame Information). Resolve addresses to function names.', `## Stack Trace

Using frame pointers:
\`\`\`c
while (rbp != 0) {
    rip = *(rbp + 8);
    rbp = *rbp;
    print_frame(rip);
}
\`\`\``],
		['Build command-line interface', 'Create gdb-style command interface: "break" sets breakpoints by address or function name, "continue" resumes execution, "step" single-steps, "next" steps over function calls, "print" evaluates expressions, "backtrace" shows stack, "info registers" dumps CPU state.', `## Commands

- break <addr/func>
- continue
- step / next
- print <expr>
- backtrace
- info registers`],
		['Add source code display', 'Display source context at current location: use DWARF .debug_line to map RIP to filename and line number, read source file, display lines around current position with current line highlighted. Update display after each step or breakpoint hit.', `## Source Display

Using DWARF line info:
1. Find file/line for current RIP
2. Read source file
3. Display context around current line`],
		['Implement expression evaluation', 'Parse and evaluate debug expressions: variable names resolved via DWARF, register names like $rax/$rip from CPU state, pointer dereference (*ptr) reads target memory, arithmetic operators for calculations. Support struct member access (ptr->field).', `## Expressions

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
		['Understand sbrk/mmap', 'Learn how allocators get memory from the OS: sbrk(n) extends the heap contiguously (simple but limited), mmap with MAP_ANONYMOUS allocates pages anywhere in address space (flexible, can return memory). Most allocators use mmap for large allocations, manage heap for small ones.', `## System Calls

sbrk(n): extend heap by n bytes
mmap: map anonymous memory pages
Both return raw memory to manage`],
		['Implement simple bump allocator', 'Build the simplest allocator: maintain a pointer to end of used heap, allocate by returning current pointer and advancing it by requested size. No support for free() - memory never returned. Used in arenas/regions where everything freed at once. Extremely fast.', `## Bump Allocator

\`\`\`c
void* alloc(size_t size) {
    void* ptr = heap_end;
    heap_end += size;
    return ptr;
}
\`\`\`

No free support, but very fast`],
		['Build free list allocator', 'Track freed blocks in a linked list: each free block stores size and pointer to next free block. On malloc, search list for suitable block. On free, add block to list. Trade-off: supports free() but has fragmentation and O(n) allocation time.', `## Free List

\`\`\`c
struct block {
    size_t size;
    struct block* next;
};
\`\`\`

Search list for fitting block on alloc`],
		['Implement first-fit strategy', 'Simplest allocation strategy: scan free list from start, return first block >= requested size. Fast average case but causes fragmentation - small blocks accumulate at list head, large allocations scan entire list. Good baseline to improve upon.', `## First Fit

Return first block >= requested size
Fast but causes fragmentation`],
		['Add best-fit strategy', 'Reduce fragmentation by searching entire free list, selecting smallest block that fits. Minimizes wasted space per allocation but slower (always O(n) scan). Creates many small unusable fragments over time. Better for memory efficiency, worse for speed.', `## Best Fit

Search entire list
Return smallest fitting block
Less fragmentation, slower`],
		['Implement block coalescing', 'Merge adjacent free blocks to reduce fragmentation: when freeing a block, check if previous and next blocks are free, merge into single larger block. Requires tracking block boundaries (headers/footers) or maintaining sorted free list.', `## Coalescing

When freeing:
- Check if prev block is free -> merge
- Check if next block is free -> merge
- Reduces fragmentation`],
	]);

	addModuleWithTasks(allocatorPath.id, 'Week 3-4: Advanced Allocator', 'Size classes and segregated lists', 1, [
		['Design size classes', 'Eliminate internal fragmentation by rounding allocations to fixed sizes: 16, 32, 64, 128, 256, 512, 1024 bytes, etc. Any request uses next-larger class (request 20 → class 32). Within a class, no fragmentation. Trade-off: some wasted space per allocation.', `## Size Classes

Classes: 16, 32, 64, 128, 256, 512, 1024, ...
Round up requests to nearest class
Eliminates fragmentation within class`],
		['Build segregated free lists', 'Maintain separate free list per size class: lists[0] for 16-byte blocks, lists[1] for 32-byte, etc. Allocation becomes O(1): index into correct list, pop first block. Free is O(1): push onto class list. Each class is effectively a slab allocator.', `## Segregated Lists

\`\`\`c
struct free_list* lists[NUM_CLASSES];
// lists[0] for 16-byte blocks
// lists[1] for 32-byte blocks
// ...
\`\`\``],
		['Implement splitting', 'When free block is larger than needed: split into two blocks, return requested portion, add remainder to appropriate free list. Only split if remainder is large enough to be useful (≥ minimum block size). Reduces waste from size rounding.', `## Block Splitting

If block too large for request:
1. Allocate requested size
2. Put remainder in free list`],
		['Add block headers', 'Store metadata before each block: size (for free to know block size), allocated flag (is block in use), optionally size class index, magic number for debugging. Header typically 8-16 bytes. Return pointer past header to user.', `## Block Header

\`\`\`c
struct header {
    size_t size;     // block size
    bool allocated;  // in use?
    // maybe: size class, magic number
};
\`\`\``],
		['Implement boundary tags', 'Add footer at end of each block duplicating header info. Enables finding previous blocks size during coalescing without scanning from heap start. When freeing at address X, can check X-footer_size to find previous blocks metadata.', `## Boundary Tags

Header at start AND footer at end
Footer allows finding prev block's size
Enables bidirectional coalescing`],
		['Build explicit free list', 'Instead of implicit list (all blocks), only link free blocks with next/prev pointers. Free blocks reuse their payload space for these pointers. Speeds up allocation (skip allocated blocks) and still allows O(1) free by unlinking.', `## Explicit List

Free blocks contain:
- Size (header)
- Next free pointer
- Prev free pointer
- Size (footer)

Faster than searching all blocks`],
	]);

	addModuleWithTasks(allocatorPath.id, 'Week 5-6: Production Features', 'Thread safety and debugging', 2, [
		['Add thread safety', 'Handle concurrent malloc/free: simplest is global mutex (high contention), better is per-size-class locks (parallel allocations in different classes), best is thread-local caches (most operations lock-free). Modern allocators like jemalloc/tcmalloc use thread-local caching.', `## Thread Safety

Options:
1. Global mutex (simple, slow)
2. Per-size-class locks
3. Thread-local caches (tcmalloc style)`],
		['Implement thread-local caches', 'Each thread maintains private cache of free blocks per size class. malloc/free usually hit cache with no locking. When cache empty, lock central pool and refill batch. When cache full, return batch to central pool. Dramatically reduces contention.', `## Thread Local

Each thread has small cache
Most allocs/frees hit cache
Only lock when cache empty/full`],
		['Build memory pool', 'Specialized allocator for fixed-size objects: pre-allocate array of N identical blocks, maintain free list via indices. O(1) alloc/free, zero fragmentation, excellent cache locality. Used for frequently-allocated types like network packets, game entities.', `## Memory Pool

Pre-allocate N objects of same size
Very fast for specific object types
Used in games, networking`],
		['Add debugging features', 'Catch memory bugs: canary values before/after blocks detect overflows (check on free), red zones are unmapped pages between allocations (immediate SIGSEGV on overflow), record caller return address for leak reports, detect double-free via allocated flag.', `## Debug Mode

- Canaries: detect buffer overflow
- Red zones: padding between blocks
- Track allocation sites (caller address)
- Detect double-free`],
		['Implement memory statistics', 'Track allocation metrics: current bytes allocated, peak allocation (high water mark), total malloc/free counts, per-size-class statistics, fragmentation ratio (used/committed). Export via function call or signal handler for debugging production issues.', `## Statistics

Track:
- Total allocated
- Peak usage
- Allocation count
- Free count
- Fragmentation ratio`],
		['Match malloc/free interface', 'Implement standard C library interface for drop-in replacement: malloc(size), free(ptr), realloc(ptr, size) - resize preserving contents, calloc(n, size) - zero-initialized array. Use LD_PRELOAD to replace system allocator in existing programs.', `## Interface

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
		['Create worker threads', 'Spawn fixed number of worker threads at pool creation (typically num_cpus). Each thread runs worker_fn loop. Store thread handles for later join during shutdown. Workers share access to task queue via pool pointer passed as argument.', `## Worker Creation

\`\`\`c
for (int i = 0; i < num_threads; i++) {
    pthread_create(&threads[i], NULL, worker_fn, pool);
}
\`\`\``],
		['Build work queue', 'Create thread-safe queue for tasks: each task is function pointer plus argument. Use mutex for exclusive access, condition variable for workers to wait when queue empty. Circular buffer or linked list for storage. Track head/tail for O(1) enqueue/dequeue.', `## Work Queue

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
		['Implement task submission', 'Add task to queue: acquire lock, add task to queue tail, signal condition variable to wake one waiting worker, release lock. Handle queue-full case (block caller, grow queue, or reject). May return future for result retrieval.', `## Submit

\`\`\`c
void submit(pool, fn, arg) {
    lock(&queue.lock);
    enqueue(&queue, {fn, arg});
    signal(&queue.not_empty);
    unlock(&queue.lock);
}
\`\`\``],
		['Build worker loop', 'Worker thread main loop: acquire lock, wait on condition variable while queue empty (releases lock during wait), dequeue task when available, release lock, execute task function with argument. Loop until shutdown flag set.', `## Worker Loop

\`\`\`c
while (running) {
    lock(&queue.lock);
    while (queue_empty()) wait(&not_empty);
    task = dequeue(&queue);
    unlock(&queue.lock);
    task.fn(task.arg);
}
\`\`\``],
		['Add graceful shutdown', 'Clean termination sequence: set shutdown flag, broadcast to all workers waiting on condition variable (wakes them to check flag), call pthread_join on each worker thread. Optionally drain remaining tasks or cancel them.', `## Shutdown

1. Set shutdown flag
2. Broadcast to all waiters
3. Join all threads
4. Drain remaining tasks`],
		['Handle task results', 'Implement future/promise pattern: submit() returns future object, worker stores result in future when done and signals condition variable. Caller waits on future to block until result ready. Add timeout support for wait.', `## Futures

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
		['Implement work stealing', 'Give each worker its own deque (double-ended queue): workers push/pop from their end (LIFO, cache-friendly), steal from other workers opposite end (FIFO) when idle. Reduces contention vs. single shared queue. Used by Cilk, TBB, Tokio.', `## Work Stealing

Each thread has own deque
Steal from other threads when idle
Reduces contention on shared queue`],
		['Add priority queues', 'Support task priorities: maintain separate queues for high/medium/low priority. Workers always check high-priority queue first. Prevent priority inversion by boosting priority of tasks blocking high-priority work. Common priorities: immediate, normal, background.', `## Priority Queue

High/medium/low priority queues
Workers check high first
Prevent priority inversion`],
		['Implement task dependencies', 'Allow tasks to depend on other tasks completion: submit_after(fn, deps) only becomes runnable when all deps complete. Build dependency graph (DAG), topologically sort for execution order. Decrement dependency counter as prerequisites complete.', `## Dependencies

\`\`\`c
task_t* t1 = submit(fn1);
task_t* t2 = submit(fn2);
task_t* t3 = submit_after(fn3, {t1, t2});
\`\`\``],
		['Build fork-join model', 'Support recursive parallelism: fork() spawns child task and returns future, join() waits for result. Classic parallel algorithms (merge sort, tree traversal) naturally express as fork-join. Worker executing join() can steal work while waiting, avoiding deadlock.', `## Fork-Join

\`\`\`c
result parallel_sum(arr, n) {
    if (n < THRESHOLD) return sequential_sum(arr, n);
    mid = n/2;
    future left = fork(parallel_sum, arr, mid);
    right = parallel_sum(arr+mid, n-mid);
    return join(left) + right;
}
\`\`\``],
		['Add thread affinity', 'Pin worker threads to specific CPU cores using pthread_setaffinity_np. Improves cache utilization by keeping thread on same core. Can dedicate cores to pool, avoiding interference from other processes. Useful for latency-sensitive applications.', `## CPU Affinity

\`\`\`c
cpu_set_t cpuset;
CPU_SET(core_id, &cpuset);
pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
\`\`\``],
		['Implement adaptive sizing', 'Dynamically adjust worker count based on load: monitor queue depth and worker utilization, spawn new workers when queue grows faster than processing, terminate idle workers after timeout. Bounds: min_threads to max_threads. Handles bursty workloads efficiently.', `## Adaptive Pool

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
		['Understand async/await model', 'Learn cooperative concurrency: async functions can pause at await points, allowing other tasks to run on same thread. Event loop schedules ready tasks, polls pending I/O. Thousands of concurrent tasks on single thread without OS thread overhead.', `## Async Model

Tasks yield at await points
Event loop schedules ready tasks
No OS threads per task`],
		['Build basic event loop', 'Core loop: poll all ready tasks (call their poll method), if any return Pending, add to wait list. Then block on epoll/kqueue waiting for I/O events. When events arrive, wake corresponding tasks and repeat. Single thread can handle many connections.', `## Event Loop

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
		['Implement Future trait', 'Define the core abstraction: Future has poll() method returning Ready(value) when complete or Pending when blocked. Compiler transforms async fn into state machine implementing Future. Each await point becomes a state. Context provides Waker for notifications.', `## Future Trait

\`\`\`rust
trait Future {
    type Output;
    fn poll(&mut self, cx: &Context) -> Poll<Self::Output>;
}
enum Poll<T> { Ready(T), Pending }
\`\`\``],
		['Build Waker mechanism', 'Enable futures to notify when ready: Waker is handle passed to poll(), stored by I/O resources. When I/O completes, resource calls waker.wake() which adds task back to ready queue. Waker must be thread-safe (Send+Sync) for cross-thread I/O completion.', `## Waker

\`\`\`rust
// When I/O ready, wake task
waker.wake();
// Task re-added to ready queue
\`\`\``],
		['Create executor', 'Run futures to completion: block_on(future) creates waker, repeatedly polls future until Ready. On Pending, executor parks thread until woken. Spawn creates new top-level task. Handle task lifecycle: creation, polling, completion, cleanup.', `## Executor

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
		['Add task spawning', 'Create concurrent tasks: spawn(async { ... }) wraps future in Task struct with unique ID and waker, adds to ready queue, returns JoinHandle for awaiting result. Tasks run independently, may complete in any order. Handle panics gracefully.', `## Spawn

\`\`\`rust
spawn(async {
    let a = fetch("url1").await;
    let b = fetch("url2").await;
    a + b
});
\`\`\``],
	]);

	addModuleWithTasks(asyncPath.id, 'Week 3-4: I/O Integration', 'Async I/O operations', 1, [
		['Integrate epoll/kqueue', 'Register file descriptors with OS event system: epoll_ctl adds fd with interest (EPOLLIN/EPOLLOUT), epoll_wait blocks until events ready. Store waker per fd, call waker.wake() when event fires. On macOS use kqueue, Windows use IOCP.', `## Event Source

\`\`\`c
epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &event);
epoll_wait(epfd, events, max_events, timeout);
\`\`\``],
		['Build async TCP', 'Non-blocking socket wrapper: set O_NONBLOCK on socket, read/write return EAGAIN when would block. poll() checks socket readiness via epoll, returns Pending if not ready (stores waker), Ready(bytes) when data available. Implement AsyncRead/AsyncWrite traits.', `## Async TCP

\`\`\`rust
let stream = TcpStream::connect("addr").await;
stream.write(data).await;
let n = stream.read(&mut buf).await;
\`\`\``],
		['Implement async timers', 'Timer wheel or timerfd: track expiration times, return Pending until deadline. Use timerfd_create on Linux for kernel-managed timers. timeout() wraps future, cancels if deadline exceeded. sleep() pauses current task without blocking thread.', `## Timers

\`\`\`rust
sleep(Duration::from_secs(1)).await;
timeout(Duration::from_secs(5), slow_op()).await;
\`\`\``],
		['Add async file I/O', 'File I/O is inherently blocking on most systems. Options: spawn_blocking() runs on thread pool (simple, portable), io_uring on Linux 5.1+ provides true async file I/O, IOCP on Windows. Abstract behind same async interface.', `## Async Files

Options:
- Thread pool for blocking ops
- io_uring on Linux
- IOCP on Windows`],
		['Build async channels', 'MPSC channels for task communication: sender.send(value).await blocks if channel full, receiver.recv().await blocks if empty. Internal queue with wakers for blocked senders/receivers. Handle channel close gracefully. Bounded and unbounded variants.', `## Channels

\`\`\`rust
let (tx, rx) = channel();
spawn(async move { tx.send(42).await; });
let val = rx.recv().await;
\`\`\``],
		['Implement select/join', 'Concurrency combinators: select! polls multiple futures, returns first to complete (cancels others). join! polls all futures, returns when all complete. try_join! returns on first error. Implemented via macro generating combined Future.', `## Select

\`\`\`rust
select! {
    result = future1 => handle(result),
    result = future2 => handle(result),
    _ = timeout => return Err(Timeout),
}
\`\`\``],
	]);

	addModuleWithTasks(asyncPath.id, 'Week 5-6: Multi-threading', 'Multi-threaded runtime', 2, [
		['Build work-stealing scheduler', 'Multi-threaded executor: each worker thread has local task queue, steals from others when idle. Tasks can run on any thread (must be Send). Global injector queue for newly spawned tasks. Balance load automatically across cores.', `## Work Stealing

Each thread has local queue
Steal from others when empty
Good load balancing`],
		['Add thread-safe wakers', 'Waker called from I/O threads must safely wake tasks on worker threads: use atomic reference counting, thread-safe ready queue with mutex or lock-free structure. Handle case where waker outlives task. Implement RawWaker vtable.', `## Thread-Safe Waker

Waker must be Send + Sync
Use atomic operations
Handle thread parking`],
		['Implement task pinning', 'Some futures hold thread-local data (!Send): spawn_local() pins task to current thread, never migrates. LocalSet runs local tasks on dedicated thread. Error at compile time if !Send future passed to spawn(). Common with Rc, thread-local storage.', `## Local Tasks

Some futures can't move threads
Pin to spawning thread
spawn_local vs spawn`],
		['Build blocking task pool', 'Run synchronous/blocking code without blocking runtime: spawn_blocking() runs closure on dedicated thread pool (separate from async workers). Returns future for result. Useful for file I/O, CPU-heavy work, calling non-async libraries.', `## Blocking

\`\`\`rust
let result = spawn_blocking(|| {
    std::fs::read_to_string("file")
}).await;
\`\`\``],
		['Add graceful shutdown', 'Clean termination: shutdown signal stops accepting new tasks, optionally cancel pending tasks (drop futures) or let them complete. Wait for in-progress tasks with timeout. Close I/O resources, join worker threads. Handle Ctrl+C gracefully.', `## Shutdown

1. Stop accepting new tasks
2. Cancel/drain pending tasks
3. Wait for running tasks
4. Clean up resources`],
		['Benchmark and optimize', 'Performance tuning: reduce allocations (reuse buffers, small future sizes), minimize lock contention (finer-grained locks, lock-free structures), batch wakeups, use perf/flamegraph to find hotspots. Compare against tokio/async-std benchmarks.', `## Optimization

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
		['Design job data structure', 'Define job properties: unique ID, human-readable name, shell command to execute, cron schedule expression, next/last run timestamps, current status (pending/running/failed/succeeded), optional timeout and retry settings.', `## Job Structure

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
		['Parse cron expressions', 'Parse standard 5-field cron format: minute (0-59), hour (0-23), day of month (1-31), month (1-12), day of week (0-6). Support wildcards (*), ranges (1-5), steps (*/15), and lists (1,3,5). Library option: github.com/robfig/cron.', `## Cron Parsing

Format: min hour day month weekday
* * * * * = every minute
0 */2 * * * = every 2 hours
0 0 * * 0 = weekly on Sunday`],
		['Calculate next run time', 'Given current time and parsed cron expression, find next matching time: iterate forward from now, check each field matches cron spec. Handle edge cases: daylight saving time transitions, month-end dates (Feb 30), timezone differences.', `## Next Run

Given current time and cron expression:
1. Parse cron fields
2. Find next matching time
3. Handle edge cases (DST, etc.)`],
		['Implement job queue', 'Priority queue (min-heap) ordered by NextRun time: peek to find next job, pop when NextRun <= now. After execution, calculate new NextRun from cron expression and re-insert into heap. Sleep until next job due or new job added.', `## Job Queue

Use min-heap ordered by NextRun
Pop jobs when NextRun <= now
Re-insert with new NextRun after execution`],
		['Build execution engine', 'Execute jobs via shell: spawn child process with exec.Command, capture stdout/stderr combined. Set environment variables, working directory. Enforce timeout via context.WithTimeout. Record exit code, output, duration in execution history.', `## Execution

\`\`\`go
cmd := exec.Command("sh", "-c", job.Command)
output, err := cmd.CombinedOutput()
// Log result, update job status
\`\`\``],
		['Add job persistence', 'Store scheduler state in SQLite: jobs table (definition and schedule), executions table (history with timestamp, duration, exit code, output). Load all jobs on startup, rebuild priority queue. Handle schema migrations.', `## Persistence

Store in SQLite:
- Jobs table
- Execution history
- Load on startup`],
	]);

	addModuleWithTasks(schedulerPath.id, 'Week 3-4: Features', 'Advanced scheduling', 1, [
		['Implement job dependencies', 'Support job dependencies: job B runs only after job A succeeds. Build dependency graph, validate no cycles on creation. When A completes successfully, check if B dependencies satisfied, schedule if ready. Failed dependencies fail dependents.', `## Dependencies

Job B depends on Job A:
- Wait for A to complete
- Only run B if A succeeded
- Handle circular dependencies`],
		['Add retries with backoff', 'Retry failed jobs: configure max attempts and backoff strategy. Exponential backoff doubles delay each retry (1s, 2s, 4s, 8s) up to max. Add jitter to prevent thundering herd. Track retry count, distinguish retriable vs permanent failures.', `## Retries

\`\`\`go
type RetryConfig struct {
    MaxRetries int
    Backoff    time.Duration
    MaxBackoff time.Duration
}
// Exponential backoff: 1s, 2s, 4s, 8s...
\`\`\``],
		['Build job timeout', 'Kill jobs exceeding time limit: use context.WithTimeout, command inherits context cancellation. On timeout, send SIGTERM then SIGKILL after grace period. Record as timeout failure, different from execution failure. Prevent runaway jobs.', `## Timeout

\`\`\`go
ctx, cancel := context.WithTimeout(ctx, job.Timeout)
cmd := exec.CommandContext(ctx, ...)
\`\`\``],
		['Implement concurrent limits', 'Control parallelism with semaphores: global limit on total running jobs, per-resource limits (one job accessing database at a time). Acquire token before execution, release after. Queue jobs waiting for tokens by priority.', `## Concurrency

Global limit: max N jobs running
Per-resource limits: only 1 job using database
Use semaphores/tokens`],
		['Add notifications', 'Alert on job events: send email, Slack webhook, or custom HTTP callback on failure, success, or long runtime. Configure per-job notification rules. Include job name, output, duration in notification payload. Rate-limit to avoid spam.', `## Notifications

On failure:
- Send email
- Slack webhook
- Custom webhook`],
		['Build TUI dashboard', 'Terminal interface with tview/bubbletea: show running jobs with progress, next scheduled jobs with countdown, recent execution history color-coded by status. View job logs, trigger manual runs, pause/resume jobs. Auto-refresh.', `## TUI

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
		['Implement basic hash map', 'Core data structure: Go map with RWMutex for concurrent access. RLock for reads (multiple readers), Lock for writes (exclusive). Store values as byte slices to support any data. Consider sharded maps for better concurrency at scale.', `## Hash Map

\`\`\`go
type Store struct {
    data map[string][]byte
    mu   sync.RWMutex
}
\`\`\``],
		['Add GET/SET/DELETE', 'Implement Redis-like basic operations: GET returns value or nil if missing, SET stores value with optional EX for expiration in seconds, DEL removes key and returns count deleted. Handle binary-safe values and UTF-8 keys.', `## Operations

GET key -> value or nil
SET key value [EX seconds]
DEL key -> deleted count`],
		['Implement TTL support', 'Key expiration: store expiry timestamp alongside value. Two cleanup strategies: lazy deletion (check on access, remove if expired) and active expiration (background goroutine periodically scans and removes expired keys). Combine both.', `## TTL

Store expiration time with value
Background goroutine cleans expired
Lazy deletion on access`],
		['Add atomic operations', 'Thread-safe operations: INCR/DECR atomically increment/decrement integer value (parse string as int, modify, store). SETNX (set if not exists) for locks/leases. GETSET returns old value while setting new. All under single lock.', `## Atomic Ops

INCR key: increment integer
DECR key: decrement integer
SETNX key value: set if not exists`],
		['Build pub/sub', 'Publish/subscribe messaging: SUBSCRIBE adds channel to connection subscriber list, PUBLISH sends message to all subscribers of channel. Maintain map of channels to subscriber connections. Support pattern subscriptions (PSUBSCRIBE foo*).', `## Pub/Sub

SUBSCRIBE channel
PUBLISH channel message
Multiple subscribers per channel`],
		['Add data types', 'Redis-style compound types: LIST (doubly-linked list for LPUSH/RPUSH/LPOP/RPOP), SET (hash set for SADD/SREM/SMEMBERS/SINTER), HASH (nested hash map for HSET/HGET/HGETALL). Type-check on access, error if wrong type.', `## Data Types

LIST: LPUSH, RPUSH, LPOP, RPOP
SET: SADD, SREM, SMEMBERS
HASH: HSET, HGET, HGETALL`],
	]);

	addModuleWithTasks(kvPath.id, 'Week 3-4: Persistence', 'Disk storage', 1, [
		['Implement RDB snapshots', 'Point-in-time dump of entire dataset: fork process (copy-on-write memory sharing), child writes all key-values to temp file in binary format, atomically rename to final name when complete. Parent continues serving requests. Compact format.', `## RDB

Periodically dump entire dataset
Fork process, write to temp file
Rename atomically when done`],
		['Build AOF logging', 'Append-only file: log every write command (SET, DEL, etc.) as it executes. On restart, replay entire log to rebuild state. Append-only is crash-safe (never overwrites). File grows unbounded until compacted.', `## AOF

Log every write operation
Replay on startup
Rewrite to compact`],
		['Add fsync policies', 'Control durability/performance trade-off: "always" fsyncs after every write (safest, slowest), "everysec" batches fsyncs per second (good balance), "no" lets OS decide (fastest, data loss on crash). Configurable per deployment.', `## Fsync

- always: after each write (slow, safe)
- everysec: batch per second
- no: let OS decide`],
		['Implement log compaction', 'Rewrite AOF to remove redundancy: fork child, write current dataset state (equivalent to minimal commands), parent buffers new writes. When child done, append buffer and atomically replace old AOF. Dramatically smaller file.', `## Compaction

Background rewrite:
1. Fork process
2. Write current state
3. Append buffered writes
4. Atomic replace`],
		['Build recovery', 'Startup sequence: check for both RDB and AOF files. If both exist, prefer AOF (more recent data). Parse file format, replay commands/load data into memory. Validate checksums if present. Report progress for large datasets.', `## Recovery

On startup:
1. Check for RDB file
2. Check for AOF file
3. Prefer AOF (more recent)
4. Replay to memory`],
		['Add background saves', 'Non-blocking persistence: BGSAVE forks child to write RDB while parent continues serving. Use COW (copy-on-write) so forked child sees consistent snapshot. Track save progress, prevent concurrent BGSAVEs. Return immediately to caller.', `## Background Save

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
		['Design message format', 'Define message structure: unique ID (UUID or sequence), topic name, body as binary blob, timestamp for ordering/debugging, headers map for metadata (content-type, correlation-id, etc.). Serialize efficiently with protobuf or msgpack.', `## Message

\`\`\`go
type Message struct {
    ID        string
    Topic     string
    Body      []byte
    Timestamp time.Time
    Headers   map[string]string
}
\`\`\``],
		['Implement producer API', 'Client API for sending messages: Send(topic, body) publishes single message synchronously, SendBatch for multiple messages efficiently. Handle connection pooling, retries on transient failures. Return message ID or error.', `## Producer

\`\`\`go
producer.Send(topic, body)
producer.SendBatch(topic, messages)
\`\`\``],
		['Build consumer API', 'Client API for receiving: Subscribe(topic) registers interest, Receive() blocks until message available (or timeout), Ack(id) confirms processing. Support manual ack (at-least-once) or auto-ack (at-most-once). Handle redelivery of unacked messages.', `## Consumer

\`\`\`go
consumer.Subscribe(topic)
msg := consumer.Receive()
consumer.Ack(msg.ID)
\`\`\``],
		['Add topic management', 'Administrative operations: CREATE TOPIC with partition count and retention settings, DELETE TOPIC (must be empty or force), LIST TOPICS shows all topics with stats. Store topic metadata persistently. Auto-create topics on first publish optionally.', `## Topics

CREATE TOPIC name
DELETE TOPIC name
LIST TOPICS`],
		['Implement message persistence', 'Durable storage: append messages to log file per partition, index by offset for fast seeking. Segment files for efficient cleanup (delete old segments). Retain messages for configurable period (7 days) or size limit (10GB). Memory-map for fast reads.', `## Persistence

Write messages to disk
Index by offset
Retain for configurable period`],
		['Build consumer groups', 'Parallel consumption across multiple consumers: consumers in same group share partitions (each message delivered once per group). Track committed offset per partition per group. Rebalance partitions when consumers join/leave.', `## Consumer Groups

Multiple consumers share topic
Each message delivered once per group
Track offsets per group`],
	]);

	addModuleWithTasks(mqPath.id, 'Week 3-4: Advanced Features', 'Reliability and performance', 1, [
		['Add acknowledgment modes', 'Delivery guarantee options: auto-ack on receive (at-most-once, fast but may lose messages), manual ack after processing (at-least-once, may see duplicates), batch ack for efficiency. Nack to reject and requeue failed messages.', `## Ack Modes

- Auto: ack on receive
- Manual: explicit ack
- Batch: ack multiple`],
		['Implement dead letter queue', 'Handle poison messages: after N failed processing attempts (caught exceptions, timeouts), move message to dead letter topic instead of infinite retry. Separate DLQ for inspection, alerting, manual reprocessing. Track original topic and failure reason.', `## DLQ

After N retries, move to DLQ
Separate topic for inspection
Manual reprocessing`],
		['Build message ordering', 'Preserve message order within partition: messages with same partition key always go to same partition, consumed by one consumer (no parallel processing). Ordering guaranteed within partition, not across partitions. Key-based routing.', `## Ordering

Per-partition ordering
Partition by key
Single consumer per partition`],
		['Add batching', 'Improve throughput by batching: producer batches messages until size limit (16KB) or time limit (100ms linger), sends as single network request. Apply compression (gzip, snappy, lz4) to batch. Consumer fetches batches for efficiency.', `## Batching

Batch messages for network efficiency
Configurable batch size and linger
Compress batches`],
		['Implement backpressure', 'Flow control when overwhelmed: producer blocks or fails when queue reaches high watermark, resumes at low watermark. Consumer can pause fetching when local buffer full. Prevents memory exhaustion. Configurable watermark thresholds.', `## Backpressure

Producer: block when queue full
Consumer: pause when overwhelmed
Configurable high/low watermarks`],
		['Build monitoring', 'Export metrics for observability: queue depth (messages pending), message rate (in/out per second), consumer lag (how far behind real-time), error rates, producer/consumer connection counts. Prometheus format, health check endpoints.', `## Monitoring

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
