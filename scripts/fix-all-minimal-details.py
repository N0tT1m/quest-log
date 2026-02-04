#!/usr/bin/env python3
"""Generate comprehensive details for all tasks with minimal content."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "quest-log.db"


def generate_details(task: dict) -> str:
    """Generate comprehensive markdown details for a task."""
    title = task["title"].lower()
    desc = (task["description"] or "").lower()
    path = task["path_name"].lower()

    details = []

    # Overview section based on context
    details.append("## Overview")

    # === BUILD YOUR OWN paths ===
    if "shell" in path and "build" in path:
        if "path" in title:
            details.append("Implement PATH environment variable searching to locate executables.")
        elif "pipe" in title:
            details.append("Connect multiple processes using pipes, where stdout of one becomes stdin of the next.")
        elif "redirect" in title:
            details.append("Implement file redirection using dup2() to replace stdin/stdout/stderr with file descriptors.")
        elif "environment" in title:
            details.append("Manage shell environment variables that are inherited by child processes.")
        elif "background" in title:
            details.append("Run processes in the background without blocking the shell prompt.")
        elif "here" in title:
            details.append("Implement here documents for multi-line input to commands.")
        elif "job" in title:
            details.append("Track and manage background processes with job control.")
        elif "signal" in title:
            details.append("Handle signals like SIGINT (Ctrl+C) and SIGTSTP (Ctrl+Z) properly.")
        elif "glob" in title or "wildcard" in title:
            details.append("Expand wildcards like *, ?, and [...] to match filenames.")
        elif "history" in title:
            details.append("Maintain command history with up/down arrow navigation and history expansion.")
        elif "completion" in title or "tab" in title:
            details.append("Implement tab completion for commands, files, and arguments.")
        else:
            details.append(f"Core shell functionality: {task['description']}")

    elif "sqlite" in path and "build" in path:
        if "select" in title or "parser" in title:
            details.append("Parse SQL SELECT statements into an abstract syntax tree for query execution.")
        elif "create" in title:
            details.append("Parse CREATE TABLE statements to define table schemas with columns and constraints.")
        elif "b-tree" in title or "btree" in title:
            details.append("Implement B-tree operations for efficient key-value storage and retrieval.")
        elif "transaction" in title:
            details.append("Implement ACID transactions with BEGIN, COMMIT, and ROLLBACK support.")
        elif "lock" in title:
            details.append("Implement database locking to handle concurrent read/write access safely.")
        elif "repl" in title:
            details.append("Build an interactive command-line interface for executing SQL queries.")
        elif "index" in title:
            details.append("Implement secondary indexes using B-trees for faster query lookups.")
        elif "free list" in title:
            details.append("Track and reuse deleted pages to prevent database file bloat.")
        elif "row" in title or "storage" in title:
            details.append("Serialize and deserialize table rows with variable-length column encoding.")
        else:
            details.append(f"SQLite implementation: {task['description']}")

    elif "debugger" in path and "build" in path:
        if "single" in title or "step" in title:
            details.append("Implement single-step execution using PTRACE_SINGLESTEP to execute one instruction at a time.")
        elif "int3" in title or "breakpoint" in title:
            details.append("Use INT3 (0xCC) instruction to create software breakpoints that trap execution.")
        elif "watchpoint" in title:
            details.append("Use hardware debug registers (DR0-DR7) to break on memory read/write access.")
        elif "symbol" in title:
            details.append("Parse DWARF debug information to map addresses to source code locations.")
        elif "backtrace" in title or "stack" in title:
            details.append("Walk the call stack using frame pointers to show function call history.")
        elif "register" in title:
            details.append("Read and modify CPU registers using ptrace PEEKUSER/POKEUSER.")
        elif "memory" in title:
            details.append("Read and write debuggee memory using ptrace PEEKDATA/POKEDATA.")
        else:
            details.append(f"Debugger functionality: {task['description']}")

    elif "async" in path and "runtime" in path:
        if "waker" in title:
            details.append("Implement the Waker mechanism that notifies the executor when a task is ready to make progress.")
        elif "async/await" in title or "model" in title:
            details.append("Understand how async/await transforms code into state machines that yield at await points.")
        elif "file" in title or "i/o" in title:
            details.append("Implement non-blocking file I/O using thread pools or io_uring for true async file operations.")
        elif "work-stealing" in title or "scheduler" in title:
            details.append("Build a work-stealing scheduler where idle threads steal tasks from busy threads' queues.")
        elif "thread" in title:
            details.append("Make wakers thread-safe using atomic operations for cross-thread task waking.")
        elif "pin" in title:
            details.append("Handle !Unpin futures that must not be moved in memory after first poll.")
        else:
            details.append(f"Async runtime: {task['description']}")

    elif "container" in path and "runtime" in path:
        if "mount" in title:
            details.append("Create isolated filesystem view using mount namespace and pivot_root.")
        elif "uts" in title:
            details.append("Isolate hostname and domain name using UTS namespace.")
        elif "pid" in title:
            details.append("Create isolated process ID space where container's init is PID 1.")
        elif "network" in title or "net" in title:
            details.append("Create isolated network stack with virtual ethernet pairs and bridges.")
        elif "cgroup" in title:
            details.append("Limit container resources (CPU, memory, I/O) using cgroups v2.")
        elif "image" in title or "layer" in title:
            details.append("Stack filesystem layers using overlayfs for copy-on-write image support.")
        elif "seccomp" in title:
            details.append("Restrict system calls available to container using seccomp-bpf filters.")
        else:
            details.append(f"Container runtime: {task['description']}")

    elif "memory" in path and "allocator" in path:
        if "first" in title:
            details.append("Implement first-fit allocation that returns the first free block large enough.")
        elif "best" in title:
            details.append("Implement best-fit allocation that finds the smallest adequate free block.")
        elif "buddy" in title:
            details.append("Implement buddy allocation that splits and coalesces power-of-2 sized blocks.")
        elif "slab" in title:
            details.append("Implement slab allocation for efficient fixed-size object allocation.")
        elif "coalesce" in title or "merge" in title:
            details.append("Merge adjacent free blocks to reduce fragmentation.")
        elif "free list" in title:
            details.append("Maintain a linked list of free memory blocks for allocation.")
        else:
            details.append(f"Memory allocator: {task['description']}")

    # === REIMPLEMENT paths ===
    elif "reimplement" in path:
        tool_name = path.replace("reimplement:", "").replace("reimplement", "").strip()

        if "mimikatz" in path:
            if "sam" in title:
                details.append("Extract password hashes from the SAM database using registry or in-memory techniques.")
            elif "lsass" in title:
                details.append("Dump credentials from LSASS process memory using MiniDumpWriteDump or direct memory access.")
            elif "dcsync" in title:
                details.append("Replicate Active Directory credentials using Directory Replication Service (DRS) protocol.")
            elif "kerberos" in title or "ticket" in title:
                details.append("Extract and manipulate Kerberos tickets from memory for pass-the-ticket attacks.")
            elif "dpapi" in title:
                details.append("Decrypt DPAPI-protected secrets using master keys from memory or backup.")
            else:
                details.append(f"Mimikatz credential extraction: {task['description']}")

        elif "bloodhound" in path:
            if "ldap" in title:
                details.append("Query Active Directory via LDAP to enumerate users, groups, and computers.")
            elif "session" in title:
                details.append("Enumerate logged-in users and sessions across domain computers.")
            elif "acl" in title:
                details.append("Collect and analyze AD object permissions to find attack paths.")
            elif "trust" in title:
                details.append("Map domain and forest trust relationships for lateral movement paths.")
            elif "graph" in title:
                details.append("Build attack path graph from collected data for visualization.")
            else:
                details.append(f"BloodHound collection: {task['description']}")

        elif "rubeus" in path:
            if "asrep" in title:
                details.append("Request AS-REP for accounts without pre-authentication for offline cracking.")
            elif "kerberoast" in title:
                details.append("Request service tickets for SPNs to crack service account passwords offline.")
            elif "s4u" in title:
                details.append("Abuse S4U2Self and S4U2Proxy for Kerberos delegation attacks.")
            elif "ticket" in title:
                details.append("Forge or manipulate Kerberos tickets for persistence and lateral movement.")
            elif "ptt" in title or "pass" in title:
                details.append("Inject Kerberos tickets into memory for pass-the-ticket attacks.")
            else:
                details.append(f"Kerberos attack tool: {task['description']}")

        elif "responder" in path:
            if "llmnr" in title:
                details.append("Spoof LLMNR responses to capture NTLMv2 hashes from name resolution requests.")
            elif "nbt" in title or "netbios" in title:
                details.append("Spoof NBT-NS responses to poison NetBIOS name resolution.")
            elif "mdns" in title:
                details.append("Spoof mDNS responses for multicast DNS poisoning attacks.")
            elif "wpad" in title:
                details.append("Serve malicious WPAD configuration to intercept HTTP proxy traffic.")
            elif "smb" in title:
                details.append("Capture SMB authentication attempts and relay or crack the hashes.")
            else:
                details.append(f"Network poisoning: {task['description']}")

        elif "nmap" in path:
            if "syn" in title:
                details.append("Implement SYN scan that sends SYN packets and analyzes responses without completing handshake.")
            elif "connect" in title:
                details.append("Implement full TCP connect scan using standard socket connections.")
            elif "udp" in title:
                details.append("Implement UDP scan with ICMP unreachable detection for closed ports.")
            elif "service" in title or "banner" in title:
                details.append("Detect services by grabbing banners and matching against signature database.")
            elif "os" in title:
                details.append("Fingerprint operating systems using TCP/IP stack behavior analysis.")
            elif "script" in title or "nse" in title:
                details.append("Implement scripting engine for extensible vulnerability detection.")
            else:
                details.append(f"Network scanner: {task['description']}")

        elif "hashcat" in path:
            if "dictionary" in title:
                details.append("Implement dictionary attack mode that tests passwords from wordlists.")
            elif "rule" in title:
                details.append("Implement rule engine for password mutations (l33t, capitalize, append numbers).")
            elif "mask" in title or "brute" in title:
                details.append("Implement mask attack for targeted brute-force with character sets.")
            elif "gpu" in title or "cuda" in title or "opencl" in title:
                details.append("Accelerate hash computation using GPU parallel processing.")
            elif "hash" in title:
                details.append("Implement hash type detection and parsing for various formats.")
            else:
                details.append(f"Password cracking: {task['description']}")

        elif "chisel" in path or "pivot" in path:
            if "socks" in title:
                details.append("Implement SOCKS5 proxy server for tunneling arbitrary TCP traffic.")
            elif "tunnel" in title:
                details.append("Create encrypted tunnels over HTTP/WebSocket for firewall bypass.")
            elif "port forward" in title:
                details.append("Forward local/remote ports through the tunnel for service access.")
            elif "reverse" in title:
                details.append("Implement reverse tunnel where server connects back to client.")
            else:
                details.append(f"Tunneling/pivoting: {task['description']}")

        elif "sliver" in path or "c2" in path:
            if "implant" in title:
                details.append("Build the agent/implant that runs on compromised systems and executes commands.")
            elif "listener" in title:
                details.append("Implement C2 listener that accepts implant connections and manages sessions.")
            elif "beacon" in title:
                details.append("Implement beaconing with jitter for periodic check-in with the C2 server.")
            elif "task" in title or "command" in title:
                details.append("Implement command tasking system for queuing and executing operator commands.")
            elif "encrypt" in title:
                details.append("Implement encrypted communications between implant and C2 server.")
            elif "http" in title:
                details.append("Implement HTTP/HTTPS transport with malleable profiles for traffic blending.")
            elif "dns" in title:
                details.append("Implement DNS-based C2 channel using TXT/A records for covert communication.")
            else:
                details.append(f"C2 framework: {task['description']}")

        elif "burp" in path or "proxy" in path:
            if "intercept" in title:
                details.append("Implement request/response interception for manual inspection and modification.")
            elif "history" in title:
                details.append("Log all proxied traffic with search and filter capabilities.")
            elif "repeat" in title:
                details.append("Implement request repeater for manual testing with modifications.")
            elif "intruder" in title or "fuzz" in title:
                details.append("Implement automated fuzzing with payload positions and attack types.")
            elif "ssl" in title or "tls" in title or "cert" in title:
                details.append("Generate and install CA certificate for HTTPS interception.")
            elif "websocket" in title:
                details.append("Handle WebSocket connections with message interception.")
            else:
                details.append(f"Web proxy: {task['description']}")

        elif "aircrack" in path or "wifi" in path:
            if "capture" in title or "monitor" in title:
                details.append("Capture wireless packets in monitor mode for offline analysis.")
            elif "deauth" in title:
                details.append("Send deauthentication frames to disconnect clients and capture handshakes.")
            elif "handshake" in title:
                details.append("Capture and validate WPA 4-way handshake for offline cracking.")
            elif "crack" in title or "pmk" in title:
                details.append("Crack WPA PSK by computing PMK from passphrase and comparing to handshake.")
            elif "inject" in title:
                details.append("Inject arbitrary frames into wireless networks.")
            else:
                details.append(f"WiFi security: {task['description']}")

        elif "nuclei" in path:
            if "template" in title:
                details.append("Parse YAML templates that define vulnerability detection logic.")
            elif "matcher" in title:
                details.append("Implement response matching with regex, status codes, and word matching.")
            elif "request" in title:
                details.append("Send templated HTTP requests with variable substitution.")
            elif "workflow" in title:
                details.append("Chain multiple templates with conditional execution logic.")
            else:
                details.append(f"Vulnerability scanner: {task['description']}")

        elif "privilege" in path or "escalation" in path:
            if "suid" in title:
                details.append("Find SUID/SGID binaries that can be exploited for privilege escalation.")
            elif "sudo" in title:
                details.append("Check sudo configuration for exploitable rules and misconfigurations.")
            elif "cron" in title:
                details.append("Find writable cron jobs and scripts that run as privileged users.")
            elif "service" in title:
                details.append("Find misconfigured services running as SYSTEM/root with weak permissions.")
            elif "capability" in title:
                details.append("Find binaries with dangerous Linux capabilities that enable privilege escalation.")
            else:
                details.append(f"Privilege escalation: {task['description']}")

        elif "coercion" in path or "coerce" in path:
            if "petitpotam" in title:
                details.append("Coerce authentication using EfsRpcOpenFileRaw via MS-EFSR protocol.")
            elif "printerbug" in title or "spooler" in title:
                details.append("Coerce authentication using Print Spooler RemoteFindFirstPrinterChangeNotification.")
            elif "dfscoerce" in title:
                details.append("Coerce authentication using DFS NetrDfsRemoveStdRoot.")
            elif "shadowcoerce" in title:
                details.append("Coerce authentication using VSS IsPathShadowCopied.")
            else:
                details.append(f"Authentication coercion: {task['description']}")

        else:
            details.append(f"Reimplement {tool_name}: {task['description']}")

    # === RED TEAM TOOLING paths ===
    elif "red team" in path and "tool" in path:
        lang = ""
        if "python" in path:
            lang = "Python"
        elif "go" in path:
            lang = "Go"
        elif "c#" in path or "csharp" in path:
            lang = "C#"
        elif "rust" in path:
            lang = "Rust"
        elif "c++" in path:
            lang = "C++"

        if "process" in title and "inject" in title:
            details.append(f"Implement process injection in {lang} to execute code in remote process context.")
        elif "shellcode" in title:
            details.append(f"Load and execute shellcode in {lang} using VirtualAlloc and CreateThread or equivalent.")
        elif "dll" in title and "inject" in title:
            details.append(f"Inject DLL into remote process using CreateRemoteThread or other techniques in {lang}.")
        elif "token" in title:
            details.append(f"Manipulate Windows access tokens for impersonation in {lang}.")
        elif "credential" in title or "dump" in title:
            details.append(f"Extract credentials from memory or registry using {lang}.")
        elif "persistence" in title:
            details.append(f"Implement persistence mechanisms (registry, scheduled tasks, services) in {lang}.")
        elif "evasion" in title or "bypass" in title:
            details.append(f"Implement AV/EDR evasion techniques in {lang}.")
        elif "enumerat" in title:
            details.append(f"Enumerate system information (processes, users, network) in {lang}.")
        elif "lateral" in title:
            details.append(f"Implement lateral movement techniques (WMI, SMB, WinRM) in {lang}.")
        elif "keylog" in title:
            details.append(f"Capture keystrokes using hooks or polling in {lang}.")
        elif "screenshot" in title:
            details.append(f"Capture screen contents programmatically in {lang}.")
        elif "exfil" in title:
            details.append(f"Exfiltrate data over various channels (HTTP, DNS, ICMP) in {lang}.")
        else:
            details.append(f"{lang} red team tool: {task['description']}")

    # === EVASION paths ===
    elif "evasion" in path or "payload" in path:
        if "packer" in title:
            details.append("Implement executable packer that compresses and encrypts payload with runtime unpacker stub.")
        elif "encrypt" in title:
            details.append("Encrypt payload with runtime decryption to avoid static signature detection.")
        elif "obfuscat" in title:
            details.append("Obfuscate code structure, strings, and control flow to hinder analysis.")
        elif "unhook" in title:
            details.append("Restore hooked API functions by reloading clean ntdll from disk.")
        elif "syscall" in title:
            details.append("Call NT syscalls directly to bypass user-mode API hooks.")
        elif "etw" in title:
            details.append("Patch ETW to disable telemetry collection by security products.")
        elif "amsi" in title:
            details.append("Bypass AMSI by patching AmsiScanBuffer or other techniques.")
        elif "sandbox" in title:
            details.append("Detect sandbox/VM environment and alter behavior to avoid analysis.")
        elif "sleep" in title:
            details.append("Implement sleep obfuscation to encrypt payload in memory during sleep.")
        else:
            details.append(f"Evasion technique: {task['description']}")

    # === EXPLOIT DEVELOPMENT ===
    elif "exploit" in path:
        if "overflow" in title:
            details.append("Exploit stack buffer overflow to overwrite return address and redirect execution.")
        elif "rop" in title:
            details.append("Build ROP chain from gadgets to bypass DEP/NX protection.")
        elif "heap" in title:
            details.append("Exploit heap corruption vulnerabilities using heap spray or use-after-free.")
        elif "format" in title:
            details.append("Exploit format string vulnerabilities to read/write arbitrary memory.")
        elif "shellcode" in title:
            details.append("Write position-independent shellcode that avoids bad characters.")
        elif "aslr" in title:
            details.append("Bypass ASLR using information leaks or brute force techniques.")
        elif "canary" in title or "stack" in title:
            details.append("Bypass stack canaries using information leaks or canary brute force.")
        else:
            details.append(f"Exploit development: {task['description']}")

    # === GENERIC FALLBACK ===
    else:
        if task["description"]:
            details.append(f"{task['description']}")
        else:
            details.append(f"Complete this task to build skills in {task['path_name']}.")

    # Implementation section
    details.append("\n### Implementation")

    # Add code examples based on context
    if "shell" in path and "build" in path:
        if "pipe" in title:
            details.append("""```c
int pipefd[2];
pipe(pipefd);

if (fork() == 0) {
    // Child: write to pipe
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    execvp(cmd1[0], cmd1);
}
// Parent: read from pipe
close(pipefd[1]);
dup2(pipefd[0], STDIN_FILENO);
execvp(cmd2[0], cmd2);
```""")
        elif "redirect" in title:
            details.append("""```c
// Input redirection: command < file
int fd = open(filename, O_RDONLY);
dup2(fd, STDIN_FILENO);
close(fd);

// Output redirection: command > file
int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
dup2(fd, STDOUT_FILENO);
close(fd);
```""")
        else:
            details.append("Follow the implementation pattern for your shell component.")

    elif "debugger" in path and "build" in path:
        if "single" in title or "step" in title:
            details.append("""```c
// Single step one instruction
ptrace(PTRACE_SINGLESTEP, pid, NULL, NULL);
waitpid(pid, &status, 0);

// Check if stopped by single step
if (WIFSTOPPED(status) && WSTOPSIG(status) == SIGTRAP) {
    // Read current instruction pointer
    struct user_regs_struct regs;
    ptrace(PTRACE_GETREGS, pid, NULL, &regs);
    printf("RIP: 0x%llx\\n", regs.rip);
}
```""")
        elif "breakpoint" in title or "int3" in title:
            details.append("""```c
// Set breakpoint by inserting INT3
long orig = ptrace(PTRACE_PEEKTEXT, pid, addr, NULL);
long trap = (orig & ~0xFF) | 0xCC;
ptrace(PTRACE_POKETEXT, pid, addr, trap);

// On breakpoint hit, restore original byte
ptrace(PTRACE_POKETEXT, pid, addr, orig);
// Back up RIP by 1 to re-execute instruction
regs.rip--;
ptrace(PTRACE_SETREGS, pid, NULL, &regs);
```""")
        else:
            details.append("Use ptrace() system call for debugger implementation.")

    elif "sqlite" in path and "build" in path:
        if "b-tree" in title:
            details.append("""```c
// B-tree node structure
typedef struct {
    bool is_leaf;
    uint32_t num_keys;
    uint32_t keys[MAX_KEYS];
    uint32_t children[MAX_KEYS + 1];  // For internal nodes
    void* values[MAX_KEYS];            // For leaf nodes
} BTreeNode;

// Search in B-tree
void* btree_search(BTreeNode* node, uint32_t key) {
    int i = 0;
    while (i < node->num_keys && key > node->keys[i]) i++;

    if (i < node->num_keys && key == node->keys[i])
        return node->values[i];
    if (node->is_leaf)
        return NULL;
    return btree_search(node->children[i], key);
}
```""")
        else:
            details.append("Implement using page-based storage with B-tree indexes.")

    elif "reimplement" in path and "mimikatz" in path:
        details.append("""```c
// Open LSASS process
HANDLE hProcess = OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, FALSE, lsassPid);

// Create minidump
MiniDumpWriteDump(hProcess, lsassPid, hFile, MiniDumpWithFullMemory, NULL, NULL, NULL);

// Or read memory directly
ReadProcessMemory(hProcess, address, buffer, size, &bytesRead);
```""")

    elif "reimplement" in path and ("nmap" in path or "scanner" in path):
        details.append("""```python
import socket

def syn_scan(host, port):
    # Create raw socket
    s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)

    # Build TCP SYN packet
    # Send and check for SYN-ACK response

def connect_scan(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, port))
        s.close()
        return True  # Port open
    except:
        return False  # Port closed/filtered
```""")

    elif "c2" in path or "sliver" in path:
        details.append("""```go
// Implant beacon loop
func beacon(c2Server string, interval time.Duration) {
    for {
        // Check in with C2
        tasks := checkIn(c2Server)

        // Execute tasks
        for _, task := range tasks {
            result := executeTask(task)
            sendResult(c2Server, result)
        }

        // Sleep with jitter
        jitter := time.Duration(rand.Intn(30)) * time.Second
        time.Sleep(interval + jitter)
    }
}
```""")

    elif "evasion" in path or "bypass" in path:
        if "unhook" in title:
            details.append("""```c
// Unhook ntdll by reloading from disk
HANDLE hFile = CreateFile("C:\\\\Windows\\\\System32\\\\ntdll.dll", ...);
HANDLE hMap = CreateFileMapping(hFile, ...);
LPVOID pClean = MapViewOfFile(hMap, ...);

// Copy .text section from clean ntdll to loaded ntdll
memcpy(pLoadedNtdll + textOffset, pClean + textOffset, textSize);
```""")
        else:
            details.append("Implement evasion using appropriate technique for the target security control.")
    else:
        details.append("Follow the standard implementation pattern for this component.")

    # Key concepts section
    details.append("\n### Key Concepts")
    details.append("- Understand the underlying mechanism before coding")
    details.append("- Handle edge cases and error conditions")
    details.append("- Test against real-world scenarios")
    details.append("- Document your implementation decisions")

    # Practice section
    details.append("\n### Practice")
    details.append("- [ ] Implement the core functionality")
    details.append("- [ ] Add error handling")
    details.append("- [ ] Test with various inputs")
    details.append("- [ ] Optimize for performance if needed")

    # Completion criteria
    details.append("\n### Completion Criteria")
    details.append("- [ ] Feature works as expected")
    details.append("- [ ] Edge cases handled properly")
    details.append("- [ ] Code is clean and documented")
    details.append("- [ ] Can explain the implementation")

    return "\n".join(details)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all tasks needing details
    cursor.execute("""
        SELECT t.id, t.title, t.description, p.name as path_name
        FROM tasks t
        JOIN modules m ON t.module_id = m.id
        JOIN paths p ON m.path_id = p.id
        WHERE t.details IS NULL OR length(t.details) < 200
        ORDER BY p.id, t.id
    """)

    tasks = [dict(row) for row in cursor.fetchall()]
    print(f"Found {len(tasks)} tasks needing details...")

    # Update each task
    update_stmt = cursor.execute
    updated = 0

    for task in tasks:
        new_details = generate_details(task)
        cursor.execute(
            "UPDATE tasks SET details = ? WHERE id = ?",
            (new_details, task["id"])
        )
        updated += 1
        if updated % 50 == 0:
            print(f"  Updated {updated} tasks...")

    conn.commit()
    conn.close()

    print(f"\nDone! Updated {updated} tasks with comprehensive details.")


if __name__ == "__main__":
    main()
