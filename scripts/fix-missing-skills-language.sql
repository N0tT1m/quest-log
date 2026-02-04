-- Fix paths missing skills and language metadata
UPDATE paths SET
  skills = 'Neural networks, Deep learning, PyTorch, Transformers, NLP, Computer vision',
  language = 'Python'
WHERE name = 'AI/ML Deep Learning' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'SIEM, Log analysis, Threat hunting, Incident response, Forensics, Malware analysis',
  language = 'Python, PowerShell'
WHERE name = 'Blue Team & Defensive Security' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Virtualization, Networking, Active Directory, Vulnerable machines, Lab infrastructure',
  language = 'Bash, PowerShell'
WHERE name = 'Build Your Own Hacking Lab' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Web exploitation, Binary exploitation, Cryptography, Forensics, Reverse engineering',
  language = 'Python, C, Assembly'
WHERE name = 'CTF Challenge Practice' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Proxmox, Docker, Kubernetes, Networking, Storage, Monitoring, Automation',
  language = 'Bash, YAML, Python'
WHERE name = 'Complete Homelab Infrastructure' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'CI/CD, Container security, SAST/DAST, Kubernetes security, Supply chain security',
  language = 'Python, Go, YAML'
WHERE name = 'DevSecOps Engineering' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'AV evasion, EDR bypass, Shellcode, Packers, Obfuscation, AMSI bypass',
  language = 'C, C++, C#, Go'
WHERE name = 'Evasion & Payload Tools' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Buffer overflow, ROP, Heap exploitation, Shellcode, Fuzzing, Vulnerability research',
  language = 'C, Python, Assembly'
WHERE name = 'Exploit Development Tools' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'MLOps, Model serving, Feature stores, Experiment tracking, Model monitoring',
  language = 'Python'
WHERE name = 'ML Engineering & Ops' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Packet analysis, Wireshark, Protocol dissection, Network forensics, Traffic analysis',
  language = 'Python, Lua'
WHERE name = 'Network Analysis & Traffic Forensics' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Penetration testing, Active Directory, Privilege escalation, Lateral movement, C2',
  language = 'Python, PowerShell, C#'
WHERE name = 'Red Team & Offensive Security' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Web hacking, Cloud security, Mobile security, API testing, AWS/Azure attacks',
  language = 'Python, JavaScript, Go'
WHERE name = 'Red Team Extended: Web, Cloud & Mobile' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Network scanning, Packet crafting, Protocol implementation, Service enumeration',
  language = 'Python, Go, C'
WHERE name = 'Reimplement Red Team Tools: Network' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Web proxying, AD attacks, LDAP, Kerberos, NTLM, Credential harvesting',
  language = 'Python, C#, Go'
WHERE name = 'Reimplement Red Team Tools: Web & AD' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Password cracking, Hash analysis, WiFi security, WPA attacks, GPU acceleration',
  language = 'C, Python, CUDA'
WHERE name = 'Reimplement: Password & WiFi Cracking' AND (skills IS NULL OR skills = '');

UPDATE paths SET
  skills = 'Tunneling, Port forwarding, SOCKS proxy, C2 development, Implant design',
  language = 'Go, C#, Rust'
WHERE name = 'Reimplement: Pivoting & C2' AND (skills IS NULL OR skills = '');
