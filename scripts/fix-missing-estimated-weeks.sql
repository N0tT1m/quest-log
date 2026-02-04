-- Fix paths missing estimated_weeks
UPDATE paths SET estimated_weeks = 12 WHERE name = 'AI/ML Deep Learning' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 10 WHERE name = 'Blue Team & Defensive Security' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 4 WHERE name = 'Build Your Own Hacking Lab' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 8 WHERE name = 'CTF Challenge Practice' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 12 WHERE name = 'Complete Homelab Infrastructure' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 10 WHERE name = 'DevSecOps Engineering' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 8 WHERE name = 'Evasion & Payload Tools' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 10 WHERE name = 'Exploit Development Tools' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 10 WHERE name = 'ML Engineering & Ops' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 6 WHERE name = 'Network Analysis & Traffic Forensics' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 12 WHERE name = 'Red Team & Offensive Security' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 10 WHERE name = 'Red Team Extended: Web, Cloud & Mobile' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 8 WHERE name = 'Reimplement Red Team Tools: Network' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 8 WHERE name = 'Reimplement Red Team Tools: Web & AD' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 6 WHERE name = 'Reimplement: Password & WiFi Cracking' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
UPDATE paths SET estimated_weeks = 8 WHERE name = 'Reimplement: Pivoting & C2' AND (estimated_weeks IS NULL OR estimated_weeks = 0);
