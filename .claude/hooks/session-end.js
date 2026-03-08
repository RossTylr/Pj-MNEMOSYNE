#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

// 1. Write session summary file
const sessionsDir = path.join('.claude', 'sessions');
if (!fs.existsSync(sessionsDir)) fs.mkdirSync(sessionsDir, { recursive: true });

const today = new Date().toISOString().split('T')[0];
const sessionFile = path.join(sessionsDir, `${today}.md`);
const timestamp = new Date().toISOString();

const sessionTemplate = `## Session: ${timestamp}

### Last completed notebook
<!-- Fill: e.g. 01/03_mlp_numpy.ipynb Cell 7 — loss curve rendering -->

### What was verified working
<!-- Fill: what E2E smoke test / nbconvert confirmed -->

### Decisions made
<!-- Fill: any threshold or implementation decisions -->

### Next action
<!-- Fill: exact first step for next session -->

### Blockers / open questions
<!-- Fill: anything unresolved -->

---
`;
fs.appendFileSync(sessionFile, sessionTemplate);

// 2. Append stub entry to EXPERIMENT_LOG.md
const logFile = path.join('06_research_track', 'EXPERIMENT_LOG.md');
const logEntry = `\n<!-- SESSION END ${timestamp} — fill in completed notebook and findings above -->\n`;
if (fs.existsSync(logFile)) {
  fs.appendFileSync(logFile, logEntry);
}

console.log(`[session-end] Session summary: ${sessionFile}`);
console.log(`[session-end] EXPERIMENT_LOG stub appended.`);
