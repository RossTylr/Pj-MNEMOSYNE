#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const sessionsDir = path.join('.claude', 'sessions');
if (!fs.existsSync(sessionsDir)) {
  console.log('[session-start] No previous sessions found. Starting fresh.');
  process.exit(0);
}

const files = fs.readdirSync(sessionsDir)
  .filter(f => f.endsWith('.md'))
  .sort()
  .reverse()
  .slice(0, 3);

if (files.length === 0) {
  console.log('[session-start] No session history found.');
  process.exit(0);
}

console.log(`[session-start] Found ${files.length} recent session(s):\n`);
files.forEach(f => {
  const content = fs.readFileSync(path.join(sessionsDir, f), 'utf8');
  const nextAction = content.match(/### Next action\n([\s\S]*?)\n---/)?.[1]?.trim();
  console.log(`  ${f}: ${nextAction || '(no next action recorded)'}`);
});
console.log('\nLoad most recent session with: cat .claude/sessions/' + files[0]);
