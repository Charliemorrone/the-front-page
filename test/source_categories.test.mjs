// Focused tests for source-category dashboard tagging (db + API layers).
//
// Exercises:
//   - createSource + setSourceCategories round-trip (join rows present / empty / idempotent).
//   - getSourceCategories returns categories in sorted order.
//   - setSourceCategories trims, dedupes, drops blanks, replaces atomically.
//   - Ownership: PUT /api/sources/:id with categories rejects 403 for non-owner.
//   - Worker resolver sees the tagged source under its category (the load-bearing
//     contract — categories tagged through the dashboard must actually feed the
//     daily-brief source plan).
//
// Standalone — talks to the db module directly + spins up the HTTP server on a
// throwaway port. No external services, no seed data. Run with:
//   node test/source_categories.test.mjs

import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { spawn } from 'child_process';
import { setTimeout as delay } from 'timers/promises';

import { getDb, createSource, setSourceCategories, getSourceCategories, upsertUser, createSession } from '../src/db.mjs';

let PASS = 0;
let FAIL = 0;

function check(name, cond, detail) {
  if (cond) { PASS++; console.log(`  ✅ ${name}`); }
  else { FAIL++; console.log(`  ❌ ${name}${detail ? '\n     ' + detail : ''}`); }
}

function eqArr(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

const tmp = mkdtempSync(join(tmpdir(), 'clawfeed-cattest-'));
const dbPath = join(tmp, 'digest.db');
let serverProc = null;

async function cleanup() {
  if (serverProc) { try { serverProc.kill('SIGTERM'); } catch {} }
  try { rmSync(tmp, { recursive: true, force: true }); } catch {}
}

process.on('exit', () => { if (serverProc) { try { serverProc.kill('SIGKILL'); } catch {} } });

async function main() {
  console.log('\n─── source_categories: db helpers ───');
  const db = getDb(dbPath);

  // Seed two users.
  const alice = upsertUser(db, { googleId: 'cat-alice', email: 'alice@cat.test', name: 'Alice', avatar: '' });
  const bob = upsertUser(db, { googleId: 'cat-bob', email: 'bob@cat.test', name: 'Bob', avatar: '' });

  // 1. Create source without categories → join empty + source still created.
  const s1 = createSource(db, { name: 'Bare RSS', type: 'rss', config: '{"url":"https://a.test/rss"}', isPublic: 0, createdBy: alice.id });
  check('create source without categories returns id', typeof s1.id === 'number' && s1.id > 0);
  check('source with no setSourceCategories has no rows', eqArr(getSourceCategories(db, s1.id), []));

  // 2. Create source then tag → join rows present, sorted.
  const s2 = createSource(db, { name: 'Tagged RSS', type: 'rss', config: '{"url":"https://b.test/rss"}', isPublic: 0, createdBy: alice.id });
  setSourceCategories(db, s2.id, ['ai_research', 'startup_funding']);
  check('two categories applied + returned sorted',
    eqArr(getSourceCategories(db, s2.id), ['ai_research', 'startup_funding']));

  // 3. Replay → idempotent.
  setSourceCategories(db, s2.id, ['ai_research', 'startup_funding']);
  check('setSourceCategories replay is idempotent',
    eqArr(getSourceCategories(db, s2.id), ['ai_research', 'startup_funding']));

  // 4. Track changes [] → ["x","y"] → ["y"].
  const s3 = createSource(db, { name: 'Tracker', type: 'rss', config: '{"url":"https://c.test/rss"}', isPublic: 0, createdBy: alice.id });
  check('initially empty', eqArr(getSourceCategories(db, s3.id), []));
  setSourceCategories(db, s3.id, ['x', 'y']);
  check('after [x,y]', eqArr(getSourceCategories(db, s3.id), ['x', 'y']));
  setSourceCategories(db, s3.id, ['y']);
  check('after [y] removes x', eqArr(getSourceCategories(db, s3.id), ['y']));
  setSourceCategories(db, s3.id, []);
  check('after [] clears all', eqArr(getSourceCategories(db, s3.id), []));

  // 5. Normalization: trim, dedupe, drop blanks.
  setSourceCategories(db, s3.id, ['  ai_research  ', 'ai_research', '', '   ', 'github_traction']);
  check('trim + dedupe + drop blanks',
    eqArr(getSourceCategories(db, s3.id), ['ai_research', 'github_traction']));

  // 6. Validation: non-array rejected.
  let threw = false;
  try { setSourceCategories(db, s3.id, 'ai_research'); } catch { threw = true; }
  check('non-array rejected', threw);

  // 7. Validation: non-string entries rejected.
  threw = false;
  try { setSourceCategories(db, s3.id, ['ok', 42]); } catch { threw = true; }
  check('non-string entry rejected', threw);

  // 8. Cascade: deleting a source removes its category rows (FK ON DELETE CASCADE).
  const s4 = createSource(db, { name: 'Cascade', type: 'rss', config: '{"url":"https://d.test/rss"}', isPublic: 0, createdBy: alice.id });
  setSourceCategories(db, s4.id, ['a', 'b']);
  db.prepare('DELETE FROM sources WHERE id = ?').run(s4.id);
  check('FK cascade clears source_categories on hard delete', eqArr(getSourceCategories(db, s4.id), []));

  // 9. Worker-resolver contract: a tagged source is visible through the same
  //    JOIN the Python worker uses (sources.py:_attach_db_sources).
  setSourceCategories(db, s2.id, ['ai_research']);
  const rows = db.prepare(`
    SELECT s.id, sc.category
      FROM sources s
      JOIN source_categories sc ON sc.source_id = s.id
     WHERE s.is_active = 1 AND s.id = ?
  `).all(s2.id);
  check('tagged active source visible via worker JOIN',
    rows.length === 1 && rows[0].category === 'ai_research');

  // ─── API layer: PUT auth/ownership ───
  console.log('\n─── source_categories: API auth on PUT ───');
  // Stand up the server on a free port against the same DB.
  const port = 8000 + Math.floor(Math.random() * 1000);
  serverProc = spawn(process.execPath, ['src/server.mjs'], {
    cwd: join(import.meta.dirname || new URL('.', import.meta.url).pathname, '..'),
    env: { ...process.env, DIGEST_DB: dbPath, DIGEST_PORT: String(port), DIGEST_HOST: '127.0.0.1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let stderrBuf = '';
  serverProc.stderr.on('data', (c) => { stderrBuf += c.toString(); });
  // Wait for readiness.
  const deadline = Date.now() + 5000;
  let ready = false;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/api/health`);
      if (r.ok) { ready = true; break; }
    } catch {}
    await delay(80);
  }
  check('server up on test port', ready, ready ? '' : `stderr: ${stderrBuf.slice(0, 200)}`);
  if (!ready) {
    await cleanup();
    summarize();
    return;
  }

  // Mint sessions for alice + bob directly in DB (mirrors test/setup.sh pattern).
  const aliceSess = 'cat-test-sess-alice';
  const bobSess = 'cat-test-sess-bob';
  createSession(db, { id: aliceSess, userId: alice.id, expiresAt: new Date(Date.now() + 86400000).toISOString() });
  createSession(db, { id: bobSess, userId: bob.id, expiresAt: new Date(Date.now() + 86400000).toISOString() });

  const API = `http://127.0.0.1:${port}/api`;

  // POST /api/sources with categories → join populated.
  let resp = await fetch(`${API}/sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', cookie: `session=${aliceSess}` },
    body: JSON.stringify({ name: 'API created', type: 'rss', config: '{"url":"https://e.test/rss"}', categories: ['ai_research', '  github_traction  '] }),
  });
  const created = await resp.json();
  check('POST /api/sources with categories → 201', resp.status === 201);
  check('POST sets categories (trimmed + sorted on read)',
    eqArr(getSourceCategories(db, created.id), ['ai_research', 'github_traction']));

  // GET /api/sources/:id returns categories array.
  resp = await fetch(`${API}/sources/${created.id}`, { headers: { cookie: `session=${aliceSess}` } });
  const fetched = await resp.json();
  check('GET /api/sources/:id includes categories',
    Array.isArray(fetched.categories) && eqArr(fetched.categories, ['ai_research', 'github_traction']));

  // POST without categories → still works, source created, join empty.
  resp = await fetch(`${API}/sources`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', cookie: `session=${aliceSess}` },
    body: JSON.stringify({ name: 'No-cats', type: 'rss', config: '{"url":"https://f.test/rss"}' }),
  });
  const noCats = await resp.json();
  check('POST without categories → 201', resp.status === 201);
  check('POST without categories → empty join', eqArr(getSourceCategories(db, noCats.id), []));

  // PUT /api/sources/:id by Bob (not owner) → 403, categories untouched.
  resp = await fetch(`${API}/sources/${created.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', cookie: `session=${bobSess}` },
    body: JSON.stringify({ categories: ['hijacked'] }),
  });
  check('Bob PUT on Alice source → 403', resp.status === 403);
  check('Bob attempt did not modify categories',
    eqArr(getSourceCategories(db, created.id), ['ai_research', 'github_traction']));

  // PUT /api/sources/:id by owner → updates categories.
  resp = await fetch(`${API}/sources/${created.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', cookie: `session=${aliceSess}` },
    body: JSON.stringify({ categories: ['ai_coding_tools'] }),
  });
  check('Alice PUT on own source → 200', resp.status === 200);
  check('Alice PUT replaced categories',
    eqArr(getSourceCategories(db, created.id), ['ai_coding_tools']));

  // PUT without categories field → preserves existing categories (optional field).
  resp = await fetch(`${API}/sources/${created.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', cookie: `session=${aliceSess}` },
    body: JSON.stringify({ name: 'Renamed' }),
  });
  check('PUT without categories → 200', resp.status === 200);
  check('PUT without categories preserves prior categories',
    eqArr(getSourceCategories(db, created.id), ['ai_coding_tools']));

  await cleanup();
  summarize();
}

function summarize() {
  const total = PASS + FAIL;
  console.log('\n═══════════════════════════════════════════');
  console.log(`  source_categories: ${PASS}/${total} passed${FAIL ? `, ${FAIL} failed` : ''}`);
  console.log('═══════════════════════════════════════════\n');
  process.exit(FAIL > 0 ? 1 : 0);
}

main().catch(async (e) => {
  console.error('test crashed:', e);
  await cleanup();
  process.exit(1);
});
