#!/usr/bin/env node
/**
 * Protected sync-marketplace script
 *
 * Prevents accidental rsync overwrite when installed plugin is on beta branch.
 * If on beta, the user should use the UI to update instead.
 */

const { execSync } = require('child_process');
const { existsSync, readFileSync, readdirSync, statSync, mkdirSync, copyFileSync, rmSync } = require('fs');
const path = require('path');
const os = require('os');

const INSTALLED_PATH = path.join(os.homedir(), '.claude', 'plugins', 'marketplaces', 'thedotmack');
const CACHE_BASE_PATH = path.join(os.homedir(), '.claude', 'plugins', 'cache', 'thedotmack', 'claude-mem');

function getCurrentBranch() {
  try {
    if (!existsSync(path.join(INSTALLED_PATH, '.git'))) {
      return null;
    }
    return execSync('git rev-parse --abbrev-ref HEAD', {
      cwd: INSTALLED_PATH,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim();
  } catch {
    return null;
  }
}

function getGitignorePatterns(basePath) {
  const gitignorePath = path.join(basePath, '.gitignore');
  if (!existsSync(gitignorePath)) return [];

  const lines = readFileSync(gitignorePath, 'utf-8').split('\n');
  return lines
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#') && !line.startsWith('!'));
}

function matchesPattern(normalizedRel, pattern) {
  // Strip trailing slash (directory indicator in gitignore)
  const p = pattern.replace(/\/$/, '');

  // Build regex from glob pattern
  const regexStr = p
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '\x00')
    .replace(/\*/g, '[^/]*')
    .replace(/\x00/g, '.*');

  // Pattern without slash matches any path component
  if (!p.includes('/')) {
    return new RegExp(`(^|/)${regexStr}(/|$)`).test(normalizedRel);
  }
  // Pattern with slash matches from root
  return new RegExp(`^${regexStr}(/.*)?$`).test(normalizedRel);
}

function shouldExclude(relPath, excludePatterns) {
  const normalized = relPath.replace(/\\/g, '/');
  return excludePatterns.some(p => matchesPattern(normalized, p));
}

/**
 * Cross-platform directory sync (equivalent to rsync -av --delete --exclude=...).
 * Does NOT delete excluded items in the destination (mirrors rsync default --delete behavior).
 */
function syncDirectories(src, dest, excludePatterns = []) {
  if (!existsSync(dest)) {
    mkdirSync(dest, { recursive: true });
  }

  const srcFiles = new Set();

  function walkAndCopy(dir, relBase) {
    let entries;
    try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }

    for (const entry of entries) {
      const rel = relBase ? `${relBase}/${entry.name}` : entry.name;
      if (shouldExclude(rel, excludePatterns)) continue;

      const srcPath = path.join(dir, entry.name);
      const destPath = path.join(dest, rel);

      if (entry.isDirectory()) {
        srcFiles.add(rel);
        if (!existsSync(destPath)) mkdirSync(destPath, { recursive: true });
        walkAndCopy(srcPath, rel);
      } else if (entry.isFile()) {
        srcFiles.add(rel);
        let needsCopy = true;
        if (existsSync(destPath)) {
          const ss = statSync(srcPath), ds = statSync(destPath);
          needsCopy = ss.size !== ds.size || ss.mtimeMs > ds.mtimeMs;
        }
        if (needsCopy) {
          mkdirSync(path.dirname(destPath), { recursive: true });
          copyFileSync(srcPath, destPath);
        }
      }
    }
  }

  walkAndCopy(src, '');

  // Delete dest items not present in src (mirrors --delete), skip excluded items
  function walkAndDelete(dir, relBase) {
    if (!existsSync(dir)) return;
    let entries;
    try { entries = readdirSync(dir, { withFileTypes: true }); } catch { return; }

    for (const entry of entries) {
      const rel = relBase ? `${relBase}/${entry.name}` : entry.name;
      if (shouldExclude(rel, excludePatterns)) continue;
      if (!srcFiles.has(rel)) {
        rmSync(path.join(dest, rel), { recursive: true, force: true });
      } else if (entry.isDirectory()) {
        walkAndDelete(path.join(dir, entry.name), rel);
      }
    }
  }

  walkAndDelete(dest, '');
}

const branch = getCurrentBranch();
const isForce = process.argv.includes('--force');

if (branch && branch !== 'main' && !isForce) {
  console.log('');
  console.log('\x1b[33m%s\x1b[0m', `WARNING: Installed plugin is on beta branch: ${branch}`);
  console.log('\x1b[33m%s\x1b[0m', 'Running sync would overwrite beta code.');
  console.log('');
  console.log('Options:');
  console.log('  1. Use UI at http://localhost:37777 to update beta');
  console.log('  2. Switch to stable in UI first, then run sync');
  console.log('  3. Force sync: npm run sync-marketplace:force');
  console.log('');
  process.exit(1);
}

// Get version from plugin.json
function getPluginVersion() {
  try {
    const pluginJsonPath = path.join(__dirname, '..', 'plugin', '.claude-plugin', 'plugin.json');
    const pluginJson = JSON.parse(readFileSync(pluginJsonPath, 'utf-8'));
    return pluginJson.version;
  } catch (error) {
    console.error('\x1b[31m%s\x1b[0m', 'Failed to read plugin version:', error.message);
    process.exit(1);
  }
}

// Normal sync for main branch or fresh install
console.log('Syncing to marketplace...');
try {
  const rootDir = path.join(__dirname, '..');
  const gitignorePatterns = getGitignorePatterns(rootDir);
  const mainExcludes = ['.git', 'bun.lock', 'package-lock.json', ...gitignorePatterns];

  syncDirectories(rootDir, INSTALLED_PATH, mainExcludes);

  console.log('Running bun install in marketplace...');
  execSync('bun install', { cwd: INSTALLED_PATH, stdio: 'inherit' });

  // Sync to cache folder with version
  const version = getPluginVersion();
  const CACHE_VERSION_PATH = path.join(CACHE_BASE_PATH, version);

  const pluginDir = path.join(rootDir, 'plugin');
  const pluginGitignorePatterns = getGitignorePatterns(pluginDir);
  const cacheExcludes = ['.git', ...pluginGitignorePatterns];

  console.log(`Syncing to cache folder (version ${version})...`);
  syncDirectories(pluginDir, CACHE_VERSION_PATH, cacheExcludes);

  // Install dependencies in cache directory so worker can resolve them
  console.log(`Running bun install in cache folder (version ${version})...`);
  execSync(`bun install`, { cwd: CACHE_VERSION_PATH, stdio: 'inherit' });

  console.log('\x1b[32m%s\x1b[0m', 'Sync complete!');

  // Trigger worker restart after file sync
  console.log('\n🔄 Triggering worker restart...');
  const http = require('http');
  const req = http.request({
    hostname: '127.0.0.1',
    port: 37777,
    path: '/api/admin/restart',
    method: 'POST',
    timeout: 2000
  }, (res) => {
    if (res.statusCode === 200) {
      console.log('\x1b[32m%s\x1b[0m', '✓ Worker restart triggered');
    } else {
      console.log('\x1b[33m%s\x1b[0m', `ℹ Worker restart returned status ${res.statusCode}`);
    }
  });
  req.on('error', () => {
    console.log('\x1b[33m%s\x1b[0m', 'ℹ Worker not running, will start on next hook');
  });
  req.on('timeout', () => {
    req.destroy();
    console.log('\x1b[33m%s\x1b[0m', 'ℹ Worker restart timed out');
  });
  req.end();

} catch (error) {
  console.error('\x1b[31m%s\x1b[0m', 'Sync failed:', error.message);
  process.exit(1);
}