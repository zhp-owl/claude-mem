---
name: clear-memory
description: Clear claude-mem observations and memory data. Use when user wants to reset memory, clear history, or free up storage.
---

# Clear Memory

Clear observation history from claude-mem database. Use with caution - this operation is irreversible.

## When to Use

- User explicitly requests clearing memory
- User wants to reset the observation history
- User wants to free up storage space
- User is starting fresh with a project

## Database Location

The claude-mem database is located at:
- Windows: `C:\Users\<username>\.claude-mem\claude-mem.db`
- macOS/Linux: `~/.claude-mem/claude-mem.db`

## Clear Commands

### Clear All Observations

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations; DELETE FROM user_prompts; DELETE FROM session_summaries; DELETE FROM pending_messages; DELETE FROM sdk_sessions;"
```

### Clear Observations by Project

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations WHERE project='project-name';"
```

### Clear Observations by Type

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations WHERE type='bugfix';"
```

### Clear Observations by Date Range

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations WHERE created_at_epoch < strftime('%s', '2025-01-01') * 1000;"
```

### Clear Old Observations (Keep Last N Days)

```bash
# Keep only last 30 days
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations WHERE created_at_epoch < (strftime('%s', 'now') - 30*86400) * 1000;"
```

### Clear Specific Observation by ID

```bash
sqlite3 ~/.claude-mem/claude-mem.db "DELETE FROM observations WHERE id=12345;"
```

## Clear Corpora

Corpora are stored in `~/.claude-mem/corpora/` as JSON files:

```bash
rm -rf ~/.claude-mem/corpora/*.json
```

## Verify Clear

```bash
sqlite3 ~/.claude-mem/claude-mem.db "SELECT 'observations' as tbl, COUNT(*) FROM observations UNION ALL SELECT 'user_prompts', COUNT(*) FROM user_prompts UNION ALL SELECT 'session_summaries', COUNT(*) FROM session_summaries;"
```

## Tables Reference

| Table | Description |
|-------|-------------|
| `observations` | Main observation records |
| `user_prompts` | User prompt history |
| `session_summaries` | Session summaries |
| `pending_messages` | Queued messages |
| `sdk_sessions` | SDK session metadata |

## Safety Notes

1. **Backup before clearing** - Consider backing up the database:
   ```bash
   cp ~/.claude-mem/claude-mem.db ~/.claude-mem/claude-mem.db.backup
   ```

2. **Worker support** - After clearing, consider restarting the worker:
   ```bash
   curl -X POST http://localhost:37777/api/restart
   ```

3. **Chroma data** - If using Chroma, also clear the vector store for clean slate.