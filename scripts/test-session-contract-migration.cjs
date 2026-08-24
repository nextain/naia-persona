#!/usr/bin/env node

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { migrateStatus } = require("./migrate-process-status-session-contract.cjs");

const root = path.resolve(__dirname, "..");
const status = JSON.parse(fs.readFileSync(path.join(root, ".agents", "context", "process-status.json"), "utf8"));
assert.equal(status.legacy_migration.mode, "read_only_compatibility");
assert.deepEqual(status.legacy_migration.current_work_snapshot, status.current_work, "legacy current_work migration must preserve every field and value");
assert(!status._usage.session_start.some((step) => /current_work|last_updated/.test(step)), "session start must not mutate shared current work");
assert(status._usage.new_issue.some((step) => step.includes("session-contracts")), "new work must route to per-session contracts");

const unknownLegacy = {
	schema_version: "1.0",
	current_work: {
		issue: "legacy-17",
		unknown_scalar: "preserve exactly",
		unknown_nested: { list: [1, { keep: true }] },
	},
	other_shared_state: { untouched: true },
};
const migratedUnknown = migrateStatus(unknownLegacy);
assert.deepEqual(migratedUnknown.current_work, unknownLegacy.current_work, "migration must not rewrite legacy current_work");
assert.deepEqual(migratedUnknown.legacy_migration.current_work_snapshot, unknownLegacy.current_work, "unknown legacy fields and nested values must survive migration");
assert.deepEqual(migratedUnknown.other_shared_state, unknownLegacy.other_shared_state, "unrelated shared state must remain unchanged");

for (const name of ["AGENTS.md", "CLAUDE.md", "GEMINI.md", "OPENCODE.md", "CODEX.md"]) {
	const content = fs.readFileSync(path.join(root, name));
	assert(content.equals(fs.readFileSync(path.join(root, "AGENTS.md"))), `${name} must be byte-identical to AGENTS.md`);
}

console.log("template session-contract migration: PASS");
