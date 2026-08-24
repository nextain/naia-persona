#!/usr/bin/env node
/** Lossless field/value migration of legacy process-status.current_work. */
const fs = require("node:fs");
const path = require("node:path");

function clone(value) {
	return JSON.parse(JSON.stringify(value));
}

function migrateStatus(input) {
	if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("process status must be an object");
	if (!input.current_work || typeof input.current_work !== "object" || Array.isArray(input.current_work)) throw new Error("current_work must be an object");
	const output = clone(input);
	output.schema_version = "1.1";
	output.legacy_migration = {
		...(output.legacy_migration || {}),
		mode: "read_only_compatibility",
		migrated_to: ".agents/session-contracts/",
		note: "current_work is preserved as read-only migration evidence and never grants mutation authority.",
		current_work_snapshot: clone(input.current_work),
	};
	return output;
}

function main() {
	const filePath = process.argv[2];
	if (!filePath) throw new Error("usage: migrate-process-status-session-contract.cjs <process-status.json>");
	const target = path.resolve(filePath);
	const next = migrateStatus(JSON.parse(fs.readFileSync(target, "utf8")));
	const temporary = `${target}.session-contract-${process.pid}.tmp`;
	fs.writeFileSync(temporary, `${JSON.stringify(next, null, 2)}\n`);
	fs.renameSync(temporary, target);
}

if (require.main === module) main();
module.exports = { migrateStatus };
