import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const packagesDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

const packageRequirements = {
    common: ["ArtifactSource", "@fnnx-ai/node", "@fnnx-ai/web"],
    node: ["Model.fromPath", "Model.fromBuffer", "NDArray", "warmup()", "compute()", "cleanup()"],
    web: ["Model.fromPath", "Model.fromBuffer", "fetch(", "NDArray", "warmup()", "compute()"],
} as const;

describe("package READMEs", () => {
    for (const [directory, requiredTerms] of Object.entries(packageRequirements)) {
        it(`documents the ${directory} package API`, () => {
            const packageDirectory = path.join(packagesDirectory, directory);
            const manifest = JSON.parse(
                readFileSync(path.join(packageDirectory, "package.json"), "utf8")
            ) as { name: string; version: string };
            const readme = readFileSync(path.join(packageDirectory, "README.md"), "utf8");

            expect(manifest.version).toBe("0.1.0");
            expect(readme).toContain(`# \`${manifest.name}\``);
            expect(readme).toContain(`npm install ${manifest.name}`);
            expect(readme).toContain("../../../../spec/");
            for (const term of requiredTerms) {
                expect(readme).toContain(term);
            }
        });
    }
});
