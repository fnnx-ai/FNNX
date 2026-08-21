export type JsonValue =
    | string
    | number
    | boolean
    | null
    | { [key: string]: JsonValue }
    | JsonValue[];
export type JsonObject = { [key: string]: JsonValue };
export type JsonPatchOp = { op: string; path: string; value?: JsonValue };
export type JsonPatch = JsonPatchOp[];

function decodePointerToken(token: string): string {
    return token.replace(/~1/g, "/").replace(/~0/g, "~");
}

function splitPointer(path: string): string[] {
    if (path === "") {
        throw new Error("Empty JSON Pointer path is not supported");
    }
    if (!path.startsWith("/")) {
        throw new Error(
            `Only absolute JSON Pointer paths are supported, got: '${path}'`
        );
    }
    const rawTokens = path.slice(1).split("/");
    return rawTokens.filter((t) => t !== "").map(decodePointerToken);
}

function parseIndex(token: string, maxLen: number): number {
    const idx = parseInt(token, 10);
    if (isNaN(idx)) {
        throw new Error(`Array index must be an integer, got '${token}'`);
    }
    if (idx < 0 || idx >= maxLen) {
        throw new RangeError(
            `Array index ${idx} out of range (len=${maxLen})`
        );
    }
    return idx;
}

function parseIndexForAdd(token: string, maxLen: number): number {
    const idx = parseInt(token, 10);
    if (isNaN(idx)) {
        throw new Error(`Array index must be an integer, got '${token}'`);
    }
    if (idx < 0 || idx > maxLen) {
        throw new RangeError(
            `Array index ${idx} out of range for add (len=${maxLen})`
        );
    }
    return idx;
}

function traverseToParent(doc: JsonValue, path: string): [JsonValue, string] {
    const tokens = splitPointer(path);
    if (tokens.length === 0) {
        throw new Error(
            `Path '${path}' does not point to a child of the root`
        );
    }

    let parent: JsonValue = doc;
    for (let i = 0; i < tokens.length - 1; i++) {
        const token = tokens[i];
        if (Array.isArray(parent)) {
            const idx = parseIndex(token, parent.length);
            parent = parent[idx];
        } else if (parent !== null && typeof parent === "object") {
            if (!(token in parent)) {
                throw new Error(
                    `Path segment '${token}' not found while traversing '${path}'`
                );
            }
            parent = (parent as JsonObject)[token];
        } else {
            throw new TypeError(
                `Cannot traverse into non-container type at segment '${token}'`
            );
        }
    }
    return [parent, tokens[tokens.length - 1]];
}

function opAdd(doc: JsonValue, path: string, value: JsonValue): void {
    const [parent, token] = traverseToParent(doc, path);

    if (Array.isArray(parent)) {
        if (token === "-") {
            parent.push(value);
            return;
        }
        const idx = parseIndexForAdd(token, parent.length);
        parent.splice(idx, 0, value);
        return;
    }

    if (parent !== null && typeof parent === "object") {
        (parent as JsonObject)[token] = value;
        return;
    }

    throw new TypeError(
        `Cannot apply 'add' at '${path}': parent is not a container`
    );
}

function opReplace(doc: JsonValue, path: string, value: JsonValue): void {
    const [parent, token] = traverseToParent(doc, path);

    if (Array.isArray(parent)) {
        const idx = parseIndex(token, parent.length);
        parent[idx] = value;
        return;
    }

    if (parent !== null && typeof parent === "object") {
        if (!(token in (parent as JsonObject))) {
            throw new Error(
                `Cannot 'replace' non-existent member '${token}' at '${path}'`
            );
        }
        (parent as JsonObject)[token] = value;
        return;
    }

    throw new TypeError(
        `Cannot apply 'replace' at '${path}': parent is not a container`
    );
}

function applyPatch(document: JsonValue, patch: JsonPatch): void {
    for (const op of patch) {
        const opType = op.op;
        const path = op.path;
        if (opType === undefined || path === undefined) {
            throw new Error(`Invalid JSON Patch operation: ${JSON.stringify(op)}`);
        }

        if (opType === "add") {
            opAdd(document, path, op.value as JsonValue);
        } else if (opType === "replace") {
            opReplace(document, path, op.value as JsonValue);
        } else {
            throw new Error(
                `Unsupported JSON Patch op: '${opType}' (only 'add' and 'replace' are allowed)`
            );
        }
    }
}

export function applyPatches(
    document: JsonObject,
    patchDocuments: JsonPatch[]
): JsonObject {
    const result: JsonObject = JSON.parse(JSON.stringify(document));
    for (const patch of patchDocuments) {
        applyPatch(result, patch);
    }
    return result;
}
