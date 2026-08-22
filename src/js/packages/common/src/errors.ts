export class FNNXError extends Error {
    constructor(message: string) {
        super(message);
        this.name = new.target.name;
    }
}

export class ArtifactFileError extends FNNXError {
    constructor(
        message: string,
        public readonly filePath: string
    ) {
        super(message);
    }
}

export class MissingArtifactFileError extends ArtifactFileError {
    constructor(filePath: string) {
        super(`Artifact file \`${filePath}\` was not found`, filePath);
    }
}

export class InvalidArtifactFileError extends ArtifactFileError {
    constructor(filePath: string, reason: string) {
        super(`Artifact file \`${filePath}\` is invalid: ${reason}`, filePath);
    }
}

export class UnsupportedVariantError extends FNNXError {
    constructor(public readonly variant: string) {
        super(`Unsupported variant \`${variant}\``);
    }
}

export class UnsupportedOperationError extends FNNXError {
    constructor(
        public readonly opType: string,
        public readonly opInstanceId: string
    ) {
        super(`Unsupported operation \`${opType}\` for op instance \`${opInstanceId}\``);
    }
}

export class OperationInstanceError extends FNNXError {
    constructor(
        public readonly opInstanceId: string,
        reason: string
    ) {
        super(`Op instance \`${opInstanceId}\` failed: ${reason}`);
    }
}

export class InvalidOperationDeclarationError extends OperationInstanceError {
    constructor(
        opInstanceId: string,
        public readonly field: string,
        reason: string
    ) {
        super(opInstanceId, `invalid \`${field}\` declaration: ${reason}`);
    }
}

export class UnsupportedONNXDomainError extends OperationInstanceError {
    constructor(
        opInstanceId: string,
        public readonly domain: string
    ) {
        super(opInstanceId, `unsupported ONNX domain \`${domain}\``);
    }
}

export class UnsupportedExternalDataError extends OperationInstanceError {
    constructor(opInstanceId: string) {
        super(opInstanceId, "the backend does not support ONNX external data");
    }
}

export class ModelNotWarmedUpError extends FNNXError {
    constructor() {
        super("Model is not initialized; call warmup() before compute()");
    }
}

export class InvalidTarError extends FNNXError {}

export class UnsafeTarMemberError extends ArtifactFileError {
    constructor(filePath: string, reason: string) {
        super(`Unsafe tar member \`${filePath}\`: ${reason}`, filePath);
    }
}
