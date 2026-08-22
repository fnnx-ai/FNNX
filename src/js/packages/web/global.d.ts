import type {
    Model as FnnxModel,
    NDArray as FnnxNDArray,
    TarExtractor as FnnxTarExtractor,
} from "./src/index.ts";

declare global {
  interface Window {
        Model: typeof FnnxModel;
        NDArray: typeof FnnxNDArray;
        TarExtractor: typeof FnnxTarExtractor;
        testResults?: Record<string, any>;
    }
}

export {};
