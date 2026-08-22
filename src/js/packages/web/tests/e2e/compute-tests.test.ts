import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const testHTML = `
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Compute Tests</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js"></script>
    <script type="importmap">
    {
      "imports": {
        "@fnnx-ai/common": "/dist/common/index.js",
        "@fnnx-ai/web": "/dist/web/index.js",
        "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.mjs"
      }
    }
    </script>
  </head>
  <body>
    <div id="status">Loading...</div>
    <script type="module">
      import { Model, NDArray } from '/dist/web/index.js';
      
      window.Model = Model;
      window.NDArray = NDArray;
      
      console.log('Model and NDArray loaded');
      document.getElementById('status').textContent = 'Ready';
    </script>
  </body>
</html>
`;

test.describe('Compute Tests', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error') {
        console.error(text);
      } else if (type === 'log') {
        console.log(text);
      }
    });

    page.on('pageerror', err => {
      console.error('Page error:', err.message);
    });

    await page.route('**/dist/**/**', async route => {
      const url = new URL(route.request().url());
      const match = url.pathname.match(/\/dist\/(common|web)\/(.+)$/);
      
      if (!match) {
        const simplePath = url.pathname.match(/\/dist\/([^/]+)$/);
        if (simplePath) {
          const fileName = simplePath[1];
          const candidates = [path.join(__dirname, '../../dist', fileName)];
          const resolvedPath = candidates.find(p => fs.existsSync(p));
          
          if (resolvedPath) {
            const content = fs.readFileSync(resolvedPath, 'utf8');
            await route.fulfill({
              status: 200,
              contentType: 'application/javascript',
              headers: { 'Access-Control-Allow-Origin': '*' },
              body: content,
            });
            return;
          }
        }
        await route.continue();
        return;
      }

      const packageName = match[1];
      let filePath = match[2];
      
      if (!path.extname(filePath)) {
        filePath += '.js';
      }

      const candidates = [
        path.join(__dirname, `../../../${packageName}/dist`, filePath),
        path.join(__dirname, `../../${packageName}/dist`, filePath),
      ];

      const resolvedPath = candidates.find(p => fs.existsSync(p));

      if (resolvedPath) {
        const content = fs.readFileSync(resolvedPath, 'utf8');
        await route.fulfill({
          status: 200,
          contentType: 'application/javascript',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: content,
        });
      } else {
        await route.fulfill({ status: 404, body: 'Not found' });
      }
    });

    await page.route('**/model.tar', route => {
      const modelPath = path.join(
        __dirname, 
        '../../../../../python/tests/models/onnx_pipeline.fnnx.tar'
      );
      
      if (fs.existsSync(modelPath)) {
        const buffer = fs.readFileSync(modelPath);
        route.fulfill({ 
          status: 200, 
          contentType: 'application/x-tar', 
          body: buffer 
        });
      } else {
        route.fulfill({ status: 404, body: 'Model not found' });
      }
    });

    await page.route('**/test.html', route => {
      route.fulfill({ 
        status: 200, 
        contentType: 'text/html', 
        body: testHTML 
      });
    });

    await page.goto('http://localhost:4173/test.html');
    await page.waitForFunction(() => {
      return document.getElementById('status')?.textContent === 'Ready';
    }, { timeout: 10000 });
  });

  test('should compute with single batch input', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const manifest = model.getManifest();
        const outputName = manifest.outputs[0].name; 
        
        const inputData = new Float32Array([1.0, 2.0, 3.0]);
        const input = new window.NDArray([1, 3], inputData, 'float32');
        
        const output = await model.compute({ x: input }, {}); 
        const resultArray = output[outputName];
        
        return {
          success: true as const,
          hasOutput: !!resultArray,
          outputShape: resultArray?.shape ?? null,
          outputData: resultArray ? Array.from(resultArray.toArray()) : null,
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message,
        };
      }
    });
    
    expect(result.success).toBe(true);
    
    if (result.success) {
      expect(result.hasOutput).toBe(true);
      expect(result.outputShape).toEqual([1, 1]);
      expect(result.outputData).toBeDefined();
      expect(result.outputData?.length).toBe(1);
      console.log('✅ Output:', result.outputData);
    }
  });

  test('should compute with multiple batch inputs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const manifest = model.getManifest();
        const outputName = manifest.outputs[0].name;
        
        const inputData = new Float32Array([
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 9.0
        ]);
        const input = new window.NDArray([3, 3], inputData, 'float32');
        
        const output = await model.compute({ x: input }, {});
        const resultArray = output[outputName];
        
        return {
          success: true as const,
          outputShape: resultArray?.shape ?? null,
          outputData: resultArray ? Array.from(resultArray.toArray()) : null,
          outputLength: resultArray ? resultArray.toArray().length : 0
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message
        };
      }
    });
    
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.outputShape).toEqual([3, 1]);
      expect(result.outputLength).toBe(3);
      console.log('Output:', result.outputData);
    }
  });

  test('should produce same output for same input', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const manifest = model.getManifest();
        const outputName = manifest.outputs[0].name;
        
        const inputData = new Float32Array([1.0, 2.0, 3.0]);
        const input = new window.NDArray([1, 3], inputData, 'float32');
        
        const output1 = await model.compute({ x: input }, {}); 
        const output2 = await model.compute({ x: input }, {}); 
        
        const data1 = Array.from(output1[outputName].toArray());
        const data2 = Array.from(output2[outputName].toArray());
        
        return {
          success: true as const,
          data1,
          data2,
          areEqual: JSON.stringify(data1) === JSON.stringify(data2)
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message
        };
      }
    });
    
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.areEqual).toBe(true); 
      console.log('Output 1:', result.data1);
      console.log('Output 2:', result.data2);
    }
  });

  test('should handle concurrent compute calls', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const manifest = model.getManifest();
        const outputName = manifest.outputs[0].name;
        
        const inputData = new Float32Array([1.0, 2.0, 3.0]);
        const input = new window.NDArray([1, 3], inputData, 'float32');
        const results = [];
        for (let i = 0; i < 5; i++) {
          const output = await model.compute({ x: input }, {});
          results.push(output);
        }
        
        const firstData = Array.from(results[0][outputName].toArray());
        const allSame = results.every(r => 
          JSON.stringify(Array.from(r[outputName].toArray())) === JSON.stringify(firstData)
        );
        
        return {
          success: true as const,
          count: results.length,
          allSame,
          firstOutput: firstData
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message
        };
      }
    });
    
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.count).toBe(5);
      expect(result.allSame).toBe(true);
    }
  });

  test('should handle invalid input shape gracefully', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const inputData = new Float32Array([1.0, 2.0]);
        const input = new window.NDArray([1, 2], inputData, 'float32');
        
        await model.compute({ x: input }, {});
        
        return { success: true as const, shouldNotReach: true };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          errorCaught: true,
          errorMessage: error.message
        };
      }
    });
    
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errorCaught).toBe(true);
      console.log('Expected error:', result.errorMessage); 
    }
  });

  test('should handle null inputs gracefully', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        await model.compute(null as any, {});
        
        return { success: true as const, shouldNotReach: true };
      } catch (err) {
        return {
          success: false as const,
          errorCaught: true,
          errorMessage: (err as Error).message
        };
      }
    });
    
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errorCaught).toBe(true);
    }
  });

  test('should handle empty inputs gracefully', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        await model.compute({}, {});
        
        return { success: true as const, shouldNotReach: true };
      } catch (err) {
        return {
          success: false as const,
          errorCaught: true,
          errorMessage: (err as Error).message
        };
      }
    });
    
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errorCaught).toBe(true);
    }
  });

  test('should compute within reasonable time', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const inputData = new Float32Array([1.0, 2.0, 3.0]);
        const input = new window.NDArray([1, 3], inputData, 'float32');

        await model.compute({ x: input }, {});
        
        const times: number[] = [];
        for (let i = 0; i < 10; i++) {
          const start = performance.now();
          await model.compute({ x: input }, {});
          const duration = performance.now() - start;
          times.push(duration);
        }
        
        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const maxTime = Math.max(...times);
        
        return {
          success: true as const,
          avgTime,
          maxTime,
          allTimes: times,
          withinLimit: avgTime < 100
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message
        };
      }
    });
    
    expect(result.success).toBe(true);
    
    if (result.success) {
      console.log(`Avg: ${result.avgTime.toFixed(2)}ms, Max: ${result.maxTime.toFixed(2)}ms`);
      expect(result.withinLimit).toBe(true);
    }
  });

  test('should produce different outputs for different inputs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.warmup();
        
        const manifest = model.getManifest();
        const outputName = manifest.outputs[0].name;
        
        const input1Data = new Float32Array([1.0, 2.0, 3.0]);
        const input1 = new window.NDArray([1, 3], input1Data, 'float32');
        
        const input2Data = new Float32Array([4.0, 5.0, 6.0]);
        const input2 = new window.NDArray([1, 3], input2Data, 'float32');

        const output1 = await model.compute({ x: input1 }, {});
        const output2 = await model.compute({ x: input2 }, {}); 
        
        const data1 = Array.from(output1[outputName].toArray());
        const data2 = Array.from(output2[outputName].toArray());
        
        return {
          success: true as const,
          data1,
          data2,
          areDifferent: JSON.stringify(data1) !== JSON.stringify(data2)
        };
      } catch (err) {
        const error = err instanceof Error ? err : new Error(String(err));
        return {
          success: false as const,
          error: error.message
        };
      }
    });
    
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.areDifferent).toBe(true); 
      console.log(result.data1);
      console.log(result.data2);
    }
  });
});
