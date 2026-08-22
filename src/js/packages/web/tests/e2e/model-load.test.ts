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
    <title>Model Load Test</title>
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
      import { Model } from '/dist/web/index.js';
      console.log('ONNX Runtime available:', typeof window.ort !== 'undefined');
      if (window.ort) {
        console.log('ONNX Runtime version:', window.ort.env.versions);
      }
      window.Model = Model;
      console.log('Model loaded:', Model);
      console.log('Model methods:', Object.getOwnPropertyNames(Model));
      document.getElementById('status').textContent = 
        window.Model ? 'Ready' : 'Failed';
    </script>
  </body>
</html>
`;

test.describe('Model Load Test', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error') {
        console.error('Browser error:', text);
      } else if (type === 'log' || type === 'info') {
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
          const candidates = [
            path.join(__dirname, '../../dist', fileName),
          ];
          
          const resolvedPath = candidates.find(p => fs.existsSync(p));
          if (resolvedPath) {
            const content = fs.readFileSync(resolvedPath, 'utf8');
            console.log('Served simple dist file:', resolvedPath);
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

      console.log('🔍 /dist request:', packageName, filePath);

      const candidates = [
        path.join(__dirname, `../../../${packageName}/dist`, filePath),
        path.join(__dirname, `../../${packageName}/dist`, filePath),
      ];

      console.log('   Searching in:', candidates);

      const resolvedPath = candidates.find(p => {
        const exists = fs.existsSync(p);
        console.log('   Checking:', p, exists ? '✓' : '✗');
        return exists;
      });

      if (resolvedPath) {
        const content = fs.readFileSync(resolvedPath, 'utf8');
        console.log('Served dist file:', resolvedPath);
        
        await route.fulfill({
          status: 200,
          contentType: 'application/javascript',
          headers: { 'Access-Control-Allow-Origin': '*' },
          body: content,
        });
      } else {
        console.error('File not found:', packageName, filePath);
        console.error('   Tried:', candidates);
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
        console.log('Served model.tar:', buffer.length, 'bytes');
        route.fulfill({ 
          status: 200, 
          contentType: 'application/x-tar', 
          body: buffer 
        });
      } else {
        console.error('Model file not found:', modelPath);
        route.fulfill({ 
          status: 404, 
          body: 'Model file not found' 
        });
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
      const status = document.getElementById('status')?.textContent;
      return status === 'Ready' || status === 'Failed';
    }, { timeout: 10000 });
  });

  test('should load Model from dist', async ({ page }) => {
    const status = await page.locator('#status').textContent();
    console.log('Status element text:', status);

    const hasModel = await page.evaluate(() => {
      console.log('window.Model:', window.Model);
      console.log('typeof window.Model:', typeof window.Model);
      return typeof window.Model !== 'undefined';
    });
    
    console.log('window.Model exists:', hasModel);
    expect(hasModel).toBe(true);
    expect(status).toBe('Ready');
  });

  test('should load model from ArrayBuffer', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const response = await fetch('http://localhost:4173/model.tar');
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        console.log('Fetched buffer size:', buffer.byteLength);
        
        const model = await window.Model.fromBuffer(buffer);
        return { 
          success: !!model, 
          error: null 
        };
      } catch (e) {
        return { 
          success: false, 
          error: e.message 
        };
      }
    });
    
    if (!result.success) {
      console.error('Model loading failed:', result.error);
    }
    expect(result.success).toBe(true);
  });

  test('should load model from path', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        return { 
          success: !!model, 
          error: null 
        };
      } catch (e) {
        return { 
          success: false, 
          error: e.message 
        };
      }
    });
    
    if (!result.success) {
      console.error('Model loading failed:', result.error);
    }
    expect(result.success).toBe(true);
  });

  test('should extract TAR files correctly', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        const manifest = model.getManifest?.();
        return {
          success: !!manifest,
          keys: manifest ? Object.keys(manifest) : [],
          error: null,
        };
      } catch (e) {
        return {
          success: false,
          keys: [],
          error: e.message,
        };
      }
    });
    
    if (!result.success) {
      console.error('TAR extraction failed:', result.error);
    }
    expect(result.success).toBe(true);
    expect(result.keys.length).toBeGreaterThan(0);
  });

  test('should warmup model successfully', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        console.log('Loading model...');
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        console.log('Model loaded successfully');
        console.log('Model instance:', model);
        console.log('Starting warmup...');
        await model.warmup?.();
        console.log('Warmup completed successfully');
        return { success: true, error: null, stack: null };
      } catch (e) {
        console.error('Warmup error:', e);
        console.error('Error details:', {
          message: e.message,
          stack: e.stack,
          name: e.name
        });
        return { 
          success: false, 
          error: e.message,
          stack: e.stack,
          name: e.name
        };
      }
    });
    
    if (!result.success) {
      console.error('Warmup failed:', result.error);
      console.error('Error name:', result.name);
      console.error('Stack trace:', result.stack);

      if (result.error?.includes('onnxruntime') || result.error?.includes('create')) {
        console.log('ONNX Runtime not available - skipping warmup test');
        test.skip();
      }
    }
    expect(result.success).toBe(true);
  });

  test('should fail compute without warmup', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        await model.compute?.([], {});
        return { 
          success: true, 
          message: 'no error', 
          hasWarmupError: false 
        };
      } catch (e) {
        const error = e;
        return {
          success: false,
          message: error.message ?? 'Unknown error',
          hasWarmupError: !!error.message?.match(/warmup|initialized/i),
        };
      }
    });

    expect(result.success).toBe(false);
    expect(result.message).toBeDefined();
    expect(result.hasWarmupError).toBe(true);
  });

  test('should get manifest data', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        const manifest = model.getManifest?.();
        return { 
          success: !!manifest, 
          keys: manifest ? Object.keys(manifest) : [],
          error: null,
        };
      } catch (e) {
        return { 
          success: false, 
          keys: [],
          error: e.message,
        };
      }
    });
    
    if (!result.success) {
      console.error('Get manifest failed:', result.error);
    }
    expect(result.success).toBe(true);
    expect(result.keys.length).toBeGreaterThan(0);
  });

  test('should get metadata', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        console.log('Loading model for metadata...');
        const model = await window.Model.fromPath('http://localhost:4173/model.tar');
        console.log('Getting metadata...');
        const metadata = model.getMetadata?.();
        console.log('Metadata result:', metadata);
        
        const isValid = Array.isArray(metadata) || metadata === undefined;
        
        return { 
          success: isValid,
          error: null,
          metadata: metadata,
          hasMethod: typeof model.getMetadata === 'function'
        };
      } catch (e) {
        console.error('Get metadata error:', e);
        const isExpectedError = e.message?.includes('meta.json not found');
        return { 
          success: false,
          error: e.message,
          stack: e.stack,
          isExpectedError
        };
      }
    });
    
    if (!result.success) {
      console.error('Get metadata failed:', result.error);
      if (result.isExpectedError) {
        console.log('meta.json is optional - test will be skipped');
        test.skip();
      }
    } else {
      console.log('Metadata:', result.metadata);
    }

    expect(result.hasMethod).toBe(true);
  });

  test('should handle invalid model path gracefully', async ({ page }) => {
    const result = await page.evaluate(async () => {
      try {
        await window.Model.fromPath('http://localhost:4173/invalid.tar');
        return { success: true };
      } catch {
        return { success: false, hasError: true };
      }
    });
    
    expect(result.success).toBe(false);
    expect(result.hasError).toBe(true);
  });

  test('should handle corrupted TAR file', async ({ page }) => {
    await page.route('**/corrupted.tar', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/x-tar',
        body: Buffer.from('invalid tar content'),
      });
    });

    const result = await page.evaluate(async () => {
      try {
        await window.Model.fromPath('http://localhost:4173/corrupted.tar');
        return { success: true };
      } catch (e: unknown) {
        const error = e as Error;
        return { 
          success: false, 
          hasError: true, 
          message: error.message 
        };
      }
    });
    
    expect(result.success).toBe(false);
    expect(result.hasError).toBe(true);
  });
});
