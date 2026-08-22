import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const modelTarPath = path.join(__dirname, '../../../../../python/tests/models/onnx_pipeline.fnnx.tar');

test.describe('TarExtractor E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    const html = `
      <!DOCTYPE html>
      <html>
      <head>
        <script type="importmap">
          {"imports":{"@fnnx-ai/common":"/common/index.js"}}
        </script>
      </head>
      <body>
        <script type="module">
          import { TarExtractor } from '/dist/tar.js';
          window.TarExtractor = TarExtractor;
        </script>
      </body>
      </html>
    `;

    await page.route('**/test.html', route => {
      route.fulfill({ status: 200, contentType: 'text/html', body: html });
    });

    await page.route('**/model.tar', route => {
      const tarBuffer = fs.readFileSync(modelTarPath);
      route.fulfill({ status: 200, contentType: 'application/x-tar', body: tarBuffer });
    });

    await page.route('**/dist/**', route => {
      const url = new URL(route.request().url());
      const filePath = path.join(__dirname, '../../dist', url.pathname.split('/dist/')[1]);
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath);
        route.fulfill({ status: 200, contentType: 'application/javascript', body: content });
      } else {
        route.fulfill({ status: 404, body: 'Not found' });
      }
    });

    await page.route('**/common/**', route => {
      const url = new URL(route.request().url());
      const relativePath = url.pathname.replace(/^\/common\//, '');
      const filePath = path.join(
        __dirname,
        '../../../common/dist',
        relativePath
      );
      if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
        route.fulfill({
          status: 200,
          contentType: 'application/javascript',
          body: fs.readFileSync(filePath),
        });
      } else {
        route.fulfill({ status: 404, body: 'Not found' });
      }
    });

    await page.goto('http://localhost/test.html');
  });

  test('should extract files from TAR archive', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const response = await fetch('/model.tar');
      const arrayBuffer = await response.arrayBuffer();
      const extractor = new window.TarExtractor(arrayBuffer);
      const files = extractor.extract();
      return { success: files.length > 0, files: files.map((f: { relpath: any; }) => f.relpath) };
    });
    expect(result.success).toBe(true);
    expect(result.files).toContain('manifest.json');
  });

  test('should scan TAR archive without extracting', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const response = await fetch('/model.tar');
      const arrayBuffer = await response.arrayBuffer();
      const extractor = new window.TarExtractor(arrayBuffer);
      const scanResults = extractor.scan();
      return { success: scanResults.size > 0, files: Array.from(scanResults.keys()) };
    });
    expect(result.success).toBe(true);
    expect(result.files).toContain('manifest.json');
  });

  test('should correctly parse JSON files from TAR', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const response = await fetch('/model.tar');
      const arrayBuffer = await response.arrayBuffer();
      const extractor = new window.TarExtractor(arrayBuffer);
      const files = extractor.extract();
      const manifestFile = files.find((f: { relpath: string; }) => f.relpath === 'manifest.json');
      if (!manifestFile || !manifestFile.content) throw new Error('manifest.json missing');
      const manifest = JSON.parse(new TextDecoder().decode(manifestFile.content));
      return { success: true, keys: Object.keys(manifest) };
    });
    expect(result.success).toBe(true);
    expect(result.keys.length).toBeGreaterThan(0);
  });

  test('should handle empty TAR archive', async ({ page }) => {
    await page.route('**/empty.tar', route => {
      route.fulfill({ status: 200, contentType: 'application/x-tar', body: Buffer.alloc(1024) });
    });

    const result = await page.evaluate(async () => {
      const response = await fetch('/empty.tar');
      const arrayBuffer = await response.arrayBuffer();
      const extractor = new window.TarExtractor(arrayBuffer);
      const files = extractor.extract();
      return { success: true, fileCount: files.length };
    });
    expect(result.fileCount).toBe(0);
  });

test('should throw error for invalid TAR', async ({ page }) => {
  await page.route('**/invalid.tar', route => {
    route.fulfill({
      status: 200,
      contentType: 'application/x-tar',
      body: Buffer.from('Not a TAR file'),
    });
  });

  const result = await page.evaluate(async () => {
    try {
      const response = await fetch('/invalid.tar');
      const arrayBuffer = await response.arrayBuffer();
      const extractor = new window.TarExtractor(arrayBuffer);
      extractor.extract();
      return { success: true };
    } catch (e) {
      const error = e as Error; 
      return { success: false, message: error.message };
    }
  });

  expect(result.success).toBe(false);
});

});
