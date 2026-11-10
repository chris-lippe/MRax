import { test, expect } from '@playwright/test';
import { ImagePlaneCache } from '../src/imageCache';

test('plane cache bounds bytes, evicts least recently used planes and preserves precision', () => {
  const cache = new ImagePlaneCache(32);
  const key = (frame: number) => JSON.stringify(['study', frame]);
  const first = new Float64Array([Math.PI, NaN]);
  const second = new Float64Array([Infinity, -0]);
  const third = new Float64Array([1e-100, 1e100]);
  cache.set(key(0), first);
  cache.set(key(1), second);
  expect(cache.get(key(0))).toBe(first);
  cache.set(key(2), third);
  expect(cache.bytes).toBe(32);
  expect(cache.get(key(1))).toBeUndefined();
  expect(cache.get(key(0))).toBe(first);
  expect(cache.get(key(2))).toBe(third);
  expect(Number.isNaN(cache.get(key(0))![1])).toBe(true);
});

test('replacing a cache key cannot inflate its byte count or retain oversized planes', () => {
  const cache = new ImagePlaneCache(32);
  const key = JSON.stringify(['study', 0]);
  cache.set(key, new Float32Array(4));
  cache.set(key, new Float32Array(4));
  expect(cache.bytes).toBe(16);
  cache.set(key, new Float32Array(20));
  expect(cache.bytes).toBe(0);
  expect(cache.get(key)).toBeUndefined();
});

test('closing a workspace releases its planes without clearing another workspace', () => {
  const cache = new ImagePlaneCache(64);
  const first = JSON.stringify(['first', 0]);
  const second = JSON.stringify(['second', 0]);
  const retained = new Float32Array([1, 2]);
  cache.set(first, new Float32Array(4));
  cache.set(second, retained);
  cache.clearWorkspace('first');
  expect(cache.get(first)).toBeUndefined();
  expect(cache.get(second)).toBe(retained);
  expect(cache.bytes).toBe(retained.byteLength);
});

test('tiny image planes cannot create an unbounded number of cache entries', () => {
  const cache = new ImagePlaneCache();
  for (let frame = 0; frame < 129; frame++)
    cache.set(JSON.stringify(['study', frame]), new Float32Array(1));
  expect(cache.get(JSON.stringify(['study', 0]))).toBeUndefined();
  expect(cache.get(JSON.stringify(['study', 128]))).toBeDefined();
  expect(cache.bytes).toBe(128 * 4);
});

test('shared packet buffers are budgeted once and stay charged until every view is removed', () => {
  const cache = new ImagePlaneCache(32);
  const packet = new ArrayBuffer(32);
  const first = JSON.stringify(['first', 0]);
  const second = JSON.stringify(['second', 0]);
  cache.set(first, new Float32Array(packet, 0, 4));
  cache.set(second, new Float32Array(packet, 16, 4));
  expect(cache.bytes).toBe(32);
  cache.clearWorkspace('first');
  expect(cache.bytes).toBe(32);
  expect(cache.get(second)).toBeDefined();
  cache.clearWorkspace('second');
  expect(cache.bytes).toBe(0);
  const oversized = new Float32Array(new ArrayBuffer(64), 0, 1);
  cache.set(first, oversized);
  expect(cache.get(first)).toBeUndefined();
  expect(cache.bytes).toBe(0);
});
