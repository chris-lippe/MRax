import type { Pixels } from './api';

export class ImagePlaneCache {
  private readonly entries = new Map<string, Pixels>();
  private readonly buffers = new Map<ArrayBufferLike, number>();
  private used = 0;
  constructor(private readonly limit = 64 * 1024 * 1024) {}
  get bytes() {
    return this.used;
  }
  get(key: string) {
    const value = this.entries.get(key);
    if (value) {
      this.entries.delete(key);
      this.entries.set(key, value);
    }
    return value;
  }
  set(key: string, value: Pixels) {
    const old = this.entries.get(key);
    if (old) this.remove(key);
    // A typed-array view retains its entire buffer, including sibling planes.
    // Count shared buffers once, and release their budget only at the last view.
    if (value.buffer.byteLength > this.limit) return;
    this.entries.set(key, value);
    const references = this.buffers.get(value.buffer) ?? 0;
    if (!references) this.used += value.buffer.byteLength;
    this.buffers.set(value.buffer, references + 1);
    while (this.used > this.limit || this.entries.size > 128) {
      const oldest = this.entries.keys().next().value!;
      this.remove(oldest);
    }
  }
  clearWorkspace(id: string) {
    for (const key of this.entries.keys()) {
      if (JSON.parse(key)[0] === id) {
        this.remove(key);
      }
    }
  }
  private remove(key: string) {
    const value = this.entries.get(key)!;
    this.entries.delete(key);
    const references = this.buffers.get(value.buffer)!;
    if (references > 1) this.buffers.set(value.buffer, references - 1);
    else {
      this.buffers.delete(value.buffer);
      this.used -= value.buffer.byteLength;
    }
  }
}
