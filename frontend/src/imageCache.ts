import type { Pixels } from './api';

export class ImagePlaneCache {
  private readonly entries = new Map<string, Pixels>();
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
    if (old) {
      this.entries.delete(key);
      this.used -= old.byteLength;
    }
    if (value.byteLength > this.limit) return;
    this.entries.set(key, value);
    this.used += value.byteLength;
    while (this.used > this.limit || this.entries.size > 128) {
      const oldest = this.entries.keys().next().value!;
      this.used -= this.entries.get(oldest)!.byteLength;
      this.entries.delete(oldest);
    }
  }
  clearWorkspace(id: string) {
    for (const [key, value] of this.entries) {
      if (JSON.parse(key)[0] === id) {
        this.entries.delete(key);
        this.used -= value.byteLength;
      }
    }
  }
}
