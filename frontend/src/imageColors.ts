export const palettes: Record<string, readonly (readonly number[])[]> = {
  viridis: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ],
  diverging: [
    [40, 75, 190],
    [240, 240, 240],
    [190, 35, 50],
  ],
  hot: [
    [0, 0, 0],
    [210, 0, 0],
    [255, 210, 0],
    [255, 255, 255],
  ],
  gray: [
    [0, 0, 0],
    [255, 255, 255],
  ],
};

export function writeColor(
  pixels: Uint8ClampedArray,
  offset: number,
  value: number,
  anchors: readonly (readonly number[])[],
  alpha = 1,
) {
  const position = Math.max(0, Math.min(1, value)) * (anchors.length - 1);
  const index = Math.min(anchors.length - 2, Math.floor(position));
  for (let channel = 0; channel < 3; channel++) {
    const color = Math.round(
      anchors[index][channel] +
        (anchors[index + 1][channel] - anchors[index][channel]) * (position - index),
    );
    pixels[offset + channel] =
      alpha === 1 ? color : Math.round(pixels[offset + channel] * (1 - alpha) + color * alpha);
  }
}
