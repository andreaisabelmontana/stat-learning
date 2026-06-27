// Seedable Lehmer / Park-Miller LCG — the same generator the interactive
// page uses, exposed here so models that draw random numbers (bootstrap,
// k-means++ seeding) are reproducible and testable.

/**
 * Create a deterministic uniform [0,1) generator from an integer seed.
 * @param {number} seed positive integer seed
 * @returns {() => number} function returning a float in [0,1)
 */
export function makeRng(seed = 1234567) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 48271) % 2147483647;
    return s / 2147483647;
  };
}
