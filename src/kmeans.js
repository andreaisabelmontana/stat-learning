// k-means clustering (Lloyd's algorithm) with optional k-means++ seeding.
// Extracted from the interactive k-means demo: assignment step, centroid
// update, within-cluster sum of squares, and the iterate-to-convergence loop.

import { makeRng } from './rng.js';

function dist2(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return dx * dx + dy * dy;
}

/**
 * Assign each point to the index of its nearest centroid.
 * @returns {number[]} cluster index per point
 */
export function assign(points, centroids) {
  return points.map((p) => {
    let best = 0, bestD = Infinity;
    centroids.forEach((c, i) => {
      const d = dist2(c, p);
      if (d < bestD) { bestD = d; best = i; }
    });
    return best;
  });
}

/**
 * Recompute each centroid as the mean of its assigned points. Empty clusters
 * keep their previous centroid.
 */
export function updateCentroids(points, labels, k, prev) {
  const sums = Array.from({ length: k }, () => ({ x: 0, y: 0, n: 0 }));
  labels.forEach((c, i) => {
    sums[c].x += points[i].x;
    sums[c].y += points[i].y;
    sums[c].n++;
  });
  return sums.map((s, i) =>
    s.n > 0 ? { x: s.x / s.n, y: s.y / s.n } : prev[i]
  );
}

/**
 * Within-cluster sum of squared distances (the k-means objective).
 */
export function wss(points, labels, centroids) {
  let total = 0;
  labels.forEach((c, i) => { total += dist2(centroids[c], points[i]); });
  return total;
}

/**
 * k-means++ centroid seeding: pick points with probability proportional to
 * squared distance from the nearest already-chosen centroid.
 */
export function kmeansPlusPlusInit(points, k, rng = makeRng()) {
  if (!points.length) return [];
  const centroids = [{ ...points[Math.floor(rng() * points.length)] }];
  while (centroids.length < k) {
    const d2 = points.map((p) => {
      let m = Infinity;
      for (const c of centroids) m = Math.min(m, dist2(c, p));
      return m;
    });
    const sum = d2.reduce((a, b) => a + b, 0);
    let r = rng() * sum, pick = 0;
    for (let i = 0; i < d2.length; i++) {
      r -= d2[i];
      if (r <= 0) { pick = i; break; }
    }
    centroids.push({ ...points[pick] });
  }
  return centroids.map((c) => ({ x: c.x, y: c.y }));
}

/**
 * Run k-means to convergence.
 *
 * @param {{x:number,y:number}[]} points
 * @param {number} k number of clusters
 * @param {object} [opts]
 * @param {{x:number,y:number}[]} [opts.init] initial centroids; if omitted,
 *        k-means++ seeding is used
 * @param {number} [opts.maxIter=100]
 * @param {() => number} [opts.rng]
 * @returns {{centroids, labels, wss, iterations}}
 */
export function kmeans(points, k, opts = {}) {
  const { maxIter = 100, rng = makeRng() } = opts;
  let centroids = opts.init
    ? opts.init.map((c) => ({ x: c.x, y: c.y }))
    : kmeansPlusPlusInit(points, k, rng);
  let labels = assign(points, centroids);
  let iterations = 0;
  for (let it = 0; it < maxIter; it++) {
    iterations++;
    const next = updateCentroids(points, labels, k, centroids);
    const nextLabels = assign(points, next);
    centroids = next;
    // Converged when no point changes cluster.
    const stable = nextLabels.every((l, i) => l === labels[i]);
    labels = nextLabels;
    if (stable) break;
  }
  return { centroids, labels, wss: wss(points, labels, centroids), iterations };
}
