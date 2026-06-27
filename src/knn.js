// k-Nearest-Neighbours classification.
// Extracted from the interactive k-NN demo: distance metric + k nearest +
// (optionally distance-weighted) majority vote.

/**
 * Euclidean (L2) distance between two points {x,y}.
 */
export function euclidean(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Manhattan (L1) distance between two points {x,y}.
 */
export function manhattan(a, b) {
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
}

/**
 * Classify a query point against labelled training points by majority vote
 * of its k nearest neighbours.
 *
 * @param {{x:number,y:number,c:number}[]} train labelled points (c in {0,1})
 * @param {{x:number,y:number}} query point to classify
 * @param {object} [opts]
 * @param {number} [opts.k=1] number of neighbours
 * @param {'l2'|'l1'} [opts.metric='l2'] distance metric
 * @param {'uniform'|'inverse'} [opts.weight='uniform'] vote weighting
 * @returns {number} predicted class label, or -1 if train is empty
 */
export function knnPredict(train, query, opts = {}) {
  const { k = 1, metric = 'l2', weight = 'uniform' } = opts;
  if (!train.length) return -1;
  const dist = metric === 'l1' ? manhattan : euclidean;
  const neighbours = train
    .map((p) => ({ d: dist(p, query), c: p.c }))
    .sort((a, b) => a.d - b.d)
    .slice(0, Math.max(1, k));
  // Tally weighted votes per class.
  const votes = new Map();
  for (const n of neighbours) {
    const w = weight === 'inverse' ? 1 / (n.d + 1e-6) : 1;
    votes.set(n.c, (votes.get(n.c) || 0) + w);
  }
  // Argmax; ties broken toward the smaller class label for determinism.
  let best = -1, bestW = -Infinity;
  for (const [c, w] of [...votes].sort((a, b) => a[0] - b[0])) {
    if (w > bestW) { bestW = w; best = c; }
  }
  return best;
}
