import test from 'node:test';
import assert from 'node:assert/strict';
import { assign, updateCentroids, wss, kmeans } from '../src/kmeans.js';
import { makeRng } from '../src/rng.js';

test('assign maps points to the nearest centroid', () => {
  const pts = [{ x: 0, y: 0 }, { x: 10, y: 10 }];
  const cents = [{ x: 0, y: 0 }, { x: 10, y: 10 }];
  assert.deepEqual(assign(pts, cents), [0, 1]);
  // Swap centroids → labels flip.
  assert.deepEqual(assign(pts, [{ x: 10, y: 10 }, { x: 0, y: 0 }]), [1, 0]);
});

test('updateCentroids returns the mean of each cluster', () => {
  const pts = [{ x: 0, y: 0 }, { x: 2, y: 0 }, { x: 10, y: 10 }, { x: 12, y: 10 }];
  const labels = [0, 0, 1, 1];
  const prev = [{ x: 0, y: 0 }, { x: 0, y: 0 }];
  const cents = updateCentroids(pts, labels, 2, prev);
  assert.deepEqual(cents[0], { x: 1, y: 0 });
  assert.deepEqual(cents[1], { x: 11, y: 10 });
});

test('wss is zero when every point coincides with its centroid', () => {
  const pts = [{ x: 1, y: 1 }, { x: 2, y: 2 }];
  assert.equal(wss(pts, [0, 1], [{ x: 1, y: 1 }, { x: 2, y: 2 }]), 0);
});

// Build three well-separated blobs with a seeded RNG.
function blobs(rng) {
  const centres = [{ x: 0, y: 0 }, { x: 30, y: 0 }, { x: 15, y: 30 }];
  const pts = [];
  const truth = [];
  centres.forEach((c, ci) => {
    for (let i = 0; i < 30; i++) {
      pts.push({ x: c.x + (rng() - 0.5) * 4, y: c.y + (rng() - 0.5) * 4 });
      truth.push(ci);
    }
  });
  return { pts, truth, centres };
}

test('kmeans converges to the known clusters on separable blobs', () => {
  const rng = makeRng(42);
  const { pts, truth, centres } = blobs(rng);
  const { centroids, labels, iterations } = kmeans(pts, 3, { rng: makeRng(7) });

  assert.ok(iterations >= 1);

  // Each recovered centroid should sit close to one true centre.
  for (const c of centres) {
    const near = centroids.some(
      (k) => Math.hypot(k.x - c.x, k.y - c.y) < 3
    );
    assert.ok(near, `no centroid near (${c.x},${c.y})`);
  }

  // Points sharing a true cluster should share a recovered label (purity).
  for (let g = 0; g < 3; g++) {
    const labelsInGroup = new Set(
      truth.map((t, i) => (t === g ? labels[i] : null)).filter((v) => v !== null)
    );
    assert.equal(labelsInGroup.size, 1, `true cluster ${g} split across labels`);
  }
});

test('kmeans objective (wss) does not increase across the run', () => {
  const rng = makeRng(123);
  const { pts } = blobs(rng);
  // One Lloyd iteration vs many: more iterations never have higher wss.
  const one = kmeans(pts, 3, { rng: makeRng(1), maxIter: 1 });
  const many = kmeans(pts, 3, { rng: makeRng(1), maxIter: 100 });
  assert.ok(many.wss <= one.wss + 1e-9);
});
