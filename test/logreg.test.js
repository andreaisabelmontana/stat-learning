import test from 'node:test';
import assert from 'node:assert/strict';
import { sigmoid, trainStep, train, predict, accuracy } from '../src/logreg.js';
import { makeRng } from '../src/rng.js';

test('sigmoid returns hand-checked values', () => {
  assert.equal(sigmoid(0), 0.5);
  assert.ok(sigmoid(100) > 0.999999);
  assert.ok(sigmoid(-100) < 0.000001);
  // Symmetry: sigmoid(-z) = 1 - sigmoid(z).
  assert.ok(Math.abs(sigmoid(-2) - (1 - sigmoid(2))) < 1e-12);
});

// Linearly separable: class 0 has negative x, class 1 positive x.
function separable(rng) {
  const pts = [];
  for (let i = 0; i < 40; i++) {
    pts.push({ x: -1 + (rng() - 0.5) * 0.4, y: (rng() - 0.5), c: 0 });
    pts.push({ x: 1 + (rng() - 0.5) * 0.4, y: (rng() - 0.5), c: 1 });
  }
  return pts;
}

test('learns a linearly separable problem (training accuracy → high)', () => {
  const pts = separable(makeRng(99));
  const accBefore = accuracy([0, 0, 0], pts); // 0.5-ish at init
  const model = train(pts, { epochs: 800, lr: 0.5 });
  assert.ok(model.accuracy >= 0.98, `accuracy ${model.accuracy}`);
  assert.ok(model.accuracy > accBefore);
  // Loss should be small after convergence.
  assert.ok(model.loss < 0.2, `loss ${model.loss}`);
});

test('a single gradient step reduces the loss', () => {
  const pts = separable(makeRng(5));
  const w0 = [0, 0, 0];
  const s1 = trainStep(w0, pts, { lr: 0.5 });
  const s2 = trainStep(s1.weights, pts, { lr: 0.5 });
  assert.ok(s2.loss < s1.loss);
});

test('trained model predicts the correct side of the boundary', () => {
  const pts = separable(makeRng(11));
  const model = train(pts, { epochs: 800, lr: 0.5 });
  assert.equal(predict(model.weights, { x: -1, y: 0 }), 0);
  assert.equal(predict(model.weights, { x: 1, y: 0 }), 1);
});
