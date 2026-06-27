import test from 'node:test';
import assert from 'node:assert/strict';
import { euclidean, manhattan, knnPredict } from '../src/knn.js';

test('distance helpers return hand-checked values', () => {
  assert.equal(euclidean({ x: 0, y: 0 }, { x: 3, y: 4 }), 5);
  assert.equal(manhattan({ x: 0, y: 0 }, { x: 3, y: 4 }), 7);
  assert.equal(euclidean({ x: 1, y: 1 }, { x: 1, y: 1 }), 0);
});

// Two well-separated clusters: class 0 near (0,0), class 1 near (10,10).
const train = [
  { x: 0, y: 0, c: 0 }, { x: 1, y: 0, c: 0 }, { x: 0, y: 1, c: 0 }, { x: 1, y: 1, c: 0 },
  { x: 10, y: 10, c: 1 }, { x: 9, y: 10, c: 1 }, { x: 10, y: 9, c: 1 }, { x: 9, y: 9, c: 1 },
];

test('classifies clearly-separable points correctly', () => {
  assert.equal(knnPredict(train, { x: 0.5, y: 0.5 }, { k: 3 }), 0);
  assert.equal(knnPredict(train, { x: 9.5, y: 9.5 }, { k: 3 }), 1);
});

test('k=1 memorises training points (each maps to its own label)', () => {
  for (const p of train) {
    assert.equal(knnPredict(train, { x: p.x, y: p.y }, { k: 1 }), p.c);
  }
});

test('larger k smooths: a class-1 point surrounded by class 0 flips with big k', () => {
  // One stray class-1 point sitting inside the class-0 cluster.
  const data = [
    { x: 0, y: 0, c: 0 }, { x: 1, y: 0, c: 0 }, { x: 0, y: 1, c: 0 },
    { x: 2, y: 0, c: 0 }, { x: 0, y: 2, c: 0 },
    { x: 0.5, y: 0.5, c: 1 }, // the stray
  ];
  const q = { x: 0.5, y: 0.5 };
  // k=1 memorises the stray's own label.
  assert.equal(knnPredict(data, q, { k: 1 }), 1);
  // k=5 lets the surrounding class-0 majority win → smoothing.
  assert.equal(knnPredict(data, q, { k: 5 }), 0);
});

test('manhattan metric is selectable and still separates the clusters', () => {
  assert.equal(knnPredict(train, { x: 0.5, y: 0.5 }, { k: 3, metric: 'l1' }), 0);
  assert.equal(knnPredict(train, { x: 9.5, y: 9.5 }, { k: 3, metric: 'l1' }), 1);
});

test('empty training set returns -1', () => {
  assert.equal(knnPredict([], { x: 0, y: 0 }, { k: 1 }), -1);
});
