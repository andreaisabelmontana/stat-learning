import test from 'node:test';
import assert from 'node:assert/strict';
import { gini, entropy, bestSplit, buildTree, predictTree } from '../src/tree.js';

test('impurity helpers return hand-checked values', () => {
  // Pure node → zero impurity.
  assert.equal(gini(4, 0), 0);
  assert.equal(entropy(4, 0), 0);
  // 50/50 split → max impurity (gini 0.5, entropy 1 bit).
  assert.equal(gini(5, 5), 0.5);
  assert.equal(entropy(5, 5), 1);
  // 3:1 gini = 1 - 0.75^2 - 0.25^2 = 0.375.
  assert.ok(Math.abs(gini(3, 1) - 0.375) < 1e-12);
});

test('bestSplit picks the split that best reduces impurity', () => {
  // Class 0 on the left (x<2), class 1 on the right (x>2). The clean split is
  // a vertical cut at x≈2.
  const points = [
    { x: 0, y: 0, c: 0 }, { x: 1, y: 1, c: 0 }, { x: 1, y: 5, c: 0 },
    { x: 3, y: 0, c: 1 }, { x: 4, y: 1, c: 1 }, { x: 4, y: 5, c: 1 },
  ];
  const split = bestSplit(points);
  assert.equal(split.feature, 'x');
  assert.ok(split.threshold > 1 && split.threshold < 3);
  // This split is pure → weighted child impurity is 0, below the parent's 0.5.
  assert.ok(split.impurity < gini(3, 3));
  assert.ok(Math.abs(split.impurity) < 1e-12);
});

test('a pure node is not split (bestSplit still finds something, build leafs it)', () => {
  const pure = [
    { x: 0, y: 0, c: 1 }, { x: 1, y: 1, c: 1 }, { x: 2, y: 2, c: 1 },
  ];
  const tree = buildTree(pure);
  assert.equal(tree.leaf, true);
  assert.equal(tree.prediction, 1);
});

test('chosen split strictly decreases impurity vs the parent node', () => {
  const points = [
    { x: 0, y: 0, c: 0 }, { x: 0.2, y: 1, c: 0 }, { x: 0.1, y: 2, c: 0 }, { x: 0, y: 3, c: 0 },
    { x: 5, y: 0, c: 1 }, { x: 5.2, y: 1, c: 1 }, { x: 5.1, y: 2, c: 1 }, { x: 5, y: 3, c: 1 },
  ];
  const parentImpurity = gini(4, 4); // 0.5
  const split = bestSplit(points);
  assert.ok(split.impurity < parentImpurity);
});

test('built tree perfectly classifies a separable XOR-free dataset', () => {
  // Axis-aligned separable: class depends on whether x<2.5.
  const points = [];
  for (let y = 0; y < 5; y++) {
    points.push({ x: 0, y, c: 0 }, { x: 1, y, c: 0 });
    points.push({ x: 4, y, c: 1 }, { x: 5, y, c: 1 });
  }
  const tree = buildTree(points, { maxDepth: 4 });
  for (const p of points) {
    assert.equal(predictTree(tree, p), p.c);
  }
});

test('entropy criterion also builds a correct tree', () => {
  const points = [
    { x: 0, y: 0, c: 0 }, { x: 1, y: 0, c: 0 },
    { x: 9, y: 0, c: 1 }, { x: 10, y: 0, c: 1 },
  ];
  const tree = buildTree(points, { criterion: 'entropy' });
  assert.equal(predictTree(tree, { x: 0.5, y: 0 }), 0);
  assert.equal(predictTree(tree, { x: 9.5, y: 0 }), 1);
});
