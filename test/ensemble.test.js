import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bootstrap, majorityVote, trainBagging, baggingPredict,
  bestStump, stumpPredict, adaboostTrain, adaboostPredict,
} from '../src/ensemble.js';
import { makeRng } from '../src/rng.js';

test('majorityVote returns the modal class, ties → 1', () => {
  assert.equal(majorityVote([0, 0, 1]), 0);
  assert.equal(majorityVote([1, 1, 0]), 1);
  assert.equal(majorityVote([0, 1]), 1); // tie
});

test('bootstrap keeps the sample size and draws only from the input', () => {
  const pts = [{ x: 1 }, { x: 2 }, { x: 3 }];
  const b = bootstrap(pts, makeRng(3));
  assert.equal(b.length, pts.length);
  for (const p of b) assert.ok(pts.includes(p));
});

// Two well-separated clusters for the tree-based learners.
const cls01 = [
  { x: 0, y: 0, c: 0 }, { x: 1, y: 0, c: 0 }, { x: 0, y: 1, c: 0 }, { x: 1, y: 1, c: 0 },
  { x: 10, y: 10, c: 1 }, { x: 9, y: 10, c: 1 }, { x: 10, y: 9, c: 1 }, { x: 9, y: 9, c: 1 },
];

test('bagging ensemble classifies separable clusters correctly', () => {
  const model = trainBagging(cls01, { nTrees: 15, maxDepth: 4, rng: makeRng(8) });
  assert.equal(model.trees.length, 15);
  assert.equal(baggingPredict(model, { x: 0.5, y: 0.5 }), 0);
  assert.equal(baggingPredict(model, { x: 9.5, y: 9.5 }), 1);
});

// AdaBoost uses labels in {-1,+1}.
const pm = [
  { x: 0, y: 0, c: -1 }, { x: 1, y: 0, c: -1 }, { x: 0, y: 1, c: -1 }, { x: 1, y: 1, c: -1 },
  { x: 10, y: 10, c: 1 }, { x: 9, y: 10, c: 1 }, { x: 10, y: 9, c: 1 }, { x: 9, y: 9, c: 1 },
];

test('bestStump finds a separating threshold with low weighted error', () => {
  const w = new Array(pm.length).fill(1 / pm.length);
  const s = bestStump(pm, w);
  // The clusters are separable by a single axis-aligned cut → zero error.
  assert.ok(s.error < 1e-9);
  // And that stump labels both clusters correctly.
  assert.equal(stumpPredict(s, { x: 0.5, y: 0.5 }), stumpPredict(s, pm[0]));
});

test('AdaBoost reaches zero training error on a separable set', () => {
  const model = adaboostTrain(pm, { rounds: 20 });
  assert.ok(model.stumps.length >= 1);
  let wrong = 0;
  for (const p of pm) if (adaboostPredict(model, p) !== p.c) wrong++;
  assert.equal(wrong, 0);
});

test('AdaBoost classifies fresh points on the correct side', () => {
  const model = adaboostTrain(pm, { rounds: 20 });
  assert.equal(adaboostPredict(model, { x: 0.5, y: 0.5 }), -1);
  assert.equal(adaboostPredict(model, { x: 9.5, y: 9.5 }), 1);
});
