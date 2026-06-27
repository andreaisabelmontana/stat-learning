// Ensemble methods: bagging (bootstrap + majority vote over trees) and
// AdaBoost with decision stumps. Extracted from the interactive bagging and
// AdaBoost demos.

import { makeRng } from './rng.js';
import { buildTree, predictTree } from './tree.js';

/**
 * Sample a dataset of the same size with replacement (a bootstrap replicate).
 */
export function bootstrap(points, rng = makeRng()) {
  const out = [];
  for (let i = 0; i < points.length; i++) {
    out.push(points[Math.floor(rng() * points.length)]);
  }
  return out;
}

/**
 * Majority vote over an array of {0,1} predictions. Ties resolve to 1.
 */
export function majorityVote(votes) {
  let ones = 0;
  for (const v of votes) if (v === 1) ones++;
  return ones * 2 >= votes.length ? 1 : 0;
}

/**
 * Train a bagged ensemble of decision trees on bootstrap replicates.
 * @returns {{trees:Array}} the ensemble
 */
export function trainBagging(points, opts = {}) {
  const { nTrees = 25, maxDepth = 6, rng = makeRng(), criterion = 'gini' } = opts;
  const trees = [];
  for (let i = 0; i < nTrees; i++) {
    trees.push(buildTree(bootstrap(points, rng), { maxDepth, criterion }));
  }
  return { trees };
}

/**
 * Predict a point from a bagged ensemble by majority vote of its trees.
 */
export function baggingPredict(model, point) {
  return majorityVote(model.trees.map((t) => predictTree(t, point)));
}

// ---- AdaBoost (SAMME / discrete AdaBoost with decision stumps) ----
// Labels for AdaBoost are in {-1, +1}.

/**
 * Find the weighted-error-minimising decision stump over features x and y.
 *
 * @param {{x:number,y:number,c:number}[]} points labels c in {-1,+1}
 * @param {number[]} weights sample weights (sum to 1)
 * @returns {{feature, threshold, sign, error}}
 */
export function bestStump(points, weights) {
  let best = { error: Infinity, feature: 'x', threshold: 0, sign: 1 };
  for (const feat of ['x', 'y']) {
    const vals = points.map((p) => p[feat]).sort((a, b) => a - b);
    for (let i = 0; i < vals.length - 1; i++) {
      const thr = (vals[i] + vals[i + 1]) / 2;
      for (const sign of [1, -1]) {
        let err = 0;
        points.forEach((p, j) => {
          const pred = (p[feat] <= thr ? 1 : -1) * sign;
          if (pred !== p.c) err += weights[j];
        });
        if (err < best.error) best = { error: err, feature: feat, threshold: thr, sign };
      }
    }
  }
  return best;
}

/**
 * Predicted {-1,+1} label of a single stump.
 */
export function stumpPredict(stump, point) {
  return (point[stump.feature] <= stump.threshold ? 1 : -1) * stump.sign;
}

/**
 * Train AdaBoost. Adds rounds until `rounds` stumps accumulate or a stump's
 * weighted error reaches 0.5 (no longer better than chance).
 *
 * @returns {{stumps: Array<{feature,threshold,sign,alpha}>}}
 */
export function adaboostTrain(points, opts = {}) {
  const { rounds = 20 } = opts;
  const n = points.length;
  const w = new Array(n).fill(1 / n);
  const stumps = [];
  for (let r = 0; r < rounds; r++) {
    const s = bestStump(points, w);
    if (s.error >= 0.5) break;
    const alpha = 0.5 * Math.log((1 - s.error) / Math.max(s.error, 1e-9));
    let Z = 0;
    points.forEach((p, j) => {
      const pred = stumpPredict(s, p);
      w[j] *= Math.exp(-alpha * p.c * pred);
      Z += w[j];
    });
    for (let j = 0; j < n; j++) w[j] /= Z;
    stumps.push({ ...s, alpha });
  }
  return { stumps };
}

/**
 * Raw weighted-vote margin of an AdaBoost model at a point.
 */
export function adaboostScore(model, point) {
  let s = 0;
  for (const st of model.stumps) s += st.alpha * stumpPredict(st, point);
  return s;
}

/**
 * AdaBoost class prediction in {-1,+1} (sign of the margin).
 */
export function adaboostPredict(model, point) {
  return adaboostScore(model, point) >= 0 ? 1 : -1;
}
