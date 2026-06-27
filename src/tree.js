// Binary classification decision tree (CART-style, axis-aligned splits).
// Extracted from the interactive decision-tree demo: impurity measures,
// best-threshold search, recursive build and prediction.

/**
 * Gini impurity of a node with n0 points of class 0 and n1 of class 1.
 */
export function gini(n0, n1) {
  const n = n0 + n1;
  if (!n) return 0;
  const p0 = n0 / n, p1 = n1 / n;
  return 1 - p0 * p0 - p1 * p1;
}

/**
 * Shannon entropy (base 2) of a binary node.
 */
export function entropy(n0, n1) {
  const n = n0 + n1;
  if (!n) return 0;
  const p0 = n0 / n, p1 = n1 / n;
  let h = 0;
  if (p0 > 0) h -= p0 * Math.log2(p0);
  if (p1 > 0) h -= p1 * Math.log2(p1);
  return h;
}

function impurityFn(crit) {
  return crit === 'entropy' ? entropy : gini;
}

/**
 * Count class-0 and class-1 points.
 */
function counts(points) {
  let c0 = 0, c1 = 0;
  for (const p of points) (p.c === 0 ? c0++ : c1++);
  return [c0, c1];
}

/**
 * Find the axis-aligned split that minimises the size-weighted child
 * impurity over features 'x' and 'y'.
 *
 * @param {{x:number,y:number,c:number}[]} points
 * @param {object} [opts]
 * @param {'gini'|'entropy'} [opts.criterion='gini']
 * @param {number} [opts.minLeaf=1] minimum points required in each child
 * @returns {null | {feature:'x'|'y', threshold:number, impurity:number}}
 */
export function bestSplit(points, opts = {}) {
  const { criterion = 'gini', minLeaf = 1 } = opts;
  const imp = impurityFn(criterion);
  const n = points.length;
  let best = null;
  for (const feat of ['x', 'y']) {
    const sorted = points.slice().sort((a, b) => a[feat] - b[feat]);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i][feat] === sorted[i - 1][feat]) continue;
      const thr = (sorted[i][feat] + sorted[i - 1][feat]) / 2;
      let l0 = 0, l1 = 0, r0 = 0, r1 = 0;
      for (const p of sorted) {
        if (p[feat] <= thr) (p.c === 0 ? l0++ : l1++);
        else (p.c === 0 ? r0++ : r1++);
      }
      const nl = l0 + l1, nr = r0 + r1;
      if (nl < minLeaf || nr < minLeaf) continue;
      const weighted = (nl * imp(l0, l1) + nr * imp(r0, r1)) / n;
      if (!best || weighted < best.impurity) {
        best = { feature: feat, threshold: thr, impurity: weighted };
      }
    }
  }
  return best;
}

/**
 * Recursively grow a classification tree.
 *
 * @param {{x:number,y:number,c:number}[]} points
 * @param {object} [opts]
 * @param {number} [opts.maxDepth=6]
 * @param {number} [opts.minLeaf=1]
 * @param {'gini'|'entropy'} [opts.criterion='gini']
 * @returns tree node
 */
export function buildTree(points, opts = {}) {
  const { maxDepth = 6, minLeaf = 1, criterion = 'gini' } = opts;
  return grow(points, 0, maxDepth, minLeaf, criterion);
}

function grow(points, depth, maxDepth, minLeaf, criterion) {
  if (points.length === 0) return { leaf: true, prediction: 0, n0: 0, n1: 0 };
  const [c0, c1] = counts(points);
  const prediction = c1 >= c0 ? 1 : 0;
  // Stop on a pure node, depth limit, or too few points to split.
  if (depth >= maxDepth || points.length <= minLeaf || c0 === 0 || c1 === 0) {
    return { leaf: true, prediction, n0: c0, n1: c1 };
  }
  const split = bestSplit(points, { criterion, minLeaf });
  if (!split) return { leaf: true, prediction, n0: c0, n1: c1 };
  const left = points.filter((p) => p[split.feature] <= split.threshold);
  const right = points.filter((p) => p[split.feature] > split.threshold);
  return {
    leaf: false,
    feature: split.feature,
    threshold: split.threshold,
    n0: c0,
    n1: c1,
    left: grow(left, depth + 1, maxDepth, minLeaf, criterion),
    right: grow(right, depth + 1, maxDepth, minLeaf, criterion),
  };
}

/**
 * Predict the class of a point by walking the tree.
 */
export function predictTree(tree, point) {
  let node = tree;
  while (!node.leaf) {
    node = point[node.feature] <= node.threshold ? node.left : node.right;
  }
  return node.prediction;
}
