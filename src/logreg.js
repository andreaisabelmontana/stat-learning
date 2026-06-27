// Logistic regression on 2-D points via full-batch gradient descent.
// Extracted from the interactive logistic-regression demo: sigmoid, one
// gradient step (with optional L2), prediction and accuracy.

/**
 * Logistic sigmoid.
 */
export function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

/**
 * Linear score w0*x + w1*y + w2 (bias) for a point.
 */
export function score(weights, point) {
  return weights[0] * point.x + weights[1] * point.y + weights[2];
}

/**
 * One full-batch gradient-descent step on the binary cross-entropy loss.
 * Returns new weights plus the loss and accuracy before the update.
 *
 * @param {number[]} weights [w_x, w_y, bias]
 * @param {{x:number,y:number,c:number}[]} points labels c in {0,1}
 * @param {object} [opts]
 * @param {number} [opts.lr=0.5] learning rate
 * @param {number} [opts.l2=0] L2 penalty on the two slope weights
 * @returns {{weights:number[], loss:number, accuracy:number}}
 */
export function trainStep(weights, points, opts = {}) {
  const { lr = 0.5, l2 = 0 } = opts;
  const n = points.length;
  if (!n) return { weights: weights.slice(), loss: 0, accuracy: 0 };
  const g = [0, 0, 0];
  let loss = 0, correct = 0;
  for (const p of points) {
    const yhat = sigmoid(score(weights, p));
    const e = yhat - p.c;
    g[0] += e * p.x;
    g[1] += e * p.y;
    g[2] += e;
    loss -= p.c * Math.log(yhat + 1e-9) + (1 - p.c) * Math.log(1 - yhat + 1e-9);
    if ((yhat > 0.5 ? 1 : 0) === p.c) correct++;
  }
  const next = weights.slice();
  for (let i = 0; i < 3; i++) {
    // Bias (index 2) is not regularised.
    next[i] -= lr * (g[i] / n + (i < 2 ? l2 * weights[i] : 0));
  }
  return { weights: next, loss: loss / n, accuracy: correct / n };
}

/**
 * Train logistic regression for a fixed number of epochs.
 * @returns {{weights:number[], loss:number, accuracy:number}}
 */
export function train(points, opts = {}) {
  const { epochs = 500, lr = 0.5, l2 = 0, init = [0, 0, 0] } = opts;
  let weights = init.slice();
  let last = { weights, loss: 0, accuracy: 0 };
  for (let e = 0; e < epochs; e++) {
    last = trainStep(weights, points, { lr, l2 });
    weights = last.weights;
  }
  return last;
}

/**
 * Predicted class label for a point given weights.
 */
export function predict(weights, point) {
  return sigmoid(score(weights, point)) > 0.5 ? 1 : 0;
}

/**
 * Fraction of points correctly classified by the given weights.
 */
export function accuracy(weights, points) {
  if (!points.length) return 0;
  let correct = 0;
  for (const p of points) if (predict(weights, p) === p.c) correct++;
  return correct / points.length;
}
