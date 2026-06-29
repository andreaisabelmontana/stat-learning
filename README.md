# Statistical Learning & Prediction — Interactive

Single-page, zero-dependency interactive companion to the SLP course. Every
canvas is live: click to add points, drag, slide hyper-parameters, and watch the
model react in real time.

The visualisations cover k-NN, linear & logistic regression, regularisation,
bias–variance, gradient descent, softmax/SGD, activation functions, MLP +
backprop, decision trees, bagging, random forests, AdaBoost, the kernel trick,
soft-margin SVM (SMO), PCA, MDS, k-means / k-means++, cross-validation,
ROC/PR/confusion metrics, hierarchical clustering, and anomaly detection.

## Tested algorithm cores

The numerical heart of the supervised/unsupervised demos is factored out of the
canvas code into framework-free ES modules under [`src/`](src/). They take plain
`{x, y, c}` data — no canvas, no DOM — so they can be unit-tested directly, and
the page imports them at its real call sites.

| Module | Algorithm | Public API |
| --- | --- | --- |
| [`src/knn.js`](src/knn.js) | k-NN classification | `euclidean`, `manhattan`, `knnPredict` |
| [`src/tree.js`](src/tree.js) | CART classification tree | `gini`, `entropy`, `bestSplit`, `buildTree`, `predictTree` |
| [`src/kmeans.js`](src/kmeans.js) | k-means (Lloyd) + k-means++ | `assign`, `updateCentroids`, `wss`, `kmeansPlusPlusInit`, `kmeans` |
| [`src/logreg.js`](src/logreg.js) | logistic regression (GD) | `sigmoid`, `trainStep`, `train`, `predict`, `accuracy` |
| [`src/ensemble.js`](src/ensemble.js) | bagging + AdaBoost (stumps) | `bootstrap`, `majorityVote`, `trainBagging`, `baggingPredict`, `bestStump`, `adaboostTrain`, `adaboostPredict` |
| [`src/rng.js`](src/rng.js) | seedable LCG (the page's generator) | `makeRng` |

`index.html` loads `app.js` as a module, and `app.js` imports these cores — the
k-NN vote, the k-means assign/update/WSS and k-means++ seeding, the logistic
regression gradient step, and the bagging bootstrap all run through the tested
code, so what the tests prove is what the page runs.

## Properties proven by the tests

- **k-NN** classifies clearly-separable points correctly; **k = 1 memorises**
  the training set (each point maps to its own label); larger **k smooths** —
  a stray point inside the opposite cluster flips once enough neighbours vote.
  Both L2 and L1 metrics separate the clusters.
- **Decision tree**: `bestSplit` picks the split that **most reduces weighted
  impurity** (Gini/entropy) on a constructed dataset; the chosen split strictly
  beats the parent node's impurity; a **pure node is not split**; the grown tree
  perfectly classifies an axis-separable dataset under both criteria.
- **k-means** **converges to the known clusters** on separable blobs (recovered
  centroids land near the true centres, every true cluster keeps a single
  label) and the **WSS objective never increases** across iterations.
- **Logistic regression** **learns a linearly separable problem** (training
  accuracy → 100 %), a single gradient step reduces the loss, and the trained
  model predicts the correct side of the boundary.
- **Ensembles**: `bestStump` finds a zero-error separating stump; **AdaBoost
  reaches zero training error** on a separable set; the **bagged tree ensemble**
  classifies both clusters correctly by majority vote.
- **Helpers** — distances, `gini`/`entropy`, `sigmoid`, `wss`, `majorityVote` —
  return hand-checked values.

Randomised tests seed the RNG (`makeRng`) for reproducibility.

## Run the tests

Node 24+ (uses the built-in test runner — no npm install, no dependencies):

```bash
node --test   # 27 tests, all pass
```

## Run the site

No build step. Open `index.html` in any modern browser, or serve the folder so
the ES module imports resolve over HTTP:

```bash
python -m http.server 8000
# then open http://localhost:8000/
```

## Coursework

Hands-on projects from the course, each its own repo + live demo:

- [Fraud Detector](https://andreaisabelmontana.github.io/fraud-detector/) — live transaction stream with a threshold-driven fraud classifier
- [Mortgage Predict](https://andreaisabelmontana.github.io/mortgage-predict/) — ML pipeline predicting mortgage outcomes, with fairness/explainability
- [Sound Classifier](https://andreaisabelmontana.github.io/sound-classifier/) — ESC-50 animal sounds, Random Forest vs CNN on log-mel spectrograms
- [Statistical Learning](https://andreaisabelmontana.github.io/statistical-learning/) — core methods implemented from first principles
- [BeyondStats](https://andreaisabelmontana.github.io/beyondstats/) — gender inequality via one ML-powered composite score
