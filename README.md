# SLP Interactive — Statistical Learning & Prediction visualisations

Single-page, zero-dependency interactive companion to the SLP course. Every canvas is live: click, drag, slide hyper-parameters, and watch the model react.

## Topics

Foundations · k-NN · Linear regression · Loss functions · Ridge/Lasso · Bias–Variance tradeoff
Optimisation · Gradient descent · SGD with softmax
Neural nets · Activation functions · MLP + backprop (from scratch)
Trees & ensembles · Decision trees · Bagging · Random forest · AdaBoost
Kernels & margins · The kernel trick · Soft-margin SVM (SMO)
Unsupervised · PCA · MDS (SMACOF) · k-means / k-means++

## Local preview

Just open `index.html` in any modern browser — no build step.

## Deploying to GitHub Pages

```bash
cd slp-interactive
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

Then in the repo settings → **Pages** → source = `main` / root. Site appears at
`https://<you>.github.io/<repo>/`.

A `.nojekyll` file is included so GitHub Pages serves the files unchanged.
