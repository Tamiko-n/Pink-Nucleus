# Pink Nucleus

A four-page site for the breast cancer classification project.

## Pages
- index.html      Title page (Pink Nucleus)
- learn.html      Early detection / awareness
- diagnose.html   Diagnosis tool (enter a report, get benign/malignant)
- results.html    The project: pipeline, model, results

## Where the model comes from (not hardcoded)
- train.py reads breast_cancer.csv, cleans it, and trains the from-scratch
  NumPy neural network with manual backpropagation, then exports the LEARNED
  weights to model.js.
- The website loads those learned weights to make predictions in the browser.
- To retrain:  python train.py   (regenerates model.js + model_bundle.json)

Latest training result: 99.12% test accuracy, 97.6% recall, ROC-AUC 0.996.

## How to run
1. Open the folder in VS Code, install the "Live Server" extension.
2. Right-click index.html -> "Open with Live Server".
   (Or just double-click index.html to open it in a browser.)

## How to deploy
- Drag this folder onto https://app.netlify.com/drop  for a public link, or
- Push to a GitHub repo and enable GitHub Pages (entry file is index.html).

Keep all files together; the pages link to each other by relative path.

Educational project only. Not a medical device.
