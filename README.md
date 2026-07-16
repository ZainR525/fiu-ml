# fiu-ml

First code flow for the FIU federated learning project. A basic supervised
classification pipeline on a public dataset, built to get familiar with the
standard ML workflow before moving into the federated setup.

## What this does

`simpleML.py` trains a logistic regression classifier on sklearn's built-in
digits dataset and reports how well it does on data it has never seen.

## The dataset

1797 handwritten digits, each one an 8x8 grayscale image. Every image is
flattened into a list of 64 numbers, one per pixel, each value 0-16 for
darkness. So `X` is 1797 rows by 64 columns. Each row is paired with a
human-written label in `y`, a digit from 0 to 9.

That is what "1797 samples, 64 features, 10 classes" means in the output.
Samples are how many examples exist, features are how many numbers describe
each one, classes are the possible answers.

## The five steps

**1. Load.** `load_digits()` returns the bundle. `digits.data` is the pixel
values, `digits.target` is the labels.

**2. Split.** `train_test_split` cuts the data into 1437 training samples and
360 test samples. The model only learns from the training set. The test set
stays hidden until evaluation, which is the only way to tell whether the model
generalized or just memorized. `random_state=42` fixes the shuffle so the split
is reproducible across runs. `stratify=y` keeps the class balance the same in
both halves so no digit ends up underrepresented by chance.

**3. Scale.** `StandardScaler` rewrites each feature to mean 0 and standard
deviation 1. Logistic regression multiplies each pixel by a weight, so features
on wildly different scales distort the fit and slow convergence.

The important detail is `fit_transform` on train but only `transform` on test.
`fit` learns the mean and standard deviation. If you fit on the test set, its
statistics leak into preprocessing and the reported accuracy becomes inflated.
The test set gets scaled using the training set's statistics. The code runs
either way, which is what makes this easy to get wrong.

**4. Train.** `model.fit(X_train, y_train)` is the learning. Logistic regression
holds a weight per pixel per class, 640 weights total. To predict, it multiplies
the 64 pixel values by the 64 weights for each digit, sums them into a score,
and picks the highest-scoring class. Training starts those weights random and
repeatedly nudges them: guess, compare to the true label, adjust the weights
that pushed the wrong way. `max_iter=1000` caps the passes, since the default
100 is not enough to converge here.

**5. Evaluate.** `predict` runs the trained model on the 360 held-out samples,
then accuracy, a per-class report, and a confusion matrix compare predictions
to truth.

## Results

Test accuracy: **0.9722**, so 350 of 360 correct.

The per-class report shows 8 of the 10 digits at or near perfect. The two
exceptions are 1 and 8, both at 0.89 precision and recall.

The confusion matrix locates it precisely. Row 8, column 1 holds a 4, meaning
four true 8s were predicted as 1. Row 1 shows two 1s predicted as 8. Nearly all
remaining error is a single 1 vs 8 confusion, and it is asymmetric, with 8s
misread as 1s more often than the reverse.

At 8x8 resolution that tracks. The loops of an 8 collapse into a small number of
dark pixels and the shape flattens toward a vertical stroke. That reads as a
resolution limit rather than a model limit. Accuracy alone would not have
surfaced this. The confusion matrix did.

## Running it

```
pip3 install scikit-learn
python3 simpleML.py
```

No download needed, the dataset ships with sklearn.

## Summary of what I learned

AI is the broad field of getting machines to do things that normally need human intelligence. Machine learning sits inside it as the specific idea of learning patterns from data instead of following rules a programmer wrote by hand. Deep learning sits inside machine learning and uses neural networks with many hidden layers. The logistic regression model here is machine learning but not deep learning since it is really just one layer of weights mapping inputs to outputs.

Supervised learning means every input comes with the correct answer attached so the model can compare its guess against the truth and adjust. Unsupervised means there are no answers and the model has to find structure on its own. Classification vs regression comes down to what you are predicting. Power demand in megawatts is a continuous number so that is regression. Bucketing demand into High Medium or Low makes it classification.

Federated learning was the part most relevant to this project. You cannot just pool everything on one server because a lot of data legally cannot move. Patient records are protected and utility customer data is restricted. So instead of bringing the data to the model you bring the model to the data. Each client trains locally and only sends back the updated weights and the server averages those weights into a new global model which is FedAvg. The hard part is non-IID data. One substation serving a busy city and another serving a rural area with solar panels have very different usage patterns so each client learns something that fits its own data and averaging those models means they pull in different directions. That gives slower convergence and less stable training than centralized. It seems to be one of the biggest open problems in the field and it is the piece I want to understand better once I start running the federated code.
