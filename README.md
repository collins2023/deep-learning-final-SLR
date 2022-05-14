# Sign Language Recognition

## Dataset

The dataset we used is [Kaggle's ASL Alphabets](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). The dataset must be unzipped and placed in the repository root, and the directory must be named `data`. The structure should be something like `data/asl_alphabet_train/asl_alphabet_train/A/A1.jpg`

## How to run

To train the model, run `python3 src/slr.py` from the root of the repository.

If you have not previously trained the model, you will have to adjust line 69 in `slr.py`. You should change `epochs=0` to your desired number of training epochs, and `load_weights=True` to `load_weights=False`.

To run the live feed, you must have already trained the model. The model should be saved as `model.onnx` in the repository root. You can run `python3 src/feed.py` to run the live feed.