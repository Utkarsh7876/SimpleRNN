# SimpleRNN

A small example project demonstrating a Simple RNN for sentiment analysis on the IMDB dataset. The repository includes training scripts, notebooks for experimentation, and a pretrained model for quick predictions.

## Contents
- `embedding.ipynb` — Notebook exploring embedding layers.
- `main.py` — Script to build and train the SimpleRNN model.
- `prediction.ipynb` — Notebook to load the pretrained model and run predictions on custom text.
- `simplelern.ipynb` — Additional experiments and learning notes.
- `simplernn_imdb_model.h5` — Pretrained model weights (Keras .h5 file).

## Requirements
Install dependencies (you can use the repo `requirements.txt` at the project root):

```bash
pip install -r ../requirements.txt
```

If you prefer to install manually, the main dependency is TensorFlow (tested with TensorFlow 2.x) and common libraries such as NumPy.

## Setup
1. Create and activate a Python virtual environment (recommended):

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```
2. Install dependencies as shown above.
3. Ensure `simplernn_imdb_model.h5` is present in this folder if you want to skip training.

## Usage
- To train a model from scratch, run:

```bash
python main.py
```

- To run predictions interactively, open `prediction.ipynb` in Jupyter or VS Code and run the cells. The notebook demonstrates how to load `simplernn_imdb_model.h5`, preprocess text, and call the model to get a sentiment prediction.

- Example prediction workflow (from the notebook): the notebook defines `preprocess_text()` and `predict_sentiment()` which you can reuse in scripts to get a `Positive`/`Negative` label and a prediction score.

## Notes
- The IMDB dataset used by the examples limits vocabulary to the top 10,000 words; unknown words are mapped to an unknown token.
- Input sequences are padded/truncated to length 500 in the provided preprocessing function.

## Troubleshooting
- If you get errors loading the model, make sure your TensorFlow version is compatible with the model file.
- If training is slow or runs out of memory on CPU, consider reducing batch size or training on a GPU-enabled environment.

## License
This project is provided as-is for learning and experimentation.

--
Generated README for the `SimpleRNN` example project.
