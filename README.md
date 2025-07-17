# Solar Flare Prediction App (CNN-TCN)

A simple, modular app to predict solar flare class (A, B, C, M, X) using Temporal Convolutional Networks (TCN), based on preprocessed CNN image feature vectors.

## Features

* Supports training with and without class weights (to handle imbalance)
* Uses TCN model architecture with dilations and dropout
* Outputs accuracy, classification report, confusion matrix, and saved predictions
* CLI and Streamlit modes available
* Accepts manual feature vector input for real-time prediction

## Project Structure

```
solarflare_app/
├── app.py                      # CLI-based app entrypoint
├── model/
│   ├── evaluate.py            # Model evaluation and output
│   └── tcn_model.py           # TCN model architecture
├── data/
│   ├── preprocessing.py       # Data preprocessing logic
│   └── datasets/
│       └── flare_data.csv     # Main dataset
├── utils/
│   └── helpers.py             # Label mapping and helpers
├── streamlit_app.py           # Interactive Streamlit interface
├── requirements.txt           # Required packages
└── README.md
```

## Run the App

### CLI Mode (for training & prediction)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with weighted training
python3 app.py --mode weighted

# Run without class weight
python3 app.py --mode plain
```

### Streamlit Mode (Interactive)

```bash
streamlit run streamlit_app.py
```

## Manual Prediction

You can input a CNN-based feature vector manually in Streamlit interface to get real-time prediction.

## Output

* Confusion matrix plot
* Classification report (precision, recall, F1)
* Accuracy
* CSV file with predictions per sample

## Note

Ensure your feature file is in `flare_data.csv` format with `feat_1`, `feat_2`, ..., and `flare_label` columns. CNN features should be preprocessed beforehand.

---

Created as part of an undergraduate thesis project.
