# TextGen

## Dataset

No data is included in this repo, but I used my soical media message history from facebook messenger and iMessage to train the model. To preapre the data run the following commands.

```bash
make data
make short-vectors
```

## Training

I used the notebook [`C1-train_model.ipynb`](./notebooks/C1-train_model.ipynb) to train the model as I was using Google Colab. To train the model locally run the [`src/models/train_model.py`](`./src/models/train_model.py`) script.

## Inference

Inference is carried out using the [`src/model/predict_model.py`](./src/models/predict.py) script, I also made command line tool to chat with the bot in [`src/cli`](./src/cli)

---

## Project Organization

```
  ├── LICENSE
  ├── Makefile           <- Makefile with commands like `make data` or `make train`
  ├── README.md          <- The top-level README for developers using this project.
  ├── data
  │   ├── external       <- Data from third party sources.
  │   ├── interim        <- Intermediate data that has been transformed.
  │   ├── processed      <- The final, canonical data sets for modeling.
  │   └── raw            <- The original, immutable data dump.
  │
  │
  ├── models             <- Trained and serialized models, model predictions, or model summaries
  │
  ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
  │                         the creator's initials, and a short `-` delimited description, e.g.
  │                         `1.0-jqp-initial-data-exploration`.
  │
  ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
  │                         generated with `pip freeze > requirements.txt`
  │
  ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
  ├── src                <- Source code for use in this project.
      ├── __init__.py    <- Makes src a Python module
      │
      ├── data           <- Scripts to download or generate data
      │   └── make_dataset.py
      │   └── msg_pipeline.py
      │
      ├── data_transform <- Scripts to turn raw data into features for modeling
      │   └── preprocessing.py
      │
      ├── models         <- Scripts to train models and then use trained models to make
          │                 predictions
          ├── predict_model.py
          └── train_model.py
```