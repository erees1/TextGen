TextGen
==============================


## Dataset

No data is included in this repo, but I used my soical media message history from facebook messenger and iMessage to train the model. To preapre the data run the following commands.

```bash
make data
make short-vectors
```
## Training

I used the notebook [C1-train_model.ipynb](./notebooks/C1-train_model.ipynb) to train the model as I was using Google Colab. To train the model locally run

```bash
python src/models/train_model.py data/processed/train/strings_X.txt data/processed/train/strings_Y.txt --spec_path $spec_path --log_dir $log_dir --checkpoint_dir $checkpoint_dir --vocab_filepath $vocab_filepath
``` 

## Inference

Inference is carried out using the predict_model.py script, I also made command line tool to chat with the bot in [`src/cli`](./src/cli)

---
## Project Organization


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
    ├── [notebooks](./notebooks)          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── [src](./src)                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py
    
