# FIFA_trainAImodle
## Installation

After cloning and navigate to the project folder

Extract data.zip and put it into a data folder under the root directory.

1. Create a virtual environment and activate it
```
python -m venv env
source env/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```

## Usage
1. Data preprocessing
```
python src/data_preprocess.py
```
2. Train the model
```
python src/train_model.py
```
3. Evaluating
```
python src/evaluate_model.py
```
4. Make predictions
```
python src/infer.py
```

## Notes
The _nf files are an attempt to train with k-fold, which lead to better results.
