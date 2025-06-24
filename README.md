# FIFA_trainAImodle
Installation
After cloning and navigate to the project folder

Extract data.zip and put it into a data folder under the root directory.

Create a virtual environment and activate it
python -m venv env
source env/bin/activate
Install dependencies
pip install -r requirements.txt
Usage
Data preprocessing
python src/data_preprocess.py
Train the model
python src/train_model.py
Evaluating
python src/evaluate_model.py
Make predictions
python src/infer.py
Notes
The _nf files are an attempt to train with k-fold, which lead to better results.
