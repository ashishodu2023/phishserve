# PhishServe: Phishing URL Detection

PhishServe is a machine learning project for detecting phishing URLs. It uses a simple Bi-directional GRU model trained on a dataset of benign and phishing URLs. The trained model is served using TorchServe, allowing for easy integration into other applications.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/phishserve.git
    cd phishserve
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train the model, run the `train.py` script. You can customize the training parameters using command-line arguments.

```bash
python train.py --epochs 10 --batch_size 64 --lr 0.001
```

The trained model and vocabulary will be saved in the `artifacts` directory.

### Hyperparameter Tuning

To find the best hyperparameters for your model, you can use the `tune.py` script. This script will automatically search for the best combination of hyperparameters.

```bash
python tune.py --num_trials 20
```

After the tuning process completes, the best hyperparameters will be printed to the console. You can then use these parameters to train your final model.

### Evaluation

To evaluate the trained model on the test set, run the `eval.py` script.

```bash
python eval.py --ckpt artifacts/best.pt
```

This will print a classification report, confusion matrix, and other metrics.

### Exporting the Model

After training, you can export the model to a `.mar` file for serving with TorchServe.

```bash
python export.py --ckpt artifacts/best.pt
```

This will create a `phishserve.mar` file in the `artifacts` directory.

### Running the Server

To start the TorchServe server, run the following command:

```bash
torchserve --start --ncs --model-store artifacts --models phishserve.mar
```

### Making Predictions

You can send prediction requests to the running server using `curl`.

First, get the inference token from the `key_file.json` file:
```bash
INFERENCE_TOKEN=$(jq -r '.inference.key' key_file.json)
```

Then, send the prediction request:
```bash
curl http://127.0.0.1:8080/predictions/phishserve -H "Authorization: Bearer $INFERENCE_TOKEN" -T data/urls.csv
```

## Project Flowchart

```
+-------------------+
|      Start        |
+-------------------+
         |
         v
+-------------------+
|  Data Preparation |
| (dataset.py,      |
|  utils.py)        |
| - Load CSV        |
| - Clean URLs      |
| - Build Vocabulary|
| - Split Data      |
+-------------------+
         |
         v
+-------------------+
|   Model Training  |
|   (train.py)      |
| - Initialize Model|
| - Train on        |
|   Training Data   |
| - Validate on     |
|   Validation Data |
| - Save Best Model |
+-------------------+
         |
         v
+-------------------+
| Hyperparameter    |
|   Tuning          |
|   (tune.py)       |
| - Search for      |
|   Optimal Params  |
+-------------------+
         |
         v
+-------------------+
|  Model Evaluation |
|   (eval.py)       |
| - Load Best Model |
| - Evaluate on     |
|   Test Data       |
| - Generate Metrics|
+-------------------+
         |
         v
+-------------------+
|    Model Export   |
|   (export.py)     |
| - Export Model    |
|   to .mar file    |
+-------------------+
         |
         v
+-------------------+
|   Model Serving   |
|   (TorchServe)    |
| - Start TorchServe|
| - Load .mar Model |
+-------------------+
         |
         v
+-------------------+
|    Prediction     |
|  (curl request)   |
| - Send URL        |
| - Get Prediction  |
+-------------------+
         |
         v
+-------------------+
|       End         |
+-------------------+
```

## Model Architecture

The model is a simple Bi-directional GRU with an embedding layer and a linear layer. The architecture is defined in `model.py`.

## Dataset

The dataset used for training is `data/urls.csv`. It contains a list of URLs and their corresponding labels (0 for benign, 1 for phishing).

## Results

After hyperparameter tuning, the best validation loss achieved was `0.693115234375` with the following hyperparameters:

```json
{
  "csv": "data/urls.csv",
  "epochs": 13,
  "batch_size": 128,
  "max_len": 116,
  "emb_dim": 128,
  "hid": 128,
  "lr": 0.00969559369403676,
  "wd": 0.001139501545145627,
  "out_dir": "artifacts"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
