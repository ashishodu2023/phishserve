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

## Model Architecture

The model is a simple Bi-directional GRU with an embedding layer and a linear layer. The architecture is defined in `model.py`.

## Dataset

The dataset used for training is `data/urls.csv`. It contains a list of URLs and their corresponding labels (0 for benign, 1 for phishing).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
