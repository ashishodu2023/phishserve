# phishserve/tune.py
import argparse
import random
import train

def tune(args):
    best_loss = float('inf')
    best_params = None

    for i in range(args.num_trials):
        print(f"Trial {i+1}/{args.num_trials}")

        # Sample hyperparameters
        params = {
            'csv': 'data/malicious_phish.csv',
            'epochs': random.randint(5, 20),
            'batch_size': random.choice([32, 64, 128, 256]),
            'max_len': random.randint(64, 128),
            'emb_dim': random.choice([64, 128, 256]),
            'hid': random.choice([64, 128, 256]),
            'lr': random.uniform(1e-4, 1e-2),
            'wd': random.uniform(1e-5, 1e-2),
            'out_dir': 'artifacts'
        }

        print("Training with params:", params)

        # Create a namespace object from the dictionary
        train_args = argparse.Namespace(**params)

        # Train the model
        val_loss = train.train(train_args)

        print(f"Validation loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            print("New best validation loss:", best_loss)
            print("Best params:", best_params)

    print("\n--- Tuning Complete ---")
    print("Best validation loss:", best_loss)
    print("Best hyperparameters:", best_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trials", type=int, default=10, help="number of trials to run")
    args = parser.parse_args()
    tune(args)
