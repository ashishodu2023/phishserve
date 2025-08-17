# phishserve/export.py
import os, json, torch, shutil, argparse
from utils import device

def export(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    # create a temporary directory
    tmp_dir = os.path.join(args.out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu")
    # Save state_dict, vocab and config for TorchServe
    torch.save(ck["model"], os.path.join(tmp_dir, "model.pt") )
    
    itos = ck["itos"]
    with open(os.path.join(tmp_dir, "itos.txt"), "w") as f:
        f.write("\n".join(itos))

    # copy handler and checkpoint
    shutil.copy(os.path.abspath("handler.py"), tmp_dir)
    print(os.path.abspath(args.ckpt))
    shutil.copy(os.path.abspath(args.ckpt), os.path.join(tmp_dir, "best.pt"))

    # create mar file
    cmd = f"""
    torch-model-archiver --model-name phishserve \
    --version 1.0 \
    --model-file {os.path.join(os.getcwd(), 'model.py')}\
    --serialized-file {os.path.join(tmp_dir, "model.pt")}\
    --handler {os.path.join(tmp_dir, "handler.py")}\
    --extra-files {os.path.join(tmp_dir, "itos.txt")},{os.path.join(tmp_dir, "best.pt")}\
    --export-path {args.out_dir} -f
    """
    print(cmd)
    os.system(f"ls -l {tmp_dir}")
    os.system(cmd)

    # remove temporary directory
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="artifacts/best.pt", help="path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default="artifacts", help="output directory")
    args = parser.parse_args()
    export(args)
