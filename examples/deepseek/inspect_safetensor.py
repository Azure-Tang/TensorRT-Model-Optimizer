import argparse
from safetensors import safe_open

def main(file_path: str):
    with safe_open(file_path, framework="pt") as f:
        print(f"\nReading: {file_path}\n")
        for key in f.keys():
            if "layers.1." not in key:
                continue
            tensor = f.get_tensor(key)
            print(f"Key: {key}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Shape: {tensor.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect safetensors file")
    parser.add_argument("file_path", type=str, help="Path to the .safetensors file")
    args = parser.parse_args()

    main(args.file_path)
