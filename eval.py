"""Apply a trained DAPI model to a single brightfield image."""

import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from Models.unet import UNet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict DAPI staining for a brightfield image")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--input", required=True, help="Brightfield image file")
    parser.add_argument("--output", required=True, help="Where to save the DAPI prediction")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    img = Image.open(args.input).convert("L")
    tensor = TF.normalize(TF.to_tensor(img), [0.5], [0.5]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).squeeze(0).cpu()

    out = pred * 0.5 + 0.5
    TF.to_pil_image(out).save(args.output)


if __name__ == "__main__":
    main()
