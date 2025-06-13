"""Apply a trained DAPI model to a single brightfield image."""

import torch
from PIL import Image
import torchvision.transforms.functional as TF
from Models.unet import UNet

MODEL_PATH = "dapi_model.pth"
INPUT_IMAGE = "bf_example.tif"
OUTPUT_PATH = "predicted_dapi.tif"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    img = Image.open(INPUT_IMAGE).convert("L")
    tensor = TF.normalize(TF.to_tensor(img), [0.5], [0.5]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor).squeeze(0).cpu()

    out = pred * 0.5 + 0.5
    TF.to_pil_image(out).save(OUTPUT_PATH)


if __name__ == "__main__":
    main()
