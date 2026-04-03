import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from cnn_model import DiceClassifierCNN, CLASS_NAMES


MODEL_PATH = "dice_cnn.pth"
IMAGE_SIZE = 64


class DicePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DiceClassifierCNN(num_classes=len(CLASS_NAMES)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def predict_bgr(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

        score_map = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        return pred_label, confidence, score_map


if __name__ == "__main__":
    predictor = DicePredictor()

    img = cv2.imread("test_cell.png")
    label, confidence, scores = predictor.predict_bgr(img)

    print("label =", label)
    print("confidence =", confidence)
    print("scores =", scores)