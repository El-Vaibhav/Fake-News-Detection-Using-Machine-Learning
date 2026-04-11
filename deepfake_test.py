import torch
import timm
from PIL import Image
from torchvision import transforms

# ================= CONFIG =================
MODEL_PATH = r"C:\Users\HP\OneDrive\Desktop\Fake_News_Detection\Fake-News-Detection-Using-Machine-Learning\model.pth"   # your saved model
IMG_PATH = r"C:\Users\HP\OneDrive\Pictures\Screenshots\Screenshot 2026-04-09 220505.png"  # change this
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", DEVICE)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= LOAD MODEL =================
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("✅ Model Loaded Successfully")

# ================= LOAD IMAGE =================
img = Image.open(IMG_PATH).convert("RGB")
img = transform(img).unsqueeze(0).to(DEVICE)

# ================= PREDICTION =================
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

fake_score = probs[0][1].item()
real_score = probs[0][0].item()

# ================= RESULT =================
print("\nPrediction Score:", fake_score)

if fake_score > 0.5:
    print("❌ FAKE IMAGE")
else:
    print("✅ REAL IMAGE")

print(f"Fake Confidence: {fake_score*100:.2f}%")
print(f"Real Confidence: {real_score*100:.2f}%")