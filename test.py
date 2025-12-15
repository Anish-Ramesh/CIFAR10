import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import os
import sys

# 1. DEFINE THE ARCHITECTURE (Must match training exactly)
# -------------------------------------------------------
hyp = {
    'net': {
        'widths': {'block1': 64, 'block2': 256, 'block3': 256},
        'batchnorm_momentum': 0.6,
        'scaling_factor': 1.0 / 9.0,
    }
}

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x): return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12):
        super().__init__(num_features, eps=eps, momentum=0.1) 

class Conv(nn.Conv2d):
    def __init__(self, c_in, c_out, k_size=3, padding='same', bias=False):
        super().__init__(c_in, c_out, kernel_size=k_size, padding=padding, bias=bias)

class ConvGroup(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = Conv(c_in, c_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(c_out)
        self.conv2 = Conv(c_out, c_out)
        self.norm2 = BatchNorm(c_out)
        self.activ = nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        return x

def make_net():
    widths = hyp['net']['widths']
    whiten_width = 2 * 3 * 2**2
    net = nn.Sequential(
        Conv(3, whiten_width, 2, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width, widths['block1']),
        ConvGroup(widths['block1'], widths['block2']),
        ConvGroup(widths['block2'], widths['block3']),
        nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(hyp['net']['scaling_factor']),
    )
    for m in net.modules():
        if isinstance(m, BatchNorm): m.float()
    return net

# 2. SETUP & LOAD MODEL
# ---------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CIFAR_CLASSES = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Define Normalization (Must match training!)
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

# Load Model Once
print(f"🚀 Loading model on {device}...")
model = make_net().to(device).half() # Use FP16
try:
    state_dict = torch.load("airbench94.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Weights loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'airbench94.pth' not found. Run training first!")
    sys.exit(1)

# 3. PREDICTION FUNCTION
# ----------------------
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        return

    # Load and Preprocess Image
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Transform: Resize -> Tensor -> Normalize
        transform = T.Compose([
            T.Resize((32, 32)),  # Force resize to CIFAR size
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        
        # Prepare for GPU (Batch size 1, FP16)
        img_tensor = transform(img).unsqueeze(0).to(device).half()
        
        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            class_idx = predicted.item()
            class_name = CIFAR_CLASSES[class_idx]
            conf_score = confidence.item() * 100
            
            print(f"\n🖼️  Image: {image_path}")
            print(f"🤖 Prediction: {class_name}")
            print(f"📊 Confidence: {conf_score:.2f}%")
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")

# 4. MAIN EXECUTION
# -----------------
if __name__ == "__main__":
    # Check if user provided an image path
    if len(sys.argv) > 1:
        # Mode 1: Predict Single Image
        # Usage: python test.py my_dog.jpg
        user_image = sys.argv[1]
        predict_image(user_image)
    else:
        # Mode 2: Test on Full Test Set (Default)
        print("\n(No image provided. Testing on full CIFAR-10 dataset...)")
        print("To test a specific image, run: python test.py path/to/image.jpg\n")
        
        test_set = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True)
        images = torch.tensor(test_set.data).permute(0, 3, 1, 2).float().div(255)
        normalize = T.Normalize(torch.tensor(CIFAR_MEAN), torch.tensor(CIFAR_STD))
        images = normalize(images).to(device).half()
        labels = torch.tensor(test_set.targets, device=device)
        
        batch_size = 1000
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_img = images[i:i+batch_size]
                batch_lbl = labels[i:i+batch_size]
                outputs = model(batch_img)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_lbl.size(0)
                correct += (predicted == batch_lbl).sum().item()

        acc = 100 * correct / total
        print(f"🏆 Final Test Accuracy: {acc:.2f}%")