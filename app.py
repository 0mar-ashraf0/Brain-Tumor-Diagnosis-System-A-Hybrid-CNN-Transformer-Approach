import gradio as gr
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from model import SOTA_HybridModel
from utils import GradCAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABELS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Load Model
model = SOTA_HybridModel(num_classes=4)
model.load_state_dict(torch.load('best_brain_tumor_model_patched.pth', map_location=device))
model.to(device)
model.eval()

# Grad-CAM Setup
grad_cam = GradCAM(model, model.effnet.features[-1])

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    processed_img = transform(image).unsqueeze(0).to(device)
    heatmap, pred_idx = grad_cam(processed_img)
    
    with torch.no_grad():
        output = model(processed_img)
        probs = F.softmax(output, dim=1)[0]
        
    confidences = {LABELS[i]: float(probs[i]) for i in range(4)}
    
    # Overlay
    img_cv = np.array(image.resize((224, 224)))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)
    
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB), confidences

if __name__ == "__main__":
    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Image(label="Grad-CAM"), gr.Label(num_top_classes=4)],
        title="Brain Tumor Diagnosis AI"
    )
    interface.launch()
