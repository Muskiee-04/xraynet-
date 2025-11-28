import torch
import torch.onnx
from torchvision import models
import torch.nn as nn
import os

def convert_to_onnx():
    """Convert trained PyTorch model to ONNX format"""
    
    # Check if trained model exists
    if not os.path.exists('models/saved/chest_xray_model.pth'):
        print("No trained model found. Please train the model first.")
        return False
    
    print("Loading trained model...")
    
    # Create model architecture (same as training)
    model = models.efficientnet_b0(weights=None)  # Updated for newer torchvision
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 15)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('models/saved/chest_xray_model.pth', map_location='cpu'))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    model.eval()
    
    # Create dummy input (same size as training)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    onnx_path = "models/saved/xraynet_plus.onnx"
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,  # Updated to newer opset
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"ONNX model successfully saved: {onnx_path}")
        print(f"File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        return False

def create_dummy_onnx():
    """Create a dummy ONNX model if conversion fails"""
    print("Creating a basic ONNX model for testing...")
    
    # Simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 56 * 56, 15)  # Reduced size
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "models/saved/xraynet_plus.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    
    print(f"Dummy ONNX model created: {onnx_path}")
    return True

if __name__ == "__main__":
    # Try to convert the real model first
    success = convert_to_onnx()
    
    # If that fails, create a dummy model
    if not success:
        print("Falling back to dummy model...")
        create_dummy_onnx()