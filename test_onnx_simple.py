import onnxruntime as ort
import numpy as np
import os

def test_onnx_simple():
    print("Testing ONNX model...")
    
    # Check if ONNX model exists
    model_path = "models/saved/xraynet_plus.onnx"
    
    if not os.path.exists(model_path):
        print("ERROR: ONNX model not found!")
        return False
    
    try:
        # Load ONNX model
        session = ort.InferenceSession(model_path)
        print("SUCCESS: ONNX model loaded successfully!")
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Create dummy input (same as training: 1, 3, 224, 224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Apply sigmoid to get probabilities (same as training)
        probabilities = 1 / (1 + np.exp(-predictions))
        
        # Disease classes (same as training)
        diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No Finding'
        ]
        
        # Get top prediction
        top_idx = np.argmax(probabilities)
        top_disease = diseases[top_idx]
        top_confidence = probabilities[top_idx]
        
        print(f"SUCCESS: Inference successful!")
        print(f"Top prediction: {top_disease} ({top_confidence:.1%})")
        print(f"All probabilities:")
        for i, disease in enumerate(diseases):
            print(f"  {disease}: {probabilities[i]:.1%}")
            
        return True
        
    except Exception as e:
        print(f"ERROR testing ONNX model: {e}")
        return False

if __name__ == "__main__":
    test_onnx_simple()