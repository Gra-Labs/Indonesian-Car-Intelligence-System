#!/usr/bin/env python3
"""
Script untuk menguji pemuatan model klasifikasi mobil
Untuk memastikan model dapat dimuat dengan benar
"""

import torch
import os
from models.car_classifier import CarClassifier, load_trained_convnext_classifier

# Konfigurasi
CLASSIFIER_MODEL_PATH = 'models/best_classifier_augmented_v2_convnext_base.pt'
NUM_CLASSES_CLASSIFIER = 8
CLASSIFIER_CLASS_NAMES = ['City_car', 'Commercial_Van', 'Hatchback', 'LCGC', 'MPV', 'Pickup_truck', 'SUV', 'Sedan']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan device: {device}")

def test_model_loading():
    """Test different methods to load the classifier model"""
    print("=" * 60)
    print("TESTING MODEL LOADING METHODS")
    print("=" * 60)
    
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        print(f"❌ Error: Model file tidak ditemukan di {CLASSIFIER_MODEL_PATH}")
        return
    
    print(f"✅ Model file ditemukan: {CLASSIFIER_MODEL_PATH}")
    
    # Test Method 1: Direct loading function
    print("\n🔄 Method 1: Testing direct loading function...")
    try:
        model1 = load_trained_convnext_classifier(
            CLASSIFIER_MODEL_PATH, 
            num_classes=NUM_CLASSES_CLASSIFIER, 
            device=device
        )
        print("✅ Method 1 SUCCESS: Model loaded using direct function")
        print(f"   Model type: {type(model1)}")
        print(f"   Device: {next(model1.parameters()).device}")
        print(f"   Total parameters: {sum(p.numel() for p in model1.parameters())}")
        
        # Test inference
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model1(test_input)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
        print(f"   Test inference SUCCESS: Output shape {output.shape}")
        print(f"   Predicted class: {CLASSIFIER_CLASS_NAMES[predicted_class.item()]}")
        print(f"   Confidence: {probs.max().item():.3f}")
        
    except Exception as e:
        print(f"❌ Method 1 FAILED: {e}")
    
    # Test Method 2: Manual loading with CarClassifier
    print("\n🔄 Method 2: Testing manual loading with CarClassifier...")
    try:
        model2 = CarClassifier(num_classes=NUM_CLASSES_CLASSIFIER).to(device)
        saved_state_dict = torch.load(CLASSIFIER_MODEL_PATH, map_location=device)
        
        # Try direct loading
        try:
            model2.load_state_dict(saved_state_dict, strict=True)
            print("✅ Method 2a SUCCESS: Direct state dict loading")
        except RuntimeError as e:
            print(f"⚠️  Method 2a PARTIAL: Direct loading failed: {e}")
            print("   Trying key mapping...")
            
            # Try key mapping
            new_state_dict = {}
            for key, value in saved_state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            missing_keys, unexpected_keys = model2.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Method 2b SUCCESS: Key mapping successful")
            print(f"   Missing keys: {len(missing_keys)}")
            print(f"   Unexpected keys: {len(unexpected_keys)}")
            
            if missing_keys:
                print(f"   Missing: {missing_keys[:3]}...")
            if unexpected_keys:
                print(f"   Unexpected: {unexpected_keys[:3]}...")
        
        model2.eval()
        print(f"   Model type: {type(model2)}")
        print(f"   Device: {next(model2.parameters()).device}")
        
        # Test inference
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model2(test_input)
            probs = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1)
        print(f"   Test inference SUCCESS: Output shape {output.shape}")
        print(f"   Predicted class: {CLASSIFIER_CLASS_NAMES[predicted_class.item()]}")
        print(f"   Confidence: {probs.max().item():.3f}")
        
    except Exception as e:
        print(f"❌ Method 2 FAILED: {e}")
    
    print("\n=" * 60)
    print("MODEL LOADING TEST COMPLETE")
    print("=" * 60)

def inspect_model_file():
    """Inspect the saved model file structure"""
    print("\n🔍 INSPECTING MODEL FILE STRUCTURE")
    print("-" * 40)
    
    try:
        checkpoint = torch.load(CLASSIFIER_MODEL_PATH, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            print(f"✅ Model file contains a dictionary with {len(checkpoint)} keys")
            print("\nTop-level keys:")
            for i, key in enumerate(list(checkpoint.keys())[:10]):
                print(f"   {i+1}. {key}")
            if len(checkpoint) > 10:
                print(f"   ... and {len(checkpoint) - 10} more keys")
                
            # Check for common patterns
            model_keys = [k for k in checkpoint.keys() if k.startswith('model.')]
            feature_keys = [k for k in checkpoint.keys() if 'features' in k]
            classifier_keys = [k for k in checkpoint.keys() if 'classifier' in k]
            
            print(f"\nKey patterns:")
            print(f"   Keys starting with 'model.': {len(model_keys)}")
            print(f"   Keys containing 'features': {len(feature_keys)}")
            print(f"   Keys containing 'classifier': {len(classifier_keys)}")
            
        else:
            print(f"⚠️  Model file contains: {type(checkpoint)}")
            
    except Exception as e:
        print(f"❌ Error inspecting model file: {e}")

if __name__ == "__main__":
    test_model_loading()
    inspect_model_file()

