import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import json
import time
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import threading
import uuid

class AdaptiveObjectDetector:
    def __init__(self, base_path="Objects"):
        self.base_path = base_path
        self.detection_model = None
        self.feature_extractor = None
        self.object_database = {}
        self.folder_mappings = {}
        self.confidence_threshold = 0.5
        self.learning_rate = 0.001
        self.detection_history = []
        self.clustering_threshold = 0.3
        
        # Initialize directories
        os.makedirs(base_path, exist_ok=True)
        
        # Load or initialize models
        self.setup_models()
        self.load_database()
        
        # Start folder monitoring thread
        self.monitor_folders()
    
    def setup_models(self):
        """Initialize the detection and feature extraction models"""
        # Use a pre-trained MobileNetV2 as feature extractor
        base_model = keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Feature extractor (frozen base model)
        self.feature_extractor = keras.Model(
            inputs=base_model.input,
            outputs=base_model.layers[-2].output
        )
        self.feature_extractor.trainable = False
        
        # Simple object detector using YOLO-like approach
        self.detection_model = self.create_detection_model()
    
    def create_detection_model(self):
        """Create a simple object detection model"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(416, 416, 3)),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(5, activation='sigmoid')  # x, y, w, h, confidence
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def extract_features(self, image):
        """Extract features from an image crop"""
        if image.shape[:2] != (224, 224):
            image = cv2.resize(image, (224, 224))
        
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        features = self.feature_extractor.predict(image, verbose=0)
        return features.flatten()
    
    def detect_objects(self, frame):
        """Detect objects in the frame using background subtraction and contours"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 50:  # Minimum size threshold
                    detections.append((x, y, w, h))
        
        return detections
    
    def classify_object(self, image_crop, bbox):
        """Classify detected object and assign to folder"""
        features = self.extract_features(image_crop)
        
        # Find similar objects using clustering
        best_match = self.find_best_match(features)
        
        if best_match:
            object_id = best_match
            folder_name = self.get_folder_name(object_id)
        else:
            # Create new object
            object_id = str(uuid.uuid4())[:8]
            folder_name = object_id
            self.object_database[object_id] = {
                'features': [features],
                'bbox_history': [bbox],
                'folder_name': folder_name,
                'created_at': datetime.now().isoformat(),
                'detection_count': 1
            }
            
            # Create folder
            folder_path = os.path.join(self.base_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
        
        # Update object data
        if object_id in self.object_database:
            self.object_database[object_id]['features'].append(features)
            self.object_database[object_id]['bbox_history'].append(bbox)
            self.object_database[object_id]['detection_count'] += 1
        
        # Save cropped image
        self.save_object_image(image_crop, folder_name)
        
        return object_id, folder_name
    
    def find_best_match(self, features):
        """Find the best matching object based on feature similarity"""
        if not self.object_database:
            return None
        
        best_match = None
        best_similarity = 0
        
        for object_id, data in self.object_database.items():
            for stored_features in data['features']:
                similarity = self.calculate_similarity(features, stored_features)
                if similarity > best_similarity and similarity > self.confidence_threshold:
                    best_similarity = similarity
                    best_match = object_id
        
        return best_match
    
    def calculate_similarity(self, features1, features2):
        """Calculate cosine similarity between two feature vectors"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def get_folder_name(self, object_id):
        """Get the current folder name for an object (handles renames)"""
        if object_id in self.folder_mappings:
            return self.folder_mappings[object_id]
        
        # Check if folder was renamed
        original_folder = self.object_database[object_id]['folder_name']
        current_folders = [f for f in os.listdir(self.base_path) 
                          if os.path.isdir(os.path.join(self.base_path, f))]
        
        # If original folder doesn't exist, find the renamed one
        if original_folder not in current_folders:
            # Look for folders that might be renamed versions
            for folder in current_folders:
                folder_path = os.path.join(self.base_path, folder)
                if len(os.listdir(folder_path)) > 0:  # Has images
                    # Check if this could be our renamed folder
                    self.folder_mappings[object_id] = folder
                    return folder
        
        return original_folder
    
    def save_object_image(self, image_crop, folder_name):
        """Save cropped object image to its folder"""
        folder_path = os.path.join(self.base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"object_{timestamp}.jpg"
        filepath = os.path.join(folder_path, filename)
        
        cv2.imwrite(filepath, image_crop)
    
    def monitor_folders(self):
        """Monitor for folder renames in a separate thread"""
        def folder_monitor():
            while True:
                try:
                    current_folders = [f for f in os.listdir(self.base_path) 
                                     if os.path.isdir(os.path.join(self.base_path, f))]
                    
                    # Check for renamed folders
                    for object_id, data in self.object_database.items():
                        original_name = data['folder_name']
                        if original_name not in current_folders:
                            # Find potential renamed folder
                            for folder in current_folders:
                                if folder not in [d['folder_name'] for d in self.object_database.values()]:
                                    folder_path = os.path.join(self.base_path, folder)
                                    if os.path.exists(folder_path) and len(os.listdir(folder_path)) > 0:
                                        self.folder_mappings[object_id] = folder
                                        break
                    
                    time.sleep(2)  # Check every 2 seconds
                except Exception as e:
                    print(f"Folder monitoring error: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=folder_monitor, daemon=True)
        thread.start()
    
    def load_database(self):
        """Load object database from file"""
        db_path = os.path.join(self.base_path, "object_database.json")
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    data = json.load(f)
                    # Convert features back to numpy arrays
                    for obj_id, obj_data in data.items():
                        obj_data['features'] = [np.array(f) for f in obj_data['features']]
                    self.object_database = data
            except Exception as e:
                print(f"Error loading database: {e}")
    
    def save_database(self):
        """Save object database to file"""
        db_path = os.path.join(self.base_path, "object_database.json")
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_db = {}
            for obj_id, obj_data in self.object_database.items():
                serializable_db[obj_id] = obj_data.copy()
                serializable_db[obj_id]['features'] = [f.tolist() for f in obj_data['features']]
            
            with open(db_path, 'w') as f:
                json.dump(serializable_db, f, indent=2)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        
        print("Starting Adaptive Object Detection System...")
        print("Press 'q' to quit, 's' to save database")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Process each detection
            for bbox in detections:
                x, y, w, h = bbox
                
                # Extract object crop
                object_crop = frame[y:y+h, x:x+w]
                
                # Classify object
                object_id, folder_name = self.classify_object(object_crop, bbox)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, folder_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show confidence
                confidence = len(self.object_database[object_id]['features']) if object_id in self.object_database else 1
                cv2.putText(frame, f"Conf: {confidence}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Adaptive Object Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_database()
                print("Database saved!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_database()

# Usage example
if __name__ == "__main__":
    detector = AdaptiveObjectDetector()
    detector.run_detection()
