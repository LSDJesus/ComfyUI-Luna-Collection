import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Type checking imports - using hasattr check instead of isinstance for better compatibility
# from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

POSE_LANDMARKS_FEET = [27, 28, 29, 30, 31, 32]
POSE_LANDMARKS_TORSO = [11, 12, 23, 24]
FACE_MESH_LANDMARKS_EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
FACE_MESH_LANDMARKS_MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

class Mediapipe_Engine:
    def __init__(self):
        self.models = {}

    def _get_model(self, model_type, confidence):
        model_name_map = {
            'face': 'face_mesh', 'eyes': 'face_mesh', 'mouth': 'face_mesh',
            'full_body (bbox)': 'pose', 'feet': 'pose', 'torso': 'pose',
            'person (segmentation)': 'selfie', 'hands': 'hands'
        }
        model_name = model_name_map.get(model_type, model_type)
        
        model_key = f"{model_name}_{confidence}"
        if model_name == 'selfie': model_key = "selfie_segmentation"

        if model_key not in self.models:
            try:
                if model_name == 'hands': 
                    self.models[model_key] = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=confidence)  # type: ignore
                elif model_name == 'pose': 
                    self.models[model_key] = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=confidence)  # type: ignore
                elif model_name == 'face_mesh': 
                    self.models[model_key] = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=confidence)  # type: ignore
                elif model_name == 'selfie': 
                    self.models[model_key] = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)  # type: ignore
            except AttributeError as e:
                print(f"Failed to create {model_name} model: {e}")
                return None
        return self.models.get(model_key)

    def _create_mask_from_landmarks(self, image_shape, landmarks, padding, blur, min_area=500, region_filter=None):
        if not landmarks:
            print("No landmarks detected, skipping mask.")
            return None
        H, W, _ = image_shape

        list_of_points = []
        # Check if landmarks is a MediaPipe landmark list object
        if hasattr(landmarks, 'landmark'):
            # It's a NormalizedLandmarkList or similar MediaPipe object
            list_of_points = landmarks.landmark
        else:
            # It's already a list of landmarks
            list_of_points = landmarks

        points = np.array([(lm.x * W, lm.y * H) for lm in list_of_points if hasattr(lm, 'x')], dtype=np.int32)
        if len(points) < 3:
            print("Too few points for mask, skipping.")
            return None

        hull = cv2.convexHull(points)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 255, -1)

        if padding > 0:
            kernel = np.ones((padding, padding), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        if blur > 0:
            blur_kernel_size = blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

        area = np.count_nonzero(mask)
        print(f"Mask area: {area}")
        if area < min_area:
            print("Mask area too small, skipping.")
            return None
        # Region-based filtering (for eyes)
        if region_filter is not None:
            y_indices = np.where(mask > 0)[0]
            if len(y_indices) > 0:
                y_mean = np.mean(y_indices)
                if not region_filter(y_mean, H):
                    print("Mask region filter failed, skipping.")
                    return None
        return mask

    def process_and_create_mask(self, image, options):
        final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        model_type = options['model_type']
        confidence = options['confidence']

        # Lower confidence for face/eyes/mouth
        if model_type in ['face', 'eyes', 'mouth']:
            confidence = min(confidence, 0.15)

        model = self._get_model(model_type, confidence)
        if not model:
            print(f"Model not found for type: {model_type}")
            return final_mask

        results = model.process(image)

        if model_type == 'person (segmentation)':
            if hasattr(results, 'segmentation_mask'):
                seg_mask = np.where(results.segmentation_mask > 0.5, 255, 0).astype(np.uint8)
                print(f"Segmentation mask area: {np.count_nonzero(seg_mask)}")
                return seg_mask
            print("No segmentation mask found.")
            return final_mask

        landmark_sets = []
        if model_type == 'hands' and results.multi_hand_landmarks:
            print(f"Detected {len(results.multi_hand_landmarks)} hands.")
            landmark_sets.extend(results.multi_hand_landmarks)
        elif model_type in ['face', 'eyes', 'mouth'] and results.multi_face_landmarks:
            print(f"Detected {len(results.multi_face_landmarks)} faces.")
            for face_landmarks in results.multi_face_landmarks:
                if model_type == 'eyes':
                    landmark_sets.append([face_landmarks.landmark[i] for i in FACE_MESH_LANDMARKS_EYES])
                elif model_type == 'mouth':
                    landmark_sets.append([face_landmarks.landmark[i] for i in FACE_MESH_LANDMARKS_MOUTH])
                else:
                    landmark_sets.append(face_landmarks)
        elif model_type in ['full_body (bbox)', 'feet', 'torso'] and results.pose_landmarks:
            print(f"Detected pose landmarks for {model_type}.")
            if model_type == 'feet':
                landmark_sets.append([results.pose_landmarks.landmark[i] for i in POSE_LANDMARKS_FEET])
            elif model_type == 'torso':
                landmark_sets.append([results.pose_landmarks.landmark[i] for i in POSE_LANDMARKS_TORSO])
            else:
                landmark_sets.append(results.pose_landmarks)

        # Lower min_area for face/eyes/mouth
        min_area = 500
        if model_type in ['face', 'eyes', 'mouth']:
            min_area = 100

        for idx, landmarks in enumerate(landmark_sets):
            # Region filter for eyes: exclude masks too high in the image (likely hair)
            region_filter = None
            if model_type == 'eyes':
                def eyes_region_filter(y_mean, H):
                    # Only accept masks in the lower 70% of the image
                    return y_mean > H * 0.3
                region_filter = eyes_region_filter
            # Increase mask_padding for feet
            mask_padding = options['mask_padding']
            if model_type == 'feet':
                mask_padding = max(mask_padding, 60)
            instance_mask = self._create_mask_from_landmarks(
                image.shape, landmarks, mask_padding, options['mask_blur'], min_area=min_area, region_filter=region_filter
            )
            if instance_mask is not None:
                print(f"Adding mask for {model_type} instance {idx}, area: {np.count_nonzero(instance_mask)}")
                final_mask = np.maximum(final_mask, instance_mask)
            else:
                print(f"Skipped mask for {model_type} instance {idx}")

        return final_mask