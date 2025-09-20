from .luna_detailer import Luna_Detailer
from .luna_mediapipe_detailer import Luna_MediaPipe_Detailer
from .luna_mediapipe_segs import Luna_MediaPipe_Segs

NODE_CLASS_MAPPINGS = {
    "Luna_Detailer": Luna_Detailer,
    "Luna_MediaPipe_Detailer": Luna_MediaPipe_Detailer,
    "Luna_MediaPipe_Segs": Luna_MediaPipe_Segs,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Luna_Detailer": "Luna Detailer",
    "Luna_MediaPipe_Detailer": "Luna MediaPipe Detailer",
    "Luna_MediaPipe_Segs": "Luna MediaPipe SEGS",
}