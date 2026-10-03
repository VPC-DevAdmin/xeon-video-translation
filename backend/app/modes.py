"""Explicit, shared translation presets. Avatar sessions use their own API."""

MODES = {
    "fast": {
        "lipsync_backend": "musetalk",
        "tts_backend": "auto",
        "musetalk_face_restore": "none",
        "musetalk_blend_mode": "mouth",
        "enable_output_stabilization": "false",
    },
    "quality": {
        "lipsync_backend": "latentsync",
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "dub": {
        "lipsync_backend": "none",
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
}
