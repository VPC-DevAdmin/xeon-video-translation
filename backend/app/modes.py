"""Explicit, shared translation presets. Avatar sessions use their own API."""

MODES = {
    "fast": {
        "lipsync_backend": "latentsync",
        "latentsync_steps": 10,
        "latentsync_service_tier": "fast",
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "quality": {
        "lipsync_backend": "latentsync",
        "latentsync_steps": 40,
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "dub": {
        "lipsync_backend": "none",
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
}
