"""Explicit, shared translation presets. Avatar sessions use their own API."""

MODES = {
    "fast": {
        "lipsync_backend": "latentsync",
        "latentsync_steps": 10,
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "quality": {
        "lipsync_backend": "latentsync",
        "latentsync_steps": 40,
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "stream": {
        # Streaming translation: per speech span, published while the rest renders.
        "lipsync_backend": "latentsync",
        "latentsync_steps": 10,
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
    "dub": {
        "lipsync_backend": "none",
        "tts_backend": "auto",
        "enable_output_stabilization": "false",
    },
}
