"""BiSeNet face parsing (model.py / resnet.py adapted from MuseTalk, MIT).

Used by `latentsync_driver.face_parse` to keep the generated mouth off objects
held in front of the face. Weights are the ones the MuseTalk service already
ships in the shared model volume (`musetalk/face-parse-bisent`)."""

from .model import BiSeNet  # noqa: F401
