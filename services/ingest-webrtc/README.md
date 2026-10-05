# ingest-webrtc

WebRTC webcam ingest for the GPU track. The browser streams its webcam to
this service; the service records it under `/jobs/ingest/<session>/` and,
on stop, submits it to the backend as a normal job with the chosen mode's
parameters. See [docs/gpu/README.md](../../docs/gpu/README.md).

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| `POST` | `/sessions/{id}/offer` | `{sdp, type, target_language, source_language?, mode}` | `{session_id, sdp, type}` SDP answer |
| `POST` | `/sessions/{id}/stop` | – | `{session_id, input_path, job_id}` |
| `GET` | `/sessions` | – | active sessions |
| `GET` | `/health` | – | status |

`mode` is `fast` (mode 1) or `quality` (mode 2). The mapping from mode to
backend job parameters lives in one table at the top of `app/main.py`.

## Running outside Docker

```bash
cd services/ingest-webrtc
python -m venv .venv && source .venv/bin/activate
pip install -e .
JOB_ARTIFACTS_DIR=../../jobs BACKEND_URL=http://localhost:8088 \
  uvicorn app.main:app --port 8091
```

Then open the frontend's `/live` page.

## Video assistant acknowledgement quality

The `/assistant` path rotates prepared opening acknowledgements, then shows the
persona looking away or down at a tablet while progress phrases fill the wait.
The service renders 0.8 seconds of silence after each acknowledgement and at
the end of a reply, then selects a real frame near the portrait's rest pose.
Face transitions and idle wraps do not blend whole images. Background idle
replacement keeps the existing generated take on screen until its replacement
has rendered. The thinking or tablet pose remains active until the final spoken
acknowledgement, which is scheduled just before the reply. By default a
progress phrase still talking when the reply is ready is cut off with a fade,
the way a person interrupts themselves (`ASSISTANT_FILLER_CUTOFF=1`); set it to 0
to let the phrase finish, which delays the reply by up to the phrase's length.
Joins between footage cut when the two frames nearly match and blend over three
frames when they do not, so neither pops nor whole-face dissolves appear.
New progress phrases are queued only within the first 12 seconds of a turn
(`ASSISTANT_PROGRESS_HORIZON_SECONDS`), so a long preplanned bridge does not
needlessly delay an answer that is already ready.
The front listening loop is limited to subtle motion close to the speech pose;
the separate thinking and tablet loops still move the head and gaze. Tune the
front-loop limit with `ASSISTANT_FRONT_IDLE_MAX_DELTA` (sampled RGB difference,
default 6); the loop is never trimmed below `ASSISTANT_FRONT_IDLE_MIN_SECONDS`
(default 4), since a two-second listening loop reads as repetitive and the
adaptive join covers a larger difference at a clip start.

Set `ASSISTANT_PROGRESS_FILLERS=0` to disable the progress phrases and look-away
sequence. `ASSISTANT_CLIP_TAIL_SECONDS` controls the silent render tail (minimum
0.35 seconds). Changing the tail duration creates a new clip cache key. GPU-side
review should compare join discontinuity, ghosting, reply latency, and dropped
frames for the multi-phrase path.

## Networking

WebRTC media runs over UDP, so the compose overlay runs this service with
`network_mode: host`. On a single machine nothing else is needed. If the
browser is on another machine, set `INGEST_PUBLIC_IP` and open UDP; a TURN
server is the robust answer for anything behind NAT.

## Not yet

- Live consumers (mode 3 avatar, sub-utterance streaming): the recorder is
  the only consumer today. Adding a second consumer that reads frames from
  the track as they arrive is the first step of mode 3.
- Auth. Same posture as the backend: localhost only.
