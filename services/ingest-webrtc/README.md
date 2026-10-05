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

The `/assistant` path plays one prepared acknowledgement, then a continuous idle
loop until the reply is ready. This is the default because independently rendered
progress phrases add visible face joins. The service renders 0.8 seconds of
silence after each acknowledgement and at the end of a reply, then selects a
real frame near the portrait's rest pose. Face transitions and idle wraps do not
blend whole images.

Set `ASSISTANT_PROGRESS_FILLERS=1` to compare the older multi-phrase tablet
sequence. `ASSISTANT_CLIP_TAIL_SECONDS` controls the silent render tail (minimum
0.35 seconds). Changing the tail duration creates a new clip cache key. GPU-side
review should compare join discontinuity, ghosting, reply latency, and dropped
frames before enabling the multi-phrase path by default.

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
