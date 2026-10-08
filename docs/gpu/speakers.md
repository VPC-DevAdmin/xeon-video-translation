# Several speakers: voices, faces and dubbing adaptation

Added 7 Oct 2026 for a 17-minute two-presenter lightboard video. Until then the
pipeline assumed one person: one cloned voice for everyone, and the generated
mouth on the largest face in each frame, which with two people of similar size
switched between them.

## Pipeline

1. **People on screen** (LatentSync service, `latentsync_driver/identities.py`,
   `POST /faces/identities`). Every frame on the 25 fps grid is scanned with the
   detector and 106-point landmarks the renderer already uses. Every 10th frame
   each face also gets an ArcFace embedding (`w600k_r50.onnx`, already in the
   models volume). The embeddings are clustered into people; frames in between
   are linked to people by position. Per person the service keeps the 3-point
   warp landmarks, the detector confidence and a mouth-opening measure
   (inner-lip gap / mouth width from landmarks 52-71). People are numbered left
   to right. The result is cached by content, so renders reuse it.
2. **Who speaks** (backend, `pipeline/speakers.py`, end of the transcribe stage).
   The XTTS speaker encoder embeds 1.5 s windows every 0.5 s; k-means splits
   the voices, and a second voice is accepted only with silhouette >= 0.25.
   Each word takes the majority voice of its windows; a single short word
   between two words of the other voice is smoothed away, and a turn that
   lands within two words after a sentence start moves to the sentence start.
3. **Which face** each voice belongs to: the person whose mouth moves most
   (standardised frame-to-frame change of the opening) while that voice speaks,
   matched one to one.
4. **Segments** are regrouped from the words: never across a speaker change, cut
   at sentence ends once 3 s long, capped at 15 s. Whisper's 30 s windows had
   mixed both people in most segments.
5. **Voices**: TTS already chose its XTTS reference per `speaker`; with clean
   single-speaker segments each person now gets a reference of their own.
6. **Rendering**: when more than one person is visible and a speaker is tied to
   a face, the job renders per speech span (`streaming.py`) in any mode. Spans
   never cross a speaker change (padded spans of two speakers meet at the frame
   midway between them) and each span renders on its speaker's face track
   (`face_track_identity`). The other person keeps the source pixels.

With one voice and several people visible the voice is still tied to the face
that talks. `speakers=one` in the job options, or `SPEAKER_ANALYSIS=false`,
skips the analysis.

## Dubbing adaptation

The two presenters speak about 3.3 words per second with speech in 96% of the
video. A literal Spanish translation read by XTTS needs about 1.7x the time,
and 134 of 169 segments would have exceeded the 1.3x speed ceiling. The LLM
translation now gets a character budget per segment (its slot up to the next
segment's start, at most 1 s past its end, times `SPEECH_CHARS_PER_SECOND` for
the target language, 13 for Spanish) and is told to adapt like a dubbing
writer: keep every fact, name, number, negation and term; drop fillers, false
starts, repetitions and hedges. The review pass and the overrun rewrite get the
same budget and policy. On the first 90 s the adapted text came to 93% of the
budget with the content intact.

TTS fitting also changed for conversational turns: slots under 3 s may go up to
1.5x (`TTS_MAX_SPEED_SHORT`), and retries of an overrunning take use XTTS's own
speed control at 1.2x (`TTS_RETRY_SPEED`) before any waveform stretch.

## Verification (first 90 s of the video, Fast mode)

| check | result |
| --- | --- |
| people found (full 17 min) | 2, each present in 99.5% of frames; 400 s to build, cached |
| voices | 2, silhouette 0.40 to 0.44; intro handover correct to the word |
| voice to face | Seamus to the left face, Chris to the right, scores 0.57 vs -0.61 and 0.43 vs -0.35 |
| speaking frames where the other person's face changed | 0 of 2127 |
| translated lines closest to their own speaker's original voice | 14 of 14 |
| job time | 282 s for 90 s (TTS 49 s, render 254 s) |

Extrapolated to the full 17 minutes in Fast mode: about an hour.

## Limits

* A person who is never detected cannot be tied to a voice; their lines render
  on the largest face.
* Overlapping speech (both talking at once) goes to whichever voice wins the
  window vote.
* `SPEECH_CHARS_PER_SECOND` covers Latin-script languages; others get a seconds
  budget only.
