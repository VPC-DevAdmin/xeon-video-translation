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

## Never fail a long job on one line (7 Oct 2026, evening)

The first full 17-minute run failed two minutes in on one line ("If it's a bad
output, we adjust the dial. Okay." -> 5.3 s of Spanish for a 2.3 s slot). With
169 lines of fast conversation some line always loses a one-slot-at-a-time
fit, so fitting is now planned per span:

* **Timeline plan** (`tts._plan_timeline`): every line's best take is made
  first; then each line starts at its source onset or right after the previous
  one, and is sped up only as far as its own slot needs and at most a common
  factor `g`. The smallest `g` that keeps every start within
  `TTS_MAX_DRIFT_SECONDS` (1.5 s) and ends the span in time wins. Overruns
  borrow the following pauses before anything is sped up; the lip sync follows
  the audio, so a late start only shifts speech against gestures.
* **Turn overlap**: a span's last line may run 0.6 s into the next span
  (`STREAM_TAIL_OVERLAP_SECONDS`), mixed under the next voice, when nothing
  else fits ("Understood." right before the other person speaks).
* **Last resort** 1.7x (`TTS_MAX_SPEED_LAST_RESORT`) instead of failing; such
  lines are marked `last_resort` in the span's timing file.
* **Budget enforcement**: a translation more than 10% over its character budget
  is sent back with the actual count (twice at most; shortest safe result
  kept); a draft over twice the budget is retranslated without context, and
  lines of three words or fewer never get context ("Okay." had become five
  sentences of context). A review may not push a line back over budget.
* **Verification**: digits and % are spelled out on both sides before the
  word-for-word check ("88" vs "ochenta y ocho"), and a 92% match counts as
  ambiguous (take kept whole) rather than different speech.

Full-video TTS sweep after these changes: all 72 spans synthesized (5 of them
re-run after the last fixes); 153 lines, median speed 1.00, 6 above 1.3x,
1 last resort, largest start drift 1.5 s.

## Quality pass: audio, turned heads, beard (8 Oct 2026)

Three problems on the 2-minute Quality run of the lightboard video, measured
before fixing. Speech quality is SQUIM's objective PESQ (4.5 is perfect).

* **Scratchy speech.** Three causes. (1) The main one, found only after the
  first two were fixed: `tts._prepend_silence` concatenated an `anullsrc` lead
  in front of each span's speech, and ffmpeg negotiated that graph to unsigned
  8-bit, so every span with a pause before its first word was quantized to 256
  levels (229 distinct sample values in a take that had 20,400). Speech-only
  PESQ of the same takes fell from 3.86 to 3.26 at that step; it now uses
  `adelay` on the speech and stays bit-exact. (2) Every line the timeline plan sped up,
  even by 10%, went through rubberband, which wrecked it: on the same XTTS line
  rubberband scored 1.82 at 1.1x and 1.44 at 1.25x (R3 `--fine` no better), ffmpeg
  atempo 3.87 and 3.55, and XTTS's own speed control 4.11 and 3.76. Speed-ups
  now re-speak the line at the planned XTTS speed (`tts_native_speed_min` 1.05,
  up to `tts_native_speed_takes` 2 verified takes, compounded with the speed
  the kept take was made at), and atempo covers whatever is left
  (`tts._respeak_faster`, `tts._maybe_time_stretch`). (2) XTTS cloned the voice
  from the 16 kHz copy made for recognition. The audio stage now also writes
  `voice_reference.wav` at 24 kHz, used for cloning and for the original sound
  in untranslated gaps: the raw take went from 3.11 to 3.95 (the source
  recording scores 3.64).
* **Distorted mouths on turned heads.** LatentSync warps every face to a
  frontal template; the far half of the mouth is squeezed and smears into the
  cheek, growing with head yaw (nose offset from the eye midpoint along the eye
  line, in eye distances). Clean below about 0.35, visibly squeezed at 0.36 to
  0.43, smeared from 0.55; 19% of speaking frames in the clip were above 0.4.
  Frames above `YAW_ENTER` 0.45 keep the source mouth until yaw falls below
  `YAW_EXIT` 0.38, runs up to 8 frames apart are joined, and the usual 2-frame
  gate margin applies (`face_parse.turned_frames`, `LATENTSYNC_YAW_GATE=0`
  disables). The switch is a cut, not a dissolve.
* **Beard vanishing at 19.6 to 21.1 s.** Exactly the frames where the
  translation had already ended, which render with the closed-mouth reference
  frame. That render drops the lower face texture, so on silent frames only the
  lips (parse classes 11 to 13, dilated 15 px and feathered) are pasted.
