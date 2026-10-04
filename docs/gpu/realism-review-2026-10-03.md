# Realism review of the minute-length English video

## Assessment

The speaker is recognizable, and individual frames can be convincing. The full performance does not yet look consistently like a naturally recorded person. Delivery, facial expression and behavior during pauses are now more important than simply increasing render resolution or sampling steps.

This review used full-video contact sheets, dense samples around the reported pause, all-frame facial landmarks, waveform silence detection, the raw synthesis audio, render code and GPU ASR. It does not provide an independent listening score or a validated phoneme-level audiovisual sync score. Audible timbre and stutter still need perceptual review; ASR often normalizes disfluencies.

## The three reported problems

### Serious tone

The script was too formal: long explanatory sentences and phrases such as “preserve the meaning, the personality, and the feeling” encourage a presentation cadence. The source is only 1.77 seconds, with approximately 0.72 seconds of recognized speech. It provides little evidence of friendly emphasis, conversational rhythm or the speaker's broader vocal range. The portrait also starts with a furrowed brow and neutral mouth.

The installed XTTS model offers temperature, top-p, repetition penalty and speed controls. These change sampling and timing, but do not amount to a reliable “friendly” instruction. Higher temperature can increase variability without necessarily producing warmth. See the [XTTS documentation](https://docs.coqui.ai/en/latest/models/xtts.html). The installed Qwen Base cloning model has reference-based voice conditioning; the separate CustomVoice and VoiceDesign variants expose instruction controls. Their documented controls should not be assumed to work on this Base clone. See [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

Immediate approach: conversational wording, native speaking pace, shorter coherent passages, multiple takes, and a longer friendly reference when available. A 20–30 second clean conversational recording would provide much more useful variation than two words; this is a practical recommendation, not a claimed minimum requirement. An expressively controlled cloning candidate such as Chatterbox is a next trial, not an installed improvement. Its [official implementation](https://github.com/resemble-ai/chatterbox) documents cloning and CFG/exaggeration controls for the original English model.

### Pause and sleepy face

The delivered video contains silence from **21.7117 to 24.1752 seconds: 2.4635 seconds**. It already exists in raw XTTS output at 24.2601–27.1203 seconds, lasting 2.8602 seconds before the tempo change. Rendering did not create the gap. Another measured gap occurs at 48.5557–49.6391 seconds.

During the first pause, the mouth closes and the forward gaze/expression are held. Landmark estimates do not show a two-second eyelid closure: the longest low-aperture event anywhere in the video is about 0.32 seconds, and median openness during the pause is near the clip's open-eye baseline. This supports a distinction between a brief blink and the prolonged still, disengaged expression around it. These geometry estimates are exploratory, not a validated blink-quality metric.

The fix is to reject unexplained long silent gaps before animation, regenerate defective takes, and provide plausible quiet behavior: gaze changes, small head adjustments and expression continuity. Simply adding more blinks is unlikely to solve the impression. Pause editing must preserve spoken words and be applied before generating motion; cutting audio under an existing video would break synchronization.

### Audio quality and stuttering

Two avoidable processing choices were confirmed:

- The 24 kHz XTTS output was reduced to 16 kHz for animation conditioning, and that same copy was used as the delivered soundtrack. This unnecessarily removed high-frequency information.
- The raw narration was accelerated by a factor of 1.11883. Time stretching can affect texture, although this does not prove it caused every reported stutter.

The raw signal touches full scale once, which is insufficient evidence for widespread clipping. No claim is made that a noise filter or higher bitrate alone can repair model-generated repetitions or unnatural phonemes. Keep a native-rate master, generate a separate 16 kHz conditioning copy, avoid unnecessary speed changes, and re-synthesize suspect phrases. Evaluate phoneme-level defects by listening as well as checking text.

## Remaining visual gaps

1. **Expression follows sound more than meaning.** Brow, cheek and mouth-corner behavior seldom convey warmth or emphasis. The current adapter has no explicit expression, gaze or performance-direction controls.
2. **Idle behavior holds too steadily.** Pauses need believable expression transitions and gaze behavior rather than a prolonged forward stare.
3. **Upper-body motion is limited.** Shoulder movement, breathing and small posture changes do not match the richness of a recorded conversational performance.
4. **Fine detail is softened.** Skin, beard, eyelids and teeth remain softer than the reference, with generated detail needing temporal review. Upscaling alone cannot establish authentic detail.
5. **Sync needs stronger testing.** Plausible mouth opening is not enough. Check closures on consonants, audiovisual offset and chunk boundaries. This review has not established a numeric sync score.

## Changes and controlled tests completed

- Added separate native-rate playback audio to the FlashHead benchmark; conditioning remains 16 kHz.
- Added sample-accurate duration validation between those tracks and an incomplete-frame check.
- Added four passing regression tests for matching timelines, rounding tolerance and invalid input.
- Generated the same new 37-word conversational script with XTTS at temperature 0.65 and 0.85, and Qwen Base. All three had zero ASR word errors. This does not prove which sounds best.
- Produced a **14.08-second FlashHead Pro preview** using the 0.65 XTTS take. It has 24 kHz AAC playback, no tempo/pitch alteration, and one 0.935-second cut wholly inside measured silence outside word timestamps and guards. A 5 ms fade at the cut prevents a hard sample discontinuity. Motion was generated anew after the cut.
- GPU model generation took 20.40 seconds, plus 2.42 seconds preparation, excluding encoding. The final encoded preview retained all 37 words and decoded without errors.

The preview is a controlled candidate, not a claim that friendliness, stuttering or human realism is solved. Review it alongside the other native-rate voice takes before committing to another minute-long render.

## Next acceptance gates

- [ ] Select a voice take that sounds conversational and preserves identity, with no audible syllable repeats, metallic transitions or exaggerated seriousness.
- [ ] No unexplained silent gap over roughly one second; allow intentional pauses explicitly.
- [ ] Verify eyelid, gaze, cheek and brow behavior during silence and emphasis on multiple clips.
- [ ] Confirm complete narration, native audio rate, correct duration and measured audiovisual timing in the final encoded file.
- [ ] Compare a model with stronger expression control against FlashHead using the same accepted audio and reference.
- [ ] Repeat the accepted combination at one-minute length and review the whole performance, including joins and ending.
