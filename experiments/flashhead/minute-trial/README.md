# Minute-length creation trial

These executed scripts are evidence with fixed lab paths, not production adapters. Their model generation ran on XE7740; replay requires the isolated environments described in the parent README and the exact recorded revisions.

FlashHead Pro used the original 1.77-second clip for portrait and XTTS voice reference, plus the new script in narration.txt. The final MP4 is 59.52 seconds, 512 square, 25 FPS. narration.json records the intended 60-second audio target; use media-manifest.json for measured final duration. Verify duration and complete tail delivery before calling an exact 60-second replay successful.

Qwen3-TTS ran in an isolated system-site-packages virtual environment with qwen-tts 0.1.1, CUDA BF16 and SDPA. The recorded model revision is fd4b254389122332181a7c3db7f27e918eec64e3. The executed downloader resolved the head before recording it; pin that revision when replaying. The inherited environment has dependency conflicts and is not ready for service integration.

The ASR reports check narration content, not perceived voice similarity, lip sync or facial quality. No user video, voice reference, access token or environment secret is stored in this folder.
