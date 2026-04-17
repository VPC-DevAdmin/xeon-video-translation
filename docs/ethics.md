# Responsible use

This project demonstrates open-source video translation. The same techniques
that make a transparent demo possible can also be misused. Read this before
running it on anyone's likeness — including your own.

## Required practices

- **Consent.** Only record people who have explicitly agreed to be recorded
  and to have their voice and likeness translated and shown publicly. At a
  conference, ask each subject; do not silently record audience members.
- **Disclosure.** Any output you share — slide, social post, demo reel —
  must be labeled as AI-generated. The watermark covers this for the video
  itself; you are responsible for the surrounding context.
- **Watermarking.** A persistent on-screen "AI-translated" watermark is
  applied in Stage 6 (M4). It is enabled by default and can only be
  disabled via the `ENABLE_WATERMARK=false` env var, which exists solely for
  internal pipeline testing. Do not ship outputs with the watermark off.
- **Provenance metadata.** When `ENABLE_C2PA=true` (M6+), output MP4s carry
  a [C2PA](https://c2pa.org/) manifest declaring AI generation. Recommended
  for any public-facing share.

## Things this tool is not for

- Impersonating public figures, politicians, executives, or celebrities.
- Producing content of any private individual without their explicit consent.
- Creating content that could be mistaken for an authentic recording of a
  real event (e.g., fake apologies, fake confessions, fake endorsements).
- Bypassing platform AI-disclosure requirements.

If your use case needs the watermark gone, the C2PA manifest stripped, or
the consent step skipped, this is not the right tool — and probably not the
right project for you to be contributing to.

## Conference demo checklist

Before each session:

- [ ] Subjects have given verbal consent on the record.
- [ ] Watermark is verified visible on a test render.
- [ ] You will mention "AI-generated" out loud during the demo.
- [ ] You will offer to delete recordings after the talk.

## Reporting misuse

If you see this project being used to harm someone, open an issue in the
repository. The maintainers will revoke contributions, blocklist forks where
possible, and update documentation to reduce the chance of recurrence.
