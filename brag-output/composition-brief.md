# Hyperframes Composition Brief: RestoreNet

## Objective
Create a short launch-style brag video for RestoreNet. Show the product in action: upload a .npy file, run inference, compare results with the slider, and view metrics.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 18 seconds

## Source Material
- Project root: `kla-image-restoration/frontend/`
- Primary files read: `index.html`, `src/index.css`, `src/App.jsx`, `src/components/UploadZone.jsx`, `src/components/MetricsBar.jsx`, `src/components/ImageDisplay.jsx`, `src/components/PipelineTrace.jsx`
- Product name: RestoreNet
- Tagline / strongest claim: Degradation-Aware Fidelity-First Image Restoration
- Key UI or visual moment to recreate: The full flow — upload → inference → comparison slider → metrics pipeline
- Copy that must appear verbatim:
  - "RestoreNet: Degradation-Aware Fidelity-First Image Restoration"
  - "24.64 dB"
  - "0.6646 SSIM"
  - "0.3636 LPIPS"
  - "105.7 ms"

## Creative Direction
- Tone preset: `default`
- Creative direction: warm product walkthrough — conversational, direct, no corporate language. Voiceover guides the viewer through the app in action.
- Angle: From blurry to brilliant — the restoration journey from noisy input to clean result with real metrics.
- Hook: First 2-3 seconds earn the next 15 with a warm question about sensor clarity.
- Outro / punchline: RestoreNet logo + tagline.

## Visual Identity
- Background: #050810
- Text: #e0e2ee
- Accent: #00E5FF
- Display font: Inter
- Body font: JetBrains Mono
- Strongest visual element: the comparison slider with input/restored toggling, pipeline chart, metric cards

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

Scene summary:
1. Hook — Noisy canvas enters — 3s — "Ever look at a sensor scan and wish it were clearer?"
2. Upload — .npy drops into upload zone — 3s — "Drop a .npy wafer scan, hit run, and watch it recover."
3. Inference — PROCESSING badge pulses, pipeline chart activates — 3s — "Neural inference, 50% of the pipeline — the heavy lifter."
4. Restored Reveal — Restored canvas appears, comparison slider slides up — 3s — "Drag to compare noisy vs restored side by side."
5. Metrics Land — PSNR/SSIM/LPIPS values animate in, pipeline completes — 3s — "24.64 dB PSNR. 0.6646 SSIM. 0.3636 LPIPS — the fidelity triplets."
6. Pipeline Recap & Outro — Full pipeline chart completes, RestoreNet logo fades in — 3s — "RestoreNet: degradation-aware, fidelity-first. Available now."

## Audio
- Audio role: Voiceover primary, music bed supportive, SFX complementary
- Audio arc: Music fades in under hook, ducks under full voiceover, returns for outro; SFX mark key moments (upload, inference, reveal, metrics)
- Music: `happy-beats-business-moves-vol-1-by-ende-dot-app.mp3`
- Music treatment: fade-in under opening, hold under narration, gentle fade-out before final logo; voiceover ducks to 0.12-0.15 volume
- Music cue guidance: track tempo ~120 BPM; strong cues at ~3s (hook settle), ~9s (inference peak), ~15s (metrics resolution); sequential events snap to every-other-beat for readability
- Audio-reactive treatment: subtle; music RMS used to make hero canvas glow slightly on bass, product card presence breathe on low-mid — no waveform/equalizer visuals
- Audio-coupled moments:
  - Scene 1 (0s): interface click when canvas appears
  - Scene 3 (2.5s): impactBell_heavy_000 at inference peak
  - Scene 4 (0.3s): drop_001 when slider appears
  - Scene 5: key ticks as each metric lands
  - Scene 6 (2.8s): impactSoft_medium_000 on logo landing
- SFX selection guidance: use skill's sfx-analysis.md for file selection; prefer low HF risk for polished moment; interface clicks for upload, drop for reveal, impact bell for peak, medium soft for logo
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: copy the chosen music into `brag-output/composition/assets/music/`

## Hyperframes Instructions
- Load the composition-building Hyperframes domain skills — `hyperframes-core` (composition contract + `data-*` timing), `hyperframes-animation` (motion), `hyperframes-creative` (design spec, beats, audio-reactive), `hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli` (lint/check/render).
- Show at least one real UI, copy, or visual element from the source project.
- Keep all text readable in the final render.
- Keep the video within 15-25 seconds.
- Include the planned music/SFX layer unless audio was explicitly disabled or documented as intentionally silent.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet. Choose SFX after the visual animation exists.
- Treat music cue metadata as optional timing hints. Hyperframes decides exact animation timing and should ignore cues that hurt readability, scene pacing, or the product story.
- Major reveals may move toward nearby strong cues within about 0.15s. Smaller entrances may align to nearby beat points within about 0.10s. Use only 1-3 strong cue locks in a 15-25s video unless the edit clearly benefits from more.
- Use SFX to support motion and interaction: card sounds for card-like reveals, short announcement cues for major payoffs, key/click sounds for text or user actions, and restraint when the edit is already busy.
- Honor planned music treatment such as fade-outs, ducking, beat-aligned reveals, or letting a final SFX ring over the music, using the best Hyperframes-supported implementation.
- When music is present and the treatment is not `none`, consider Hyperframes audio-reactive workflow: extract audio data and use RMS/frequency bands for subtle, brand-specific motion. Good targets are glow, depth, background warmth, card presence, title emphasis, or other existing visual elements. Avoid waveform/equalizer visuals, musical-note graphics, generic particle systems, strobing, or heavy pulsing.
- Use local assets for audio and any required runtime/media dependencies when possible.
- Run `hyperframes check` before render — it is brag's single gate.

## Voiceover script
Write the narration lines into `brag-plan.md` under a `## Voiceover script` section, then generate the audio through Kokoro:

```bash
npx hyperframes tts "<narration text or path to script>" \
  --voice af_heart \
  --output <output-dir>/composition/assets/voiceover.wav
```

Wire it into the composition on its own track. Music ducks to 0.12–0.15 for the duration of the voiceover, then returns to its normal level.

### Voiceover script

"Ever look at a sensor scan and wish it were clearer? Just drop a .npy file into the upload zone and hit run. RestoreNet's neural inference runs in under 100 milliseconds on CPU — or under 10 on GPU. The restored result pops in beside the input, and you can drag the slider to compare noisy versus restored side by side. Key metrics land: 24.64 dB PSNR, 0.6646 SSIM, 0.3636 LPIPS. The full pipeline: I/O, preprocess, memory transfer, neural inference, postprocess. RestoreNet: degradation-aware, fidelity-first. Available now."