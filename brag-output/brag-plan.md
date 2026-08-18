# Brag Plan: RestoreNet

## What is this app?
RestoreNet: a deep neural network for joint image restoration and super-resolution in semiconductor metrology, KLA Hackathon 2026. Takes degraded .npy arrays, runs inference, and reconstructs high-res images with sub-100ms latency.

## The angle
A warm, visual walkthrough of "before vs after" — starting with a noisy input, triggering inference, and revealing the restored result with real metrics. The joke is that we're over-explaining a technical restoration pipeline, but the UI makes it feel sleek and immediate.

## Hook (first 2-3 seconds)
[Noisy 128x128 grayscale canvas] "Ever look at a sensor scan and wish it were clearer?" — the noisy input appears on screen, grain flickering.

## Key moments (the middle)
1. **[Upload & Load]** — User drops a .npy file into the upload zone. Filename materializes. "Drop a .npy wafer scan, hit run, and watch it recover."
2. **[Inference Begins]** — PROCESSING badge pulses. The pipeline chart activates: I/O → Preprocess → Memory Transfer → Neural Inference → Postprocess. "Neural inference, 50% of the pipeline — the heavy lifter."
3. **[Restored Reveal]** — The restored canvas fades in beside the input. Drag the comparison slider pops up. "Drag to compare noisy vs restored side by side."
4. **[Metrics Land]** — PSNR, SSIM, LPIPS values animate into view. "24.64 dB PSNR. 0.6646 SSIM. 0.3636 LPIPS — the fidelity triplets."
5. **[Pipeline Recap]** — The full pipeline chart completes its sweep. "105.7ms on CPU. Under 10ms on GPU. Real-time restoration."

## Outro / punchline
[RestoreNet logo + cyan accent] "RestoreNet: degradation-aware, fidelity-first. Available now."
"RestoreNet: degradation-aware, fidelity-first. Available now."

## User flow worth showing
Entry → upload .npy → run inference → compare input/restored with slider → read metrics (PSNR/SSIM/LPIPS) → view pipeline breakdown. The centerpiece is the upload → inference → comparison sequence — the product in action.

## Tone
- Preset: `default`
- Creative direction: warm product walkthrough — conversational, direct, no corporate language. Voice sounds like a friendly guide showing the app in action, not a sales pitch.
- Interpretation: Pacing stays energetic but legible. Voiceover complements each visual reveal without rushing. Warm tone, first-person plural where natural, direct address.

## Format: landscape — 1920x1080

## Duration: 18 seconds

## Visual identity (from the project)
- Background: #050810
- Accent: #00E5FF (accent cyan)
- Text: #e0e2ee (on background)
- Display font: Inter
- Body font: JetBrains Mono
- Strongest visual element: the comparison slider with input/restored toggling

## Share copy (draft)
RestoreNet: restoring clarity to semiconductor scans, one .npy at a time.

## Audio direction
- Role: Voiceover primary, music bed supportive, SFX complementary
- Music: "happy-beats-business-moves-vol-1-by-ende-dot-app.mp3" at low volume, ducks under voice
- Music treatment: fade-in under opening, hold under narration, gentle fade-out before outro
- SFX: subtle interface click on upload, gentle drop on reveal, light key ticks if text types out
- Audio-coupled moments: upload trigger, inference start, slider reveal, metric animations
- Restraint rule: music must not compete with voice — duck to 0.12–0.15 during narration

## Storyboard

### Scene 1 — Hook — 3s
[Noisy canvas enters from left, scanline animates] "Ever look at a sensor scan and wish it were clearer?"
- Sequential/interaction: canvas fades in, scanline sweeps
- Audio intent: subtle ambient bed begins
- Audio-coupled idea: soft interface click when canvas appears
- Music: warm bed fades in
- Transition mood: clean → Scene 2

### Scene 2 — Upload — 3s
[.npy file drops into upload zone, filename appears] "Drop a .npy wafer scan, hit run, and watch it recover."
- Sequential/interaction: file appears one by one, upload border pulses
- Audio intent: clear UI click on drop
- Audio-coupled idea: interface/drop_001 at 0.1s
- Music: bed continues
- Transition mood: clean → Scene 3

### Scene 3 — Inference — 3s
[PROCESSING badge pulses, pipeline chart activates] "Neural inference, 50% of the pipeline — the heavy lifter."
- Sequential/interaction: pipeline bars grow one by one (I/O 10%, Preprocess 15%, Memory Transfer 10%, Infer 50%, Postprocess 15%)
- Audio intent: energetic but unobtrusive
- Audio-coupled idea: impactBell_heavy_000 at 2.5s (inference peak)
- Music: builds slightly
- Transition mood: dramatic wipe → Scene 4

### Scene 4 — Restored Reveal — 3s
[Restored canvas appears beside input, comparison slider slides up] "Drag to compare noisy vs restored side by side."
- Sequential/interaction: slider handle appears, divider animates in
- Audio intent: satisfying drop on reveal
- Audio-coupled idea: drop_001 at 0.3s when slider appears
- Music: holds steady
- Transition mood: soft crossfade → Scene 5

### Scene 5 — Metrics Land — 3s
[PSNR/SSIM/LPIPS values animate in, pipeline completes] "24.64 dB PSNR. 0.6646 SSIM. 0.3636 LPIPS — the fidelity triplets."
- Sequential/interaction: metric cards scale in one by one
- Audio intent: clear readout emphasis
- Audio-coupled idea: subtle key ticks as each metric lands
- Music: holds, slight swell on final value
- Transition mood: long hold → Scene 6

### Scene 6 — Pipeline Recap & Outro — 3s
[Full pipeline chart complete, RestoreNet logo fades in] "RestoreNet: degradation-aware, fidelity-first. Available now."
- Sequential/interaction: none — all elements already on screen
- Audio intent: resolve to logo landing
- Audio-coupled idea: impactSoft_medium_000 at 2.8s (logo swell)
- Music: gentle fade-out under final line
- Transition mood: out → black

**Music mood for this video:** upbeat but restrained — supportive, not dominant
**Audio summary:** Voiceover drives the edit; music bed fades under and ducks during narration; subtle SFX mark key interactions (upload, inference, reveal, metrics)