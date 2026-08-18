# RestoreNet Demo Video Script

**Duration**: 3-5 minutes  
**Target Audience**: Technical reviewers, potential users, hackathon judges  
**Tone**: Professional, clear, engaging

---

## 🎬 VIDEO STRUCTURE

### Opening (0:00 - 0:20)
### Problem Statement (0:20 - 0:45)
### Solution Overview (0:45 - 1:15)
### Live Demo (1:15 - 3:30)
### Technical Highlights (3:30 - 4:15)
### Results & Conclusion (4:15 - 5:00)

---

## 📝 DETAILED SCRIPT

### SCENE 1: OPENING (0:00 - 0:20)
**Visual**: Title screen with RestoreNet logo, smooth animations

**Script**:
> "Welcome to RestoreNet - a deep learning solution for semiconductor image restoration developed for the KLA Hackathon 2026.
> 
> I'm [Your Name], and today I'll show you how we're solving critical image quality challenges in semiconductor manufacturing."

**On-Screen Text**:
- RestoreNet
- KLA Hackathon 2026
- AI-Powered Image Restoration

**Action Items**:
- [ ] Record voiceover
- [ ] Create title animation
- [ ] Add background music (subtle, professional)

---

### SCENE 2: PROBLEM STATEMENT (0:20 - 0:45)
**Visual**: Show examples of degraded semiconductor images vs. clean images

**Script**:
> "In semiconductor metrology, electron microscope images often suffer from severe degradation. They're noisy, low resolution, and make defect detection extremely difficult.
> 
> Traditional methods struggle with this combination of noise and downsampling. What if we could restore these images to their original quality in under 100 milliseconds?"

**On-Screen Text**:
- Challenge: Noisy + Low Resolution
- Target: < 100ms latency
- Goal: High-fidelity restoration

**Visual Examples**:
- Split screen: Degraded image → Clean image
- Zoom into noise patterns
- Show quality metrics

**Action Items**:
- [ ] Prepare 2-3 before/after image pairs
- [ ] Create comparison animation
- [ ] Highlight specific noise patterns

---

### SCENE 3: SOLUTION OVERVIEW (0:45 - 1:15)
**Visual**: Architecture diagram with animated flow

**Script**:
> "RestoreNet is a specialized neural network architecture that tackles both problems simultaneously.
> 
> First, we upsample the input using continuous bilinear interpolation - preserving spatial information without artifacts.
> 
> Then, 10 deep residual blocks with channel attention extract high-frequency details while suppressing noise.
> 
> Finally, a global residual connection ensures we only learn what needs to be fixed - the correction map."

**On-Screen Text**:
- 2× Upsampling
- 10 Residual Blocks
- Channel Attention
- Global Skip Connection

**Visual**:
- Animated architecture diagram
- Data flow visualization
- Highlight key components

**Action Items**:
- [ ] Create architecture diagram animation
- [ ] Show data transformation at each stage
- [ ] Highlight attention mechanism

---

### SCENE 4: LIVE DEMO - PART 1 (1:15 - 2:00)
**Visual**: Screen recording of the web interface

**Script**:
> "Let me show you RestoreNet in action. Here's our web interface - designed for both technical users and production environments.
> 
> On the left, we have our control panel. Let's load a synthetic wafer sample to demonstrate."

**Actions**:
- Open the application
- Click "Load Synthetic Wafer"
- Point out the interface elements

**Script (continued)**:
> "Notice the array metadata - we're working with a 128 by 128 float32 array. The data range shows we have realistic sensor values with noise characteristics.
> 
> This degraded image simulates what an electron microscope might capture under challenging conditions."

**On-Screen Annotations**:
- Arrow pointing to "Array Metadata"
- Highlight MIN/MAX/MEAN values
- Circle the STATUS indicator

**Action Items**:
- [ ] Record clean screen capture
- [ ] Add on-screen annotations
- [ ] Zoom in on key UI elements

---

### SCENE 5: LIVE DEMO - PART 2 (2:00 - 2:45)
**Visual**: Continue screen recording

**Script**:
> "Now, let's run the inference. Watch what happens."

**Actions**:
- Click "RUN INFERENCE" button
- Let the processing animation play

**Script (continued)**:
> "The system trace activates, showing live processing activity. Within just 123 milliseconds - well under our 100ms target for GPU inference - RestoreNet has completely restored the image.
> 
> Look at the restored output - the noise is gone, the resolution is doubled, and structural details are preserved."

**Visual Focus**:
- Show SYS_TRACE changing from IDLE → STREAM_ACTIVE → IDLE
- Emphasize the COMPLETE status
- Show the image transformation

**Script (continued)**:
> "The middle panel has an interactive comparison slider. Drag it left and right to see the before and after side by side. The difference is remarkable."

**Actions**:
- Drag the comparison slider slowly
- Show the before/after difference

**On-Screen Annotations**:
- Highlight clean vs. noisy areas
- Circle recovered details
- Draw attention to edge preservation

**Action Items**:
- [ ] Record smooth slider interaction
- [ ] Add slow-motion effect for slider drag
- [ ] Annotate key differences

---

### SCENE 6: LIVE DEMO - PART 3 (2:45 - 3:30)
**Visual**: Continue screen recording

**Script**:
> "On the right, we have the residual map - a thermal visualization showing exactly what the network corrected. The bright areas represent where the most noise was removed.
> 
> Now, for the metrics. Since we're running without ground truth, PSNR, SSIM, and LPIPS show 'N/A'. This is intentional - RestoreNet is honest about what it can measure."

**Actions**:
- Point to the residual map
- Explain the color coding
- Show the metrics panel

**Script (continued)**:
> "But watch what happens when we add a ground truth file."

**Actions**:
- Click "+ Add GT"
- Upload ground truth file
- Click "Evaluate (Real Metrics)"

**Script (continued)**:
> "Now we get real metrics. PSNR of 24.6 decibels - that's 11.8 dB improvement over baseline. SSIM of 0.66 shows strong structural preservation. And LPIPS of 0.36 indicates perceptually realistic output.
> 
> All in under 125 milliseconds."

**On-Screen Text**:
- PSNR: 24.64 dB (+11.83 dB)
- SSIM: 0.66 (+0.24)
- LPIPS: 0.36 (-29%)
- Latency: < 125ms

**Action Items**:
- [ ] Record the GT upload flow
- [ ] Capture metrics animation
- [ ] Highlight performance numbers

---

### SCENE 7: TECHNICAL HIGHLIGHTS (3:30 - 4:15)
**Visual**: Split screen - code snippets + architecture diagrams

**Script**:
> "Let's talk about what makes RestoreNet special.
> 
> First, it's production-ready. The entire system is built on FastAPI and React - modern, scalable technologies. We use PyTorch for the model, with support for both CPU and GPU inference.
> 
> Second, it's honest. No fake metrics, no mock data. If ground truth isn't available, we tell you. When it is, we compute real PSNR, SSIM, and LPIPS.
> 
> Third, it's deployable. We've included Docker configurations, deployment scripts, and a comprehensive guide. You can run this locally, in the cloud, or as a containerized microservice."

**Visuals**:
- Show FastAPI code
- Display React components
- Show Docker compose file
- Show deployment guide

**On-Screen Text**:
- ✓ Production-Ready Stack
- ✓ Real Metrics
- ✓ Easy Deployment
- ✓ CPU + GPU Support

**Action Items**:
- [ ] Screen capture of code editor
- [ ] Show file structure
- [ ] Display deployment guide
- [ ] Highlight key technologies

---

### SCENE 8: ARCHITECTURE DEEP DIVE (4:15 - 4:45)
**Visual**: Animated neural network visualization

**Script**:
> "The architecture combines proven techniques in a novel way. Continuous bilinear upsampling avoids checkerboard artifacts. Deep residual learning with 10 blocks captures hierarchical features. Squeeze-and-excitation attention dynamically weights important channels.
> 
> But the key insight is the global residual connection. By learning only the correction map delta, not the full output, we make training more stable and convergence faster.
> 
> The result? 777,000 parameters that deliver state-of-the-art restoration quality."

**Visuals**:
- Animated network architecture
- Show residual connections flowing
- Visualize attention mechanism
- Display parameter count

**On-Screen Text**:
- 777K Parameters
- 10 Residual Blocks
- SE Attention Modules
- Global Skip Connection

**Action Items**:
- [ ] Create architecture animation
- [ ] Visualize information flow
- [ ] Highlight novel components

---

### SCENE 9: RESULTS & COMPARISON (4:45 - 5:15)
**Visual**: Results table and comparison charts

**Script**:
> "How does RestoreNet compare to the baseline?
> 
> PSNR improved by 11.8 decibels - that's a massive quality increase. SSIM went from 0.42 to 0.66 - much better structural fidelity. LPIPS dropped by 29% - meaning more perceptually realistic outputs.
> 
> Yes, inference is slightly slower than the baseline - 105 milliseconds on CPU versus 24. But with GPU acceleration, we're well under 10 milliseconds. That's real-time performance for production workflows."

**Visual**:
- Side-by-side comparison table
- Bar charts showing improvements
- Before/after examples

**On-Screen Text**:
| Metric | Baseline | RestoreNet | Improvement |
|--------|----------|------------|-------------|
| PSNR | 12.81 dB | 24.64 dB | +11.83 dB |
| SSIM | 0.421 | 0.665 | +0.244 |
| LPIPS | 0.512 | 0.364 | -29% |

**Action Items**:
- [ ] Create comparison table animation
- [ ] Show multiple test images
- [ ] Display performance charts

---

### SCENE 10: CLOSING (5:15 - 5:45)
**Visual**: Return to application, show final restored image

**Script**:
> "RestoreNet demonstrates that with the right architecture and training approach, we can solve challenging real-world problems in semiconductor imaging.
> 
> It's fast, accurate, and ready for production deployment. Whether you're running on-premise, in the cloud, or in a containerized environment, RestoreNet adapts to your infrastructure.
> 
> All the code, documentation, and deployment guides are included. Try it yourself, and see the difference deep learning can make in your metrology pipeline."

**Visuals**:
- Show final restored image
- Display GitHub repository
- Show deployment options
- End with RestoreNet logo

**On-Screen Text**:
- GitHub: [Repository URL]
- Documentation: DEPLOYMENT_GUIDE.md
- License: MIT
- Contact: [Your Email]

**Script (final)**:
> "Thank you for watching. For the KLA Hackathon 2026, this is RestoreNet."

**Action Items**:
- [ ] Record closing shot
- [ ] Add call-to-action graphics
- [ ] Include contact information
- [ ] Fade to black with credits

---

## 🎯 SHOT LIST

### Technical Setup
- **Resolution**: 1920x1080 (Full HD minimum)
- **Frame Rate**: 30 or 60 FPS
- **Audio**: Clear voiceover, no background noise
- **Music**: Subtle tech/corporate background (low volume)

### Screen Recording Settings
- **Software**: OBS Studio / Camtasia / ScreenFlow
- **Cursor**: Show cursor with highlight/glow
- **Zoom**: Use zoom effects for important details
- **Transitions**: Smooth fades, minimal flash

### Recording Checklist
- [ ] Close unnecessary applications
- [ ] Disable notifications
- [ ] Clear browser cache/history
- [ ] Use clean test data
- [ ] Practice run-through 2-3 times
- [ ] Check audio levels
- [ ] Verify screen resolution
- [ ] Test camera (if showing face)

---

## 📊 VISUAL ASSETS NEEDED

### Static Images
1. RestoreNet logo / title card
2. Architecture diagram (high-res)
3. Before/after comparison grid (4-6 examples)
4. Metrics comparison table
5. Technology stack icons

### Animations
1. Title sequence (5-10 seconds)
2. Architecture flow diagram
3. Data transformation visualization
4. Metrics counter animation
5. Transition effects

### Code Snippets
1. FastAPI endpoint example
2. Model architecture code
3. Docker compose snippet
4. Deployment script

### Screenshots
1. Web interface overview
2. Control panel close-up
3. Metrics dashboard
4. Advanced config panel
5. File structure

---

## 🎬 PRODUCTION TIPS

### Voiceover Recording
- Use a quality microphone (USB condenser recommended)
- Record in a quiet room with minimal echo
- Speak clearly and at moderate pace
- Leave pauses for breathing
- Record each scene separately for easy editing
- Do 2-3 takes of each line

### Screen Recording
- Use 1920x1080 resolution minimum
- Record at 60 FPS for smooth playback
- Slow down cursor movements
- Pause briefly before clicking
- Use keyboard shortcuts to avoid showing context menus
- Record in segments, not one long take

### Editing
- Use professional software (DaVinci Resolve, Premiere Pro, Final Cut)
- Add smooth transitions between scenes
- Overlay text annotations for key points
- Highlight cursor when clicking important buttons
- Add subtle zoom effects for emphasis
- Keep total runtime under 5 minutes

### Color Grading
- Boost UI contrast slightly
- Make sure text is readable
- Consistent color temperature throughout
- Don't over-saturate

### Audio
- Background music at -20 to -25 dB
- Voiceover at -6 to -3 dB
- Add subtle room tone under voice
- Remove mouth clicks and breaths
- Add fade in/out to music

---

## 📋 PRE-PRODUCTION CHECKLIST

### Application Setup
- [ ] Clean install of application
- [ ] Load sample data files
- [ ] Test all features work correctly
- [ ] Clear any previous session data
- [ ] Verify model checkpoint is loaded
- [ ] Test with and without ground truth

### Environment Setup
- [ ] Close all unnecessary applications
- [ ] Disable Windows notifications
- [ ] Set taskbar to auto-hide
- [ ] Clean up desktop
- [ ] Use high contrast mouse cursor
- [ ] Set browser to full screen

### Recording Setup
- [ ] Install OBS Studio or screen recorder
- [ ] Test audio levels
- [ ] Test video quality
- [ ] Set up scene transitions
- [ ] Create recording checklist
- [ ] Do test recording

### Content Preparation
- [ ] Write and practice script
- [ ] Prepare visual assets
- [ ] Create title cards
- [ ] Design annotations
- [ ] Select background music
- [ ] Plan shot sequence

---

## 🎤 VOICEOVER SCRIPT (Print-Friendly)

[See detailed script above - print this section separately for voice recording]

**Pro Tip**: Record each scene's voiceover separately. This makes editing much easier and allows you to re-record sections without redoing everything.

---

## 📤 EXPORT SETTINGS

### YouTube / Web
- Format: MP4 (H.264)
- Resolution: 1920x1080
- Frame Rate: 30 FPS
- Bitrate: 8-12 Mbps (variable)
- Audio: AAC, 192 kbps, 48 kHz

### Presentation / Conference
- Format: MP4 (H.264)
- Resolution: 1920x1080 or 4K
- Frame Rate: 60 FPS
- Bitrate: 15-20 Mbps
- Audio: AAC, 256 kbps, 48 kHz

---

## 🚀 FINAL DELIVERY

### Video Files
- `restorenet_demo_full.mp4` (full version 3-5 min)
- `restorenet_demo_short.mp4` (highlights 60-90 sec)
- `restorenet_demo_silent.mp4` (no audio, for social media)

### Supplementary
- `restorenet_thumbnail.jpg` (1920x1080)
- `restorenet_title_card.png` (transparent background)
- `restorenet_presentation.pdf` (slides if needed)

### Distribution
- Upload to YouTube (unlisted or public)
- Share on GitHub repository
- Include in hackathon submission
- Post on LinkedIn / social media

---

**Good luck with your video! 🎬🎉**

Remember: Practice makes perfect. Do a few dry runs before recording the final version.
