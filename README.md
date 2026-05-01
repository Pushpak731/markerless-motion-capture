# Motion Capture System

This repository is organized around a clean top-level index and a single active implementation folder.

Start here:

- Active app and launchers: [Motion-capture/](Motion-capture/)
- Root architecture assets: [docs/](docs/)
- Current implementation summary: [Motion-capture/README.md](Motion-capture/README.md)
- Change log: [Motion-capture/CHANGES.md](Motion-capture/CHANGES.md)

## What this repo does

The project captures pose data from a webcam, phone stream, or video file, then turns it into live or offline motion-capture metrics.

Key capabilities:

- single-camera capture from a local webcam, IP phone stream, or recording
- multi-camera server/master capture for stereo reconstruction
- offline video verification with annotated output video
- stabilized bone-length tracking with world-space preference
- live dashboard metrics plus exported session data

## Clean repo map

The repository is intentionally split into a small number of visible entry points:

- [Motion-capture/](Motion-capture/) contains the running application
- [docs/](docs/) contains root-level architecture visuals and screenshots
- [Motion-capture/docs/](Motion-capture/docs/) contains setup, implementation, and reference docs

## Recommended reading order

1. [Motion-capture/README.md](Motion-capture/README.md) for the active app and launch commands.
2. [Motion-capture/docs/SETUP.md](Motion-capture/docs/SETUP.md) for network and camera setup.
3. [Motion-capture/IMPLEMENTATION.md](Motion-capture/IMPLEMENTATION.md) for the validation workflow.
4. [docs/architecture.mmd](docs/architecture.mmd) for the high-level system diagram.

## Current focus

The current work is about reducing variance, keeping metric outputs stable, and making the verification path easy to run from a video file before using live cameras.