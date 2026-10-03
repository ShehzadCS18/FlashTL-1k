# FlashTL-1K: Beyond Red and Green

Official repository for the ACCV 2026 paper:

> **Beyond Red and Green: Flashing Traffic Light Recognition in Autonomous Driving**

This repository provides the **FlashTL-1K dataset**, the official implementation
of **FlashNet**, and the code used for training, evaluation, and dataset analysis.

## Overview

Flashing traffic lights convey safety-critical driving instructions through
temporal blinking patterns rather than static appearance alone. However, most
existing traffic-light datasets and recognition methods primarily focus on
conventional solid red, yellow, and green states.

To address this limitation, this work introduces:

- **FlashTL-1K** — a dedicated real-world video benchmark for flashing
  traffic-light recognition.
- **FlashNet** — a lightweight spatial-temporal recognition framework combining
  appearance information with explicit temporal modeling of flashing dynamics.
- **DHOR** — a Damped Harmonic Oscillator Resonance branch for modeling
  periodic blinking behavior.
- **DSML** — a Duty-cycle-aware Safety-asymmetric Margin Loss designed to
  emphasize safety-critical flashing-red errors while adapting the decision
  margin according to flicker quality.

## Repository Status

🚧 Code and dataset release is currently being prepared.

More documentation, pretrained models, dataset access, and reproducibility
instructions will be added shortly.
