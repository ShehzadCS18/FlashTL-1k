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

## 📥 Dataset Download

The **FlashTL-1K** dataset is publicly available for research purposes.

<p align="center">
  <a href="https://drive.google.com/drive/folders/1vtgWp19TW2UQS5GZA9BSh9qOlI5GnK3g?usp=sharing">
    <b>📂 Download FlashTL-1K from Google Drive</b>
  </a>
</p>

FlashTL-1K contains **1,300 real-world video clips** of flashing traffic lights,
covering **flashing red** and **flashing yellow** signals across diverse driving
conditions, including daytime, nighttime, inside-vehicle, outside-vehicle, and
rainy scenarios.

### Dataset Statistics

| Condition | Flashing Red | Flashing Yellow | Total |
|:---|---:|---:|---:|
| Inside-Day | 55 | 190 | 245 |
| Inside-Night | 82 | 352 | 434 |
| Outside-Day | 23 | 86 | 109 |
| Outside-Night | 64 | 220 | 284 |
| Rain | 81 | 147 | 228 |
| **Total** | **305** | **995** | **1,300** |

> **Dataset:** [Download FlashTL-1K][(https://drive.google.com/drive/folders/1vtgWp19TW2UQS5GZA9BSh9qOlI5GnK3g?usp=sharing)]
