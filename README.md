# SONIC SIGNATURES
## From Audio Signals to Song Recognition

---

## 1. Project Overview

In this project, we will explore how digital signal processing can be used to identify a song from a short audio clip, similar to how applications like Shazam work.

The goal of this project is to learn by doing! You'll explore audio signals, try different approaches, run experiments and figure out why things work (or don't).

Also refer to the main resources of week-2 given before for understanding SP (fft, spectrograms etc.).

**Using AI:** Feel free to use AI to understand concepts, learn unfamiliar code, and debug errors. However, using AI to generate the entire solution is discouraged. Make sure you understand and can explain the code you submit.

**DEADLINE: Wed 14th October, 2026**

---

## 2. Group Formation

The project will be carried out in three groups, each containing 3 to 4 students.

- **Group 1:** 3-4 students
- **Group 2:** 3-4 students
- **Group 3:** 3-4 students

Every group will have a major objective. Group members can work together and redistribute the work among themselves, though individual responsibilities are assigned for your assistance. Prerequisites and expected deliverables are mentioned.

---

## Group 1: AUDIO ANALYSIS

### Group 1 Technology Pack

**Basic Python**
- https://youtu.be/YYXdXT2l-Gg?si=D9KdClMAh2zKHnC6
- Corey Schafer playlist is beginner friendly and detailed. You can refer to the videos given below.
  - Corey Schafer Python Tutorial for beginners 1-9

**Google Colab**
- Google Colab Tutorial for Beginners | Get Started with Google Colab

**Numpy**
- https://www.w3schools.com/python/numpy/default.asp (Before Numpy random)

**Matplotlib**
- https://www.w3schools.com/python/matplotlib_intro.asp (Matplotlib intro to Matplotlib histogram)

**Scipy**
- https://www.w3schools.com/python/scipy/index.php

**Librosa**
- Audio Data Processing in Python

### Group 1 Tasks

| # | Task | Detailed Task | Deliverable |
|---|------|---------------|-------------|
| 1 | Audio loading and waveform analysis | Load the three songs, inspect their sample rates, durations and channels, convert them to a consistent format, and plot their waveforms. | A notebook and reusable audio-loading function. |
| 2 | Spectrogram and window-size study | Generate spectrograms for the three songs. Compare at least two window sizes and explain the trade-off between time and frequency resolution. | Spectrogram plots and a recommended window configuration. |
| 3 | Spectral peak detection | Detect local maxima in the spectrogram. Experiment with peak thresholds and minimum spacing to distinguish useful peaks from clutter. | A peak detector and plots showing the selected peaks. |

---

## Group 2: FINGERPRINTING AND RECOGNITION

*Starts after Group 1's handover.*

### Group 2 Technology Pack

**Python, lists and libraries**
- https://youtu.be/YYXdXT2l-Gg?si=D9KdClMAh2zKHnC6
- Corey Schafer Python Tutorial for beginners 1-9
- Python Tutorial: Slicing Lists and Strings
- Python Tutorial: Comprehensions

**Google Colab**
- Google Colab Tutorial for Beginners | Get Started with Google Colab

**Numpy**
- https://www.w3schools.com/python/numpy/default.asp (Before Numpy random)

**Matplotlib**
- https://www.w3schools.com/python/matplotlib_intro.asp (Matplotlib intro to Matplotlib histogram)

**Scipy**
- https://www.w3schools.com/python/scipy/index.php

**Librosa**
- Audio Data Processing in Python

### Group 2 Tasks

| # | Task | Detailed Task | Deliverable |
|---|------|---------------|-------------|
| 4 | Fingerprint representation | Use the selected peaks to construct simple peak pairs. Record their two frequencies and time separation, and visualise the resulting candidate fingerprints. | A list of peak-pair fingerprints for each song. No song matching is required yet. |
| 5 | Simple spectral matching | Build a baseline recogniser using average spectral features and a similarity measure to compare a query clip with the three songs. | A working baseline that returns the most similar song and its similarity score. |
| 6 | Peak-pair matching | Implement the Shazam-style approach: pair spectral peaks, index the resulting hashes, and use matching time offsets to identify the song. | A fingerprint-based recogniser that returns the predicted song and an offset-voting histogram. |

---

## Group 3: ROBUSTNESS AND COMPARISON

*Starts after Group 1's and Group 2's handover.*

### Group 3 Technology Pack

**Python, lists and libraries**
- https://youtu.be/YYXdXT2l-Gg?si=D9KdClMAh2zKHnC6
- Corey Schafer Python Tutorial for beginners 1-9
- Python Tutorial: Slicing Lists and Strings
- Python Tutorial: Comprehensions

**Google Colab**
- Google Colab Tutorial for Beginners | Get Started with Google Colab

**Numpy**
- https://www.w3schools.com/python/numpy/default.asp (Before Numpy random)

**Matplotlib**
- https://www.w3schools.com/python/matplotlib_intro.asp (Matplotlib intro to Matplotlib histogram)

**Scipy**
- https://www.w3schools.com/python/scipy/index.php

**Librosa**
- Audio Data Processing in Python

**Pandas (Optional)**
- https://www.w3schools.com/python/pandas/default.asp (Section 1. Basics & 2. Cleaning Data)

### Group 3 Tasks

| # | Task | Detailed Task | Deliverable |
|---|------|---------------|-------------|
| 8 | Query-length and starting-position study | Compare recognition performance for different query durations and starting positions. | Experiment plots or tables showing recognition performance. |
| 9 | Noise robustness | Test how adding different amounts of noise affects recognition. | Results showing recognition performance under different noise levels. |
| 10 | Transformation robustness and comparison | Test pitch shifting or time stretching, and compare the two recognition methods. | Comparison notebook showing where each method succeeds or fails. |
