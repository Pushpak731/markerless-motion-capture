# ai_greenhouse

> **Smart Plant Care Assistant**  
> AI-powered desktop app for real-time greenhouse monitoring, analytics, and automation.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Overview

**ai_greenhouse** is an intelligent desktop application that helps you monitor and automate plant care in your greenhouse. It integrates with Arduino sensors to track environmental conditions and uses AI to provide actionable insights, notifications, and interactive chatbot support.

---

## Features

- 🌱 Real-time monitoring: Temperature, humidity, sunlight, soil moisture
- 🤖 AI Chatbot: Plant care Q&A, sentiment analysis, FAQ, voice input, TTS
- 📊 Analytics: Sensor history graphs, wordclouds, interaction analytics
- 🔔 Notifications: System tray alerts for plant care events
- 💾 Data export: Save chat history (txt/csv/json)
- 🖼️ File upload: Attach images to chat
- 🛠️ Arduino integration: Automated watering based on soil moisture

---

## Architecture

- **Frontend**: PyQt5 desktop UI
- **Backend**: Python, Arduino serial communication
- **AI/ML**: TextBlob (sentiment), speech_recognition, pyttsx3, wordcloud, matplotlib

---

## Getting Started

### Prerequisites

- Python 3.8+
- Arduino (with compatible sensors)
- pip

### Installation

```bash
git clone https://github.com/Mrudula-itsjuzme/ai_greenhouse.git
cd ai_greenhouse
pip install -r requirements.txt
```

### Arduino Setup

- Upload the provided Arduino sketch to your board.
- Connect sensors (temperature, humidity, light, soil moisture).
- Update `ARDUINO_PORT` in `bot1.py` and `python.py` as needed.

---

## Usage

```bash
python bot1.py
```

- Use the chatbot to ask plant care questions.
- View real-time sensor data and analytics.
- Enable notifications and voice input as needed.

---

## Configuration

- Edit `ARDUINO_PORT` and sensor thresholds in `bot1.py`/`python.py`.
- Customize themes and FAQ in the UI.

---

## Contributing

We welcome contributions! Please open issues or pull requests for bug fixes, features, or documentation improvements.

---

## Acknowledgements

- PyQt5, matplotlib, wordcloud, textblob, pyttsx3, speech_recognition
- Arduino community for sensor integration guides

---
