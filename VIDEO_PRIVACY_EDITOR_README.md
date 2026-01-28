# 🎬 Video Privacy Editor - Professional

A professional-grade desktop video privacy editing application built with **Python**, **PyQt6**, and **OpenCV**. Easily blur sensitive content in videos using automatic object tracking or manual mouse-following blur recording.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)

---

## ✨ Features

### 🎯 Two Blur Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Auto Track** | Select an object once, AI tracks it automatically | Predictable moving objects (faces, cars) |
| **Manual Record** | Follow object with mouse as video plays slowly | Erratic movement, multiple objects, precision work |

### 🔄 Auto Track Mode
- Uses **CSRT (Discriminative Correlation Filter)** for high-accuracy tracking
- Automatic fallback to KCF or MIL trackers if CSRT unavailable
- One-click object selection with bounding box
- Real-time tracking preview

### 🖱️ Manual Record Mode
- **Slow-motion playback** while recording (adjustable 1-15 FPS)
- Mouse position recorded per frame as video plays
- **Re-record any section** - scrub back and overwrite
- Adjustable blur region size (width/height)
- Record multiple separate segments

### 🌫️ Blur System
- **Gaussian blur** with adjustable intensity (5-151 kernel size)
- Edge-case handling for objects at frame boundaries
- Real-time blur preview during playback

### 💾 Export
- Frame-by-frame processing with blur applied
- **MP4 output** using mp4v codec
- Progress bar with percentage display
- Thread-safe video processing

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install required packages
pip install PyQt6 opencv-python numpy

# Optional: For better tracking (CSRT)
pip install opencv-contrib-python
```

### Run the Application

```bash
python video_privacy_editor.py
```

---

## 📖 User Guide

### Getting Started

1. **Launch the application**
2. Click **📂 Load Video** to open a video file
3. Choose your blur mode (Auto or Manual)
4. Apply blur using the selected method
5. Click **🚀 Export Video** to save the result

---

### 🎯 Auto Track Mode (Default)

Best for objects with predictable movement paths.

#### Steps:
1. Select **🎯 Auto Track** mode (default)
2. Click **📍 Select Region of Interest**
3. **Draw a box** around the object to track (left-click and drag)
4. Click **🎯 Start Tracking**
5. Press **▶️ Play** to preview tracking with blur
6. Adjust **Blur Intensity** slider as needed
7. Click **🚀 Export Video** when satisfied

#### Tips:
- Select a distinctive part of the object for better tracking
- Ensure good contrast between object and background
- Re-select if tracking drifts

---

### 🖱️ Manual Record Mode

Best for erratic movement, multiple objects, or precision work.

#### Steps:
1. Select **🖱️ Manual** mode
2. Adjust **Blur Region Size** (W/H) to match object size
3. Set **Recording Speed** (lower = easier to track, default 5 FPS)
4. Use timeline slider to navigate to starting point
5. **Hold right-click** on the video canvas
6. **Move mouse** to follow the object as video plays slowly
7. **Release right-click** to stop recording
8. Click **🚀 Export Video**

#### Re-recording Sections:
- Release right-click to stop
- Scrub timeline back to the frame you want to redo
- Adjust blur size if needed
- Hold right-click again - new positions overwrite old ones

#### Recording Multiple Segments:
- Record first segment, release
- Scrub to another section of the video
- Hold right-click to record that segment
- Repeat for as many segments as needed
- All recorded segments will have blur applied on export

---

## 🎛️ Controls Reference

### Main Toolbar

| Button | Function |
|--------|----------|
| 📂 Load Video | Open a video file |
| ▶️ Play | Start/pause playback |
| ⏹️ Stop | Stop playback |

### Mode Selection

| Option | Description |
|--------|-------------|
| 🎯 Auto Track | Automatic object tracking mode |
| 🖱️ Manual | Mouse-following blur recording mode |

### Auto Track Settings

| Control | Function |
|---------|----------|
| 📍 Select Region of Interest | Enable ROI selection mode |
| 🎯 Start Tracking | Initialize tracker with selected region |
| 🛑 Stop Tracking | Disable tracking |

### Manual Record Settings

| Control | Function |
|---------|----------|
| W / H | Blur region width and height |
| Recording Speed (FPS) | How fast video plays during recording (1-15) |
| 📹 Recorded: X frames | Shows number of recorded frames |
| 🗑️ Clear Recording | Erase all recorded positions |

### Blur Settings

| Control | Function |
|---------|----------|
| Blur Intensity | Gaussian blur kernel size (5-151) |

### Export

| Control | Function |
|---------|----------|
| Progress Bar | Shows export progress |
| 🚀 Export Video | Save processed video |

---

## ⌨️ Mouse Controls

| Action | Mode | Function |
|--------|------|----------|
| Left-click drag | Auto (ROI selection) | Draw bounding box |
| Right-click hold | Manual | Record blur positions |
| Right-click release | Manual | Stop recording |

---

## 🔧 Technical Details

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MainWindow (QMainWindow)              │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │   VideoCanvas       │  │    Control Panel        │  │
│  │   (QLabel)          │  │    - Mode Selection     │  │
│  │   - Frame display   │  │    - Blur Settings      │  │
│  │   - ROI selection   │  │    - Export Controls    │  │
│  │   - Manual blur     │  │                         │  │
│  └─────────────────────┘  └─────────────────────────┘  │
│                              │                          │
│  ┌───────────────────────────▼─────────────────────┐   │
│  │            VideoProcessor (QThread)              │   │
│  │  - Video I/O        - CSRT Tracking             │   │
│  │  - Frame processing - Blur application          │   │
│  │  - Export pipeline  - Thread-safe operations    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Supported Formats

**Input:** MP4, AVI, MOV, MKV, WebM

**Output:** MP4 (mp4v codec)

### Threading Model

- **Main Thread:** GUI, user interaction, timer-based recording
- **Worker Thread:** Video playback, export processing
- **Thread Safety:** Mutex lock protecting VideoCapture access

---

## 🐛 Troubleshooting

### "No compatible tracker found"
Install opencv-contrib-python:
```bash
pip install opencv-contrib-python
```

### Application crashes during recording
- Ensure you're not running other video processing in parallel
- Try reducing recording speed
- Check available system memory

### Tracking drifts or loses object
- Re-select the object with a tighter bounding box
- Choose a more distinctive part of the object
- Switch to Manual mode for problematic scenes

### Export progress stuck
- Wait a moment - large videos take time
- Check disk space for output file
- Ensure video file isn't corrupted

---

## 📝 License

This project is provided as-is for educational and personal use.

---

## 🙏 Credits

Built with:
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing

---

**Made with ❤️ for video privacy protection**
