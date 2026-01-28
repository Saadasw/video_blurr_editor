# 🎬 Video Blur Tool

A professional Python application with a sleek GUI for blurring specific regions of videos for custom durations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🎯 Region Selection**: Click and drag to select blur regions directly on the video preview
- **⏰ Custom Timing**: Set precise start and end times for each blur region
- **🔧 Adjustable Blur Strength**: Control the intensity of the blur effect (5-151)
- **📋 Multiple Regions**: Add as many blur regions as needed with different timings
- **▶️ Live Preview**: Play back the video with blur effects applied in real-time
- **📊 Progress Tracking**: Visual progress bar during export
- **🎨 Modern Dark UI**: Professional dark theme with intuitive controls

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy Pillow
```

## 📖 Usage

### Launch the Application

```bash
python video_blur_tool.py
```

### Step-by-Step Guide

1. **Open a Video**
   - Click "📂 Open Video" button
   - Select your video file (supports MP4, AVI, MOV, MKV, WebM, WMV)

2. **Navigate to Start Position**
   - Use the timeline slider to find where you want the blur to begin
   - Use playback controls (⏮️ -5s, ⏪ -1s, ⏩ +1s, ⏭️ +5s) for precise navigation

3. **Set Start Time**
   - Click "Set" next to "Start Time" to capture the current position
   - Or manually enter the time in seconds

4. **Navigate to End Position**
   - Move the timeline to where you want the blur to end

5. **Set End Time**
   - Click "Set" next to "End Time" to capture the current position

6. **Adjust Blur Strength**
   - Use the slider to set blur intensity (higher = more blur)

7. **Draw Blur Region**
   - Click and drag on the video preview to draw a rectangle
   - The region will be added to the list automatically

8. **Add More Regions (Optional)**
   - Repeat steps 2-7 for additional blur regions
   - Each region can have different timing and blur strength

9. **Preview**
   - Click "▶️ Play" to preview the video with blur effects
   - Active blur regions are highlighted in red, inactive in gray

10. **Export**
    - Click "🚀 Export Blurred Video"
    - Choose output location and filename
    - Wait for processing to complete

## 🎛️ Controls Reference

| Control | Function |
|---------|----------|
| Timeline Slider | Navigate through the video |
| ⏮️ -5s | Jump back 5 seconds |
| ⏪ -1s | Jump back 1 second |
| ▶️ Play / ⏸️ Pause | Toggle preview playback |
| ⏩ +1s | Jump forward 1 second |
| ⏭️ +5s | Jump forward 5 seconds |
| Start Time "Set" | Set blur start from current position |
| End Time "Set" | Set blur end from current position |
| Blur Strength | Adjust Gaussian blur kernel size |
| 🗑️ Delete | Remove selected blur region |
| ✏️ Update | Update selected region with new settings |
| 🧹 Clear All | Remove all blur regions |

## 📁 Supported Formats

### Input
- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WebM (.webm)
- WMV (.wmv)

### Output
- MP4 (.mp4)
- AVI (.avi)

## 🔧 Technical Details

### BlurRegion Properties
- **Position**: X, Y coordinates in video pixels
- **Size**: Width and height in pixels
- **Timing**: Start and end time in seconds
- **Blur Strength**: Gaussian blur kernel size (must be odd, 5-151)

### Processing
- Uses OpenCV's GaussianBlur for high-quality blur effect
- Frame-by-frame processing with progress tracking
- Maintains original video resolution and frame rate

## 💡 Tips

1. **Fine-tune timing**: Use the "Set" buttons to quickly capture the current timeline position
2. **Preview first**: Always preview before exporting to verify blur placement
3. **Multiple passes**: Export multiple times with different regions for complex edits
4. **Blur strength**: Start with 51 (default) and adjust as needed

## ⚠️ Known Limitations

- Audio is not preserved in exported video (use external tool to add back)
- Large files may take significant time to process
- GUI requires display (not suitable for headless servers)

## 🔮 Future Improvements

- [ ] Audio preservation during export
- [ ] Face detection auto-blur
- [ ] Motion tracking for moving subjects
- [ ] Batch processing
- [ ] Custom blur shapes (circle, ellipse)
- [ ] Pixelate option as alternative to blur

## 📄 License

MIT License - feel free to use and modify!

---

Made with ❤️ for privacy-conscious video editing
