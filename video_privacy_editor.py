#!/usr/bin/env python3
"""
Video Privacy Editor - Professional Grade
A desktop video privacy editing application with object tracking and blurring.

Uses PyQt6 for GUI and OpenCV for video processing with CSRT tracking.
Author: Red Coder
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QProgressBar, QFrame, QGroupBox, QStatusBar, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont


@dataclass
class TrackedRegion:
    """Represents a tracked region with its bounding box"""
    x: int
    y: int
    width: int
    height: int
    blur_strength: int = 51
    is_tracking: bool = False


class VideoProcessor(QThread):
    """
    Video processing thread that handles:
    - Frame reading
    - Object tracking (CSRT)
    - Blur application
    - Video export
    
    Uses QThread to keep GUI responsive during processing.
    """
    
    # Signals for communicating with the main thread
    frame_ready = pyqtSignal(np.ndarray, int)  # Frame and frame number
    progress_updated = pyqtSignal(int)  # Progress percentage
    processing_finished = pyqtSignal(str)  # Success message
    error_occurred = pyqtSignal(str)  # Error message
    tracking_updated = pyqtSignal(tuple)  # Updated bounding box
    
    def __init__(self):
        super().__init__()
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Video properties
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.width: int = 0
        self.height: int = 0
        
        # Processing state
        self.is_running = False
        self.is_exporting = False
        self.export_path: Optional[str] = None
        
        # Tracking
        self.tracker: Optional[cv2.Tracker] = None
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.is_tracking = False
        self.tracking_initialized = False
        
        # Blur settings
        self.blur_strength = 51
        
    def load_video(self, path: str) -> bool:
        """Load a video file and extract its properties"""
        try:
            self.video_path = path
            self.cap = cv2.VideoCapture(path)
            
            if not self.cap.isOpened():
                return False
                
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to load video: {e}")
            return False
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame from the video"""
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def initialize_tracker(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Initialize the CSRT tracker with a region of interest.
        
        CSRT (Discriminative Correlation Filter with Channel and 
        Spatial Reliability) provides higher accuracy tracking 
        compared to KCF, making it suitable for professional use.
        
        Falls back to KCF if CSRT is not available.
        
        Args:
            frame: The frame to initialize tracking on
            roi: Bounding box as (x, y, width, height)
        """
        try:
            # Try multiple ways to create a tracker
            tracker = None
            tracker_type = "Unknown"
            
            # Method 1: Try CSRT directly (OpenCV 4.5+)
            if tracker is None:
                try:
                    tracker = cv2.TrackerCSRT_create()
                    tracker_type = "CSRT"
                except AttributeError:
                    pass
            
            # Method 2: Try legacy module CSRT
            if tracker is None:
                try:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker_type = "CSRT (legacy)"
                except (AttributeError, cv2.error):
                    pass
            
            # Method 3: Fallback to KCF (more widely available)
            if tracker is None:
                try:
                    tracker = cv2.TrackerKCF_create()
                    tracker_type = "KCF"
                except AttributeError:
                    pass
            
            # Method 4: Try legacy KCF
            if tracker is None:
                try:
                    tracker = cv2.legacy.TrackerKCF_create()
                    tracker_type = "KCF (legacy)"
                except (AttributeError, cv2.error):
                    pass
            
            # Method 5: Try MOSSE (very basic but always available)
            if tracker is None:
                try:
                    tracker = cv2.TrackerMIL_create()
                    tracker_type = "MIL"
                except AttributeError:
                    pass
            
            if tracker is None:
                self.error_occurred.emit("No compatible tracker found. Please install opencv-contrib-python.")
                return
            
            self.tracker = tracker
            self.roi = roi
            self.tracker.init(frame, roi)
            self.tracking_initialized = True
            self.is_tracking = True
            
            print(f"Tracker initialized: {tracker_type}")
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize tracker: {e}")
            self.tracking_initialized = False
    
    def update_tracking(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Update the tracker with a new frame.
        
        Returns the new bounding box if tracking successful, None otherwise.
        """
        if not self.tracking_initialized or self.tracker is None:
            return None
        
        success, bbox = self.tracker.update(frame)
        
        if success:
            # Convert to integers and ensure valid coordinates
            x, y, w, h = [int(v) for v in bbox]
            self.roi = (x, y, w, h)
            return (x, y, w, h)
        
        return None
    
    def apply_blur(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply Gaussian blur to a region of interest.
        
        Handles edge cases where the ROI extends beyond frame boundaries.
        
        Args:
            frame: The input frame
            roi: Bounding box as (x, y, width, height)
            
        Returns:
            Frame with blur applied to the ROI
        """
        result = frame.copy()
        x, y, w, h = roi
        
        # Clamp coordinates to frame boundaries (handles edge cases)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        # Only apply blur if we have a valid region
        if x2 > x1 and y2 > y1:
            region = result[y1:y2, x1:x2]
            
            # Ensure blur kernel is odd
            blur_size = self.blur_strength
            if blur_size % 2 == 0:
                blur_size += 1
            
            blurred = cv2.GaussianBlur(region, (blur_size, blur_size), 0)
            result[y1:y2, x1:x2] = blurred
        
        return result
    
    def run(self):
        """Main processing loop - runs in separate thread"""
        if self.is_exporting:
            self._export_video()
        else:
            self._preview_loop()
    
    def _preview_loop(self):
        """Preview loop with real-time tracking and blurring"""
        if self.cap is None:
            return
        
        self.is_running = True
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        while self.is_running and frame_number < self.total_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Update tracking if initialized
            if self.is_tracking and self.tracking_initialized:
                new_roi = self.update_tracking(frame)
                if new_roi:
                    self.tracking_updated.emit(new_roi)
                    frame = self.apply_blur(frame, new_roi)
            
            self.frame_ready.emit(frame, frame_number)
            frame_number += 1
            
            # Control playback speed
            self.msleep(int(1000 / self.fps))
    
    def _export_video(self):
        """Export the processed video with tracking and blur applied"""
        if self.cap is None or self.export_path is None:
            return
        
        try:
            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Re-initialize tracker if we have an ROI
            if self.roi:
                ret, first_frame = self.cap.read()
                if ret:
                    self.initialize_tracker(first_frame, self.roi)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                self.export_path, fourcc, self.fps, 
                (self.width, self.height)
            )
            
            frame_number = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Update tracking and apply blur
                if self.is_tracking and self.tracking_initialized:
                    new_roi = self.update_tracking(frame)
                    if new_roi:
                        frame = self.apply_blur(frame, new_roi)
                
                out.write(frame)
                frame_number += 1
                
                # Update progress
                progress = int((frame_number / self.total_frames) * 100)
                self.progress_updated.emit(progress)
            
            out.release()
            self.processing_finished.emit(f"Video exported to: {self.export_path}")
            
        except Exception as e:
            self.error_occurred.emit(f"Export failed: {e}")
        finally:
            self.is_exporting = False
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        self.is_tracking = False


class VideoCanvas(QLabel):
    """
    Custom widget for displaying video frames and handling ROI selection.
    
    Supports mouse-based bounding box drawing for object selection.
    """
    
    roi_selected = pyqtSignal(tuple)  # Emits (x, y, width, height)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #1a1a2e; border: 2px solid #16213e;")
        self.setMinimumSize(640, 480)
        
        # Selection state
        self.is_selecting = False
        self.selection_start: Optional[QPoint] = None
        self.selection_rect: Optional[QRect] = None
        
        # Current frame for coordinate conversion
        self.current_frame: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        
        # ROI display
        self.current_roi: Optional[Tuple[int, int, int, int]] = None
        self.selection_mode = False
        
    def enable_selection(self, enable: bool):
        """Enable or disable ROI selection mode"""
        self.selection_mode = enable
        if enable:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def display_frame(self, frame: np.ndarray):
        """Display a frame on the canvas with proper scaling"""
        self.current_frame = frame
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Calculate scaling to fit widget
        widget_w = self.width()
        widget_h = self.height()
        
        scale_x = widget_w / w
        scale_y = widget_h / h
        self.scale_factor = min(scale_x, scale_y)
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        self.offset_x = (widget_w - new_w) // 2
        self.offset_y = (widget_h - new_h) // 2
        
        # Resize frame
        scaled = cv2.resize(rgb_frame, (new_w, new_h))
        
        # Convert to QImage and display
        bytes_per_line = ch * new_w
        q_img = QImage(scaled.data, new_w, new_h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Draw ROI if present
        if self.current_roi:
            pixmap = self._draw_roi_on_pixmap(pixmap)
        
        # Draw selection rectangle if selecting
        if self.selection_rect and self.is_selecting:
            pixmap = self._draw_selection_on_pixmap(pixmap)
        
        self.setPixmap(pixmap)
    
    def _draw_roi_on_pixmap(self, pixmap: QPixmap) -> QPixmap:
        """Draw the current ROI rectangle on the pixmap"""
        if not self.current_roi:
            return pixmap
        
        x, y, w, h = self.current_roi
        
        # Convert to widget coordinates
        wx = int(x * self.scale_factor)
        wy = int(y * self.scale_factor)
        ww = int(w * self.scale_factor)
        wh = int(h * self.scale_factor)
        
        painter = QPainter(pixmap)
        pen = QPen(QColor(0, 255, 136), 3)
        painter.setPen(pen)
        painter.drawRect(wx, wy, ww, wh)
        
        # Draw label
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(wx + 5, wy - 5, "🎯 Tracking")
        
        painter.end()
        return pixmap
    
    def _draw_selection_on_pixmap(self, pixmap: QPixmap) -> QPixmap:
        """Draw selection rectangle during ROI selection"""
        if not self.selection_rect:
            return pixmap
        
        painter = QPainter(pixmap)
        pen = QPen(QColor(88, 166, 255), 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        # Adjust for offset
        rect = self.selection_rect.translated(-self.offset_x, -self.offset_y)
        painter.drawRect(rect)
        
        painter.end()
        return pixmap
    
    def mousePressEvent(self, event):
        """Start ROI selection"""
        if not self.selection_mode:
            return
        
        self.is_selecting = True
        self.selection_start = event.pos()
        self.selection_rect = QRect(self.selection_start, self.selection_start)
    
    def mouseMoveEvent(self, event):
        """Update selection rectangle during drag"""
        if not self.is_selecting or not self.selection_start:
            return
        
        self.selection_rect = QRect(self.selection_start, event.pos()).normalized()
        
        # Redraw current frame to show selection
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
    
    def mouseReleaseEvent(self, event):
        """Complete ROI selection and emit signal"""
        if not self.is_selecting:
            return
        
        self.is_selecting = False
        
        if self.selection_rect and self.current_frame is not None:
            # Convert widget coordinates back to video coordinates
            x = int((self.selection_rect.x() - self.offset_x) / self.scale_factor)
            y = int((self.selection_rect.y() - self.offset_y) / self.scale_factor)
            w = int(self.selection_rect.width() / self.scale_factor)
            h = int(self.selection_rect.height() / self.scale_factor)
            
            # Validate minimum size
            if w > 10 and h > 10:
                # Clamp to video dimensions
                h_frame, w_frame = self.current_frame.shape[:2]
                x = max(0, min(x, w_frame - w))
                y = max(0, min(y, h_frame - h))
                
                self.current_roi = (x, y, w, h)
                self.roi_selected.emit((x, y, w, h))
        
        self.selection_rect = None
        self.enable_selection(False)


class MainWindow(QMainWindow):
    """
    Main application window for the Video Privacy Editor.
    
    Provides the complete GUI with video player, controls, and settings.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 Video Privacy Editor - Professional")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self._get_stylesheet())
        
        # Video processor
        self.processor = VideoProcessor()
        self._connect_processor_signals()
        
        # State
        self.current_frame_number = 0
        self.is_playing = False
        
        # Create UI
        self._create_ui()
        self._create_menu()
        
    def _get_stylesheet(self) -> str:
        """Return the application stylesheet (dark theme)"""
        return """
            QMainWindow {
                background-color: #0d1117;
            }
            QLabel {
                color: #c9d1d9;
                font-family: 'Segoe UI';
            }
            QPushButton {
                background-color: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #30363d;
                border-color: #58a6ff;
            }
            QPushButton:pressed {
                background-color: #161b22;
            }
            QPushButton:disabled {
                background-color: #161b22;
                color: #6e7681;
            }
            QPushButton#primary {
                background-color: #238636;
                border-color: #238636;
            }
            QPushButton#primary:hover {
                background-color: #2ea043;
            }
            QPushButton#danger {
                background-color: #da3633;
                border-color: #da3633;
            }
            QPushButton#danger:hover {
                background-color: #f85149;
            }
            QSlider::groove:horizontal {
                background: #21262d;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #58a6ff;
                border-radius: 4px;
            }
            QGroupBox {
                color: #58a6ff;
                font-weight: bold;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QProgressBar {
                background-color: #21262d;
                border: none;
                border-radius: 4px;
                height: 20px;
                text-align: center;
                color: #c9d1d9;
            }
            QProgressBar::chunk {
                background-color: #238636;
                border-radius: 4px;
            }
            QStatusBar {
                background-color: #161b22;
                color: #8b949e;
            }
        """
    
    def _create_ui(self):
        """Create the main UI layout"""
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Left panel - Video canvas
        left_panel = QVBoxLayout()
        
        # Title
        title = QLabel("📹 Video Preview")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #58a6ff;")
        left_panel.addWidget(title)
        
        # Video canvas
        self.canvas = VideoCanvas()
        self.canvas.roi_selected.connect(self._on_roi_selected)
        left_panel.addWidget(self.canvas, 1)
        
        # Timeline slider
        timeline_layout = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        timeline_layout.addWidget(self.time_label)
        
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self._on_timeline_change)
        timeline_layout.addWidget(self.timeline, 1)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        timeline_layout.addWidget(self.frame_label)
        
        left_panel.addLayout(timeline_layout)
        
        # Playback controls
        controls = QHBoxLayout()
        
        self.btn_load = QPushButton("📂 Load Video")
        self.btn_load.clicked.connect(self._load_video)
        controls.addWidget(self.btn_load)
        
        self.btn_play = QPushButton("▶️ Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_playback)
        controls.addWidget(self.btn_stop)
        
        controls.addStretch()
        left_panel.addLayout(controls)
        
        layout.addLayout(left_panel, 3)
        
        # Right panel - Controls
        right_panel = self._create_right_panel()
        layout.addWidget(right_panel, 1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a video to begin")
    
    def _create_right_panel(self) -> QWidget:
        """Create the right control panel"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout(panel)
        
        # ROI Selection group
        roi_group = QGroupBox("🎯 Region Selection")
        roi_layout = QVBoxLayout(roi_group)
        
        self.btn_select_roi = QPushButton("📍 Select Region of Interest")
        self.btn_select_roi.setEnabled(False)
        self.btn_select_roi.clicked.connect(self._start_roi_selection)
        roi_layout.addWidget(self.btn_select_roi)
        
        self.roi_label = QLabel("No region selected")
        self.roi_label.setStyleSheet("color: #8b949e;")
        roi_layout.addWidget(self.roi_label)
        
        layout.addWidget(roi_group)
        
        # Tracking group
        track_group = QGroupBox("🔄 Object Tracking")
        track_layout = QVBoxLayout(track_group)
        
        self.btn_start_track = QPushButton("🎯 Start Tracking")
        self.btn_start_track.setObjectName("primary")
        self.btn_start_track.setEnabled(False)
        self.btn_start_track.clicked.connect(self._start_tracking)
        track_layout.addWidget(self.btn_start_track)
        
        self.btn_stop_track = QPushButton("🛑 Stop Tracking")
        self.btn_stop_track.setObjectName("danger")
        self.btn_stop_track.setEnabled(False)
        self.btn_stop_track.clicked.connect(self._stop_tracking)
        track_layout.addWidget(self.btn_stop_track)
        
        self.tracking_status = QLabel("⚪ Tracking: Inactive")
        track_layout.addWidget(self.tracking_status)
        
        layout.addWidget(track_group)
        
        # Blur settings group
        blur_group = QGroupBox("🌫️ Blur Settings")
        blur_layout = QVBoxLayout(blur_group)
        
        blur_label = QLabel("Blur Intensity:")
        blur_layout.addWidget(blur_label)
        
        slider_layout = QHBoxLayout()
        self.blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_slider.setMinimum(5)
        self.blur_slider.setMaximum(151)
        self.blur_slider.setValue(51)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.valueChanged.connect(self._on_blur_change)
        slider_layout.addWidget(self.blur_slider)
        
        self.blur_value_label = QLabel("51")
        self.blur_value_label.setMinimumWidth(30)
        slider_layout.addWidget(self.blur_value_label)
        
        blur_layout.addLayout(slider_layout)
        
        layout.addWidget(blur_group)
        
        # Export group
        export_group = QGroupBox("💾 Export")
        export_layout = QVBoxLayout(export_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        export_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready to export")
        self.progress_label.setStyleSheet("color: #8b949e;")
        export_layout.addWidget(self.progress_label)
        
        self.btn_export = QPushButton("🚀 Export Video")
        self.btn_export.setObjectName("primary")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_video)
        export_layout.addWidget(self.btn_export)
        
        layout.addWidget(export_group)
        
        # Tips
        tips = QLabel("💡 Tips:\n• Load video first\n• Select region to track\n• Start tracking to preview\n• Export when satisfied")
        tips.setStyleSheet("color: #6e7681; font-size: 11px;")
        tips.setWordWrap(True)
        layout.addWidget(tips)
        
        layout.addStretch()
        return panel
    
    def _create_menu(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        file_menu.addAction("Open Video", self._load_video)
        file_menu.addAction("Export Video", self._export_video)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self._show_about)
    
    def _connect_processor_signals(self):
        """Connect processor signals to slots"""
        self.processor.frame_ready.connect(self._on_frame_ready)
        self.processor.progress_updated.connect(self._on_progress_update)
        self.processor.processing_finished.connect(self._on_processing_finished)
        self.processor.error_occurred.connect(self._on_error)
        self.processor.tracking_updated.connect(self._on_tracking_updated)
    
    def _load_video(self):
        """Open file dialog and load a video"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*)"
        )
        
        if path and self.processor.load_video(path):
            # Display first frame
            frame = self.processor.get_frame(0)
            if frame is not None:
                self.canvas.display_frame(frame)
            
            # Update UI
            self.timeline.setMaximum(self.processor.total_frames - 1)
            self.timeline.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_select_roi.setEnabled(True)
            self.btn_export.setEnabled(True)
            
            # Update labels
            duration = self.processor.total_frames / self.processor.fps
            self.time_label.setText(f"00:00 / {self._format_time(duration)}")
            self.frame_label.setText(f"Frame: 0 / {self.processor.total_frames}")
            
            self.status_bar.showMessage(f"Loaded: {Path(path).name} | {self.processor.width}x{self.processor.height} @ {self.processor.fps:.1f} FPS")
    
    def _start_roi_selection(self):
        """Enable ROI selection mode on canvas"""
        self.canvas.enable_selection(True)
        self.status_bar.showMessage("🎯 Draw a box around the object to track")
    
    def _on_roi_selected(self, roi: Tuple[int, int, int, int]):
        """Handle ROI selection completion"""
        x, y, w, h = roi
        self.roi_label.setText(f"Region: {w}x{h} at ({x}, {y})")
        self.btn_start_track.setEnabled(True)
        self.status_bar.showMessage(f"✅ Region selected: {w}x{h} pixels")
        
        # Refresh canvas to show ROI
        frame = self.processor.get_frame(self.current_frame_number)
        if frame is not None:
            self.canvas.display_frame(frame)
    
    def _start_tracking(self):
        """Initialize and start object tracking"""
        if not self.canvas.current_roi:
            return
        
        frame = self.processor.get_frame(self.current_frame_number)
        if frame is not None:
            self.processor.initialize_tracker(frame, self.canvas.current_roi)
            self.processor.blur_strength = self.blur_slider.value()
            
            self.btn_start_track.setEnabled(False)
            self.btn_stop_track.setEnabled(True)
            self.tracking_status.setText("🟢 Tracking: Active")
            self.tracking_status.setStyleSheet("color: #3fb950;")
            
            self.status_bar.showMessage("🎯 Tracking started - Press Play to see it in action")
    
    def _stop_tracking(self):
        """Stop object tracking"""
        self.processor.is_tracking = False
        self.processor.tracking_initialized = False
        
        self.btn_start_track.setEnabled(True)
        self.btn_stop_track.setEnabled(False)
        self.tracking_status.setText("⚪ Tracking: Inactive")
        self.tracking_status.setStyleSheet("color: #8b949e;")
        
        self.status_bar.showMessage("Tracking stopped")
    
    def _toggle_play(self):
        """Toggle video playback"""
        if self.is_playing:
            self.processor.stop()
            self.processor.wait()
            self.is_playing = False
            self.btn_play.setText("▶️ Play")
        else:
            self.is_playing = True
            self.btn_play.setText("⏸️ Pause")
            
            # Start processing thread
            self.processor.is_exporting = False
            self.processor.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_number)
            self.processor.start()
    
    def _stop_playback(self):
        """Stop playback and reset to beginning"""
        self.processor.stop()
        if self.processor.isRunning():
            self.processor.wait()
        
        self.is_playing = False
        self.btn_play.setText("▶️ Play")
        self.current_frame_number = 0
        self.timeline.setValue(0)
        
        # Show first frame
        frame = self.processor.get_frame(0)
        if frame is not None:
            self.canvas.display_frame(frame)
    
    def _on_frame_ready(self, frame: np.ndarray, frame_number: int):
        """Handle new frame from processor"""
        self.current_frame_number = frame_number
        self.canvas.display_frame(frame)
        self.timeline.blockSignals(True)
        self.timeline.setValue(frame_number)
        self.timeline.blockSignals(False)
        
        # Update time labels
        current_time = frame_number / self.processor.fps
        total_time = self.processor.total_frames / self.processor.fps
        self.time_label.setText(f"{self._format_time(current_time)} / {self._format_time(total_time)}")
        self.frame_label.setText(f"Frame: {frame_number} / {self.processor.total_frames}")
    
    def _on_tracking_updated(self, roi: Tuple[int, int, int, int]):
        """Handle tracking position update"""
        self.canvas.current_roi = roi
    
    def _on_timeline_change(self, value: int):
        """Handle timeline slider change"""
        if not self.is_playing:
            frame = self.processor.get_frame(value)
            if frame is not None:
                self.current_frame_number = value
                
                # Apply blur if tracking
                if self.processor.is_tracking and self.canvas.current_roi:
                    frame = self.processor.apply_blur(frame, self.canvas.current_roi)
                
                self.canvas.display_frame(frame)
                
                current_time = value / self.processor.fps
                total_time = self.processor.total_frames / self.processor.fps
                self.time_label.setText(f"{self._format_time(current_time)} / {self._format_time(total_time)}")
                self.frame_label.setText(f"Frame: {value} / {self.processor.total_frames}")
    
    def _on_blur_change(self, value: int):
        """Handle blur slider change"""
        # Ensure odd value for Gaussian blur
        if value % 2 == 0:
            value += 1
        self.blur_value_label.setText(str(value))
        self.processor.blur_strength = value
    
    def _export_video(self):
        """Export the processed video"""
        if not self.processor.tracking_initialized:
            QMessageBox.warning(self, "Warning", "Please start tracking before exporting.")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", 
            f"{Path(self.processor.video_path).stem}_blurred.mp4",
            "MP4 Files (*.mp4);;AVI Files (*.avi)"
        )
        
        if path:
            self.processor.export_path = path
            self.processor.is_exporting = True
            self.processor.is_running = True
            self.progress_bar.setValue(0)
            self.progress_label.setText("Exporting...")
            self.btn_export.setEnabled(False)
            
            self.processor.start()
    
    def _on_progress_update(self, progress: int):
        """Update export progress"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Exporting: {progress}%")
    
    def _on_processing_finished(self, message: str):
        """Handle processing completion"""
        self.progress_bar.setValue(100)
        self.progress_label.setText("Export complete!")
        self.btn_export.setEnabled(True)
        QMessageBox.information(self, "Success", message)
    
    def _on_error(self, message: str):
        """Handle processing error"""
        QMessageBox.critical(self, "Error", message)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to mm:ss"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Video Privacy Editor",
            "🎬 Video Privacy Editor - Professional\n\n"
            "A desktop video privacy editing application with:\n"
            "• CSRT Object Tracking\n"
            "• Real-time Gaussian Blur\n"
            "• Professional dark theme\n\n"
            "Built with PyQt6 and OpenCV"
        )
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.processor.stop()
        if self.processor.isRunning():
            self.processor.wait()
        event.accept()


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Video Privacy Editor")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
