#!/usr/bin/env python3
"""
Video Blur Tool v2 - Smart Auto-Detection Edition
Author: Red Coder
Features: Face detection, object tracking, license plate blur, smart presets
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import threading
import os
from pathlib import Path
from enum import Enum
import json


class BlurMode(Enum):
    MANUAL = "manual"
    FACE = "face"
    OBJECT_TRACK = "track"
    LICENSE_PLATE = "plate"


@dataclass
class BlurRegion:
    """Represents a blur region with timing and tracking info"""
    x: int
    y: int
    width: int
    height: int
    start_time: float
    end_time: float
    blur_strength: int = 51
    mode: BlurMode = BlurMode.MANUAL
    track_id: Optional[int] = None
    tracked_positions: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    
    def contains_frame(self, current_time: float) -> bool:
        return self.start_time <= current_time <= self.end_time
    
    def get_position_at_frame(self, frame_num: int) -> Tuple[int, int, int, int]:
        """Get interpolated position for a frame"""
        if frame_num in self.tracked_positions:
            return self.tracked_positions[frame_num]
        
        # Find nearest tracked frames for interpolation
        frames = sorted(self.tracked_positions.keys())
        if not frames:
            return (self.x, self.y, self.width, self.height)
        
        if frame_num <= frames[0]:
            return self.tracked_positions[frames[0]]
        if frame_num >= frames[-1]:
            return self.tracked_positions[frames[-1]]
        
        # Linear interpolation between frames
        for i in range(len(frames) - 1):
            if frames[i] <= frame_num <= frames[i + 1]:
                t = (frame_num - frames[i]) / (frames[i + 1] - frames[i])
                p1 = self.tracked_positions[frames[i]]
                p2 = self.tracked_positions[frames[i + 1]]
                return (
                    int(p1[0] + t * (p2[0] - p1[0])),
                    int(p1[1] + t * (p2[1] - p1[1])),
                    int(p1[2] + t * (p2[2] - p1[2])),
                    int(p1[3] + t * (p2[3] - p1[3]))
                )
        
        return (self.x, self.y, self.width, self.height)


class SmartVideoBlurTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Smart Video Blur Tool v2")
        self.root.geometry("1300x850")
        self.root.configure(bg="#0d1117")
        
        # Video properties
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.video_width: int = 0
        self.video_height: int = 0
        self.duration: float = 0.0
        
        # Detection models
        self.face_cascade = None
        self.tracker = None
        self._load_detection_models()
        
        # Blur regions
        self.blur_regions: List[BlurRegion] = []
        self.current_region_id: Optional[int] = None
        
        # Selection state
        self.is_selecting = False
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_rect: Optional[int] = None
        self.temp_rect: Optional[Tuple[int, int, int, int]] = None
        
        # Canvas scaling
        self.scale_factor: float = 1.0
        self.canvas_offset_x: int = 0
        self.canvas_offset_y: int = 0
        
        # Processing state
        self.is_processing = False
        self.preview_running = False
        self.auto_detect_mode = tk.StringVar(value="none")
        
        # Presets
        self.blur_presets = {
            "Light": 21,
            "Medium": 51,
            "Heavy": 99,
            "Maximum": 151
        }
        
        self._setup_styles()
        self._create_ui()
        
    def _load_detection_models(self):
        """Load face detection cascade"""
        try:
            # Try to load Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Also load profile face detector
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
        except Exception as e:
            print(f"Warning: Could not load face cascade: {e}")
            self.face_cascade = None
            
    def _setup_styles(self):
        """Configure custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colors - GitHub dark theme inspired
        bg_dark = "#0d1117"
        bg_medium = "#161b22"
        bg_light = "#21262d"
        accent = "#58a6ff"
        accent_green = "#3fb950"
        accent_orange = "#d29922"
        text_light = "#c9d1d9"
        border = "#30363d"
        
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=accent)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=text_light)
        style.configure("Success.TLabel", foreground=accent_green)
        
        style.configure("TLabelframe", background=bg_dark, foreground=text_light)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=accent, font=("Segoe UI", 10, "bold"))
        
        style.configure("TButton", 
            background=bg_light, foreground=text_light,
            font=("Segoe UI", 10), padding=(12, 8),
            borderwidth=1)
        style.map("TButton",
            background=[("active", bg_medium), ("pressed", accent)])
        
        style.configure("Accent.TButton",
            background=accent, foreground="#ffffff",
            font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton",
            background=[("active", "#79c0ff")])
        
        style.configure("Success.TButton",
            background=accent_green, foreground="#ffffff",
            font=("Segoe UI", 10, "bold"))
        
        style.configure("TRadiobutton", background=bg_dark, foreground=text_light)
        style.configure("TCheckbutton", background=bg_dark, foreground=text_light)
        style.configure("TScale", background=bg_dark, troughcolor=bg_medium)
        
        style.configure("Treeview",
            background=bg_medium, foreground=text_light,
            fieldbackground=bg_medium, font=("Segoe UI", 9),
            rowheight=28)
        style.configure("Treeview.Heading",
            background=bg_light, foreground=text_light,
            font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", accent)])
        
    def _create_ui(self):
        """Create the main user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video preview
        left_panel = ttk.Frame(main_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Title with status
        title_frame = ttk.Frame(left_panel)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="📹 Video Preview", style="Title.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(title_frame, text="No video loaded", style="Header.TLabel")
        self.status_label.pack(side=tk.RIGHT)
        
        # Canvas for video display
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#010409", highlightthickness=2,
                                highlightbackground="#30363d", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        
        # Timeline
        timeline_frame = ttk.Frame(left_panel)
        timeline_frame.pack(fill=tk.X, pady=10)
        
        self.time_var = tk.DoubleVar(value=0)
        self.timeline_slider = ttk.Scale(timeline_frame, from_=0, to=100,
                                         variable=self.time_var, orient=tk.HORIZONTAL,
                                         command=self._on_timeline_change)
        self.timeline_slider.pack(fill=tk.X, pady=(0, 5))
        
        # Time display
        time_info = ttk.Frame(timeline_frame)
        time_info.pack(fill=tk.X)
        self.time_label = ttk.Label(time_info, text="00:00.00 / 00:00.00")
        self.time_label.pack(side=tk.LEFT)
        self.frame_label = ttk.Label(time_info, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Playback controls
        playback_frame = ttk.Frame(left_panel)
        playback_frame.pack(fill=tk.X, pady=5)
        
        # Quick seek buttons
        seek_btns = [
            ("⏮️", -10), ("◀◀", -5), ("◀", -1),
            ("▶️ Play", "play"),
            ("▶", 1), ("▶▶", 5), ("⏭️", 10)
        ]
        
        for text, action in seek_btns:
            if action == "play":
                self.play_btn = ttk.Button(playback_frame, text=text, command=self._toggle_preview)
                self.play_btn.pack(side=tk.LEFT, padx=2)
            else:
                ttk.Button(playback_frame, text=text,
                          command=lambda a=action: self._seek_relative(a)).pack(side=tk.LEFT, padx=2)
        
        # Frame step buttons
        ttk.Button(playback_frame, text="← Frame", 
                   command=lambda: self._step_frame(-1)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(playback_frame, text="Frame →",
                   command=lambda: self._step_frame(1)).pack(side=tk.RIGHT, padx=2)
        
        # Right panel - Controls
        right_panel = ttk.Frame(main_frame, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Create scrollable frame for controls
        canvas_scroll = tk.Canvas(right_panel, bg="#0d1117", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw", width=360)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(scrollable_frame, text="📁 Video File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=320)
        self.file_label.pack(fill=tk.X)
        
        ttk.Button(file_frame, text="📂 Open Video", style="Accent.TButton",
                   command=self._open_video).pack(fill=tk.X, pady=(10, 0))
        
        # ========== SMART DETECTION SECTION ==========
        detect_frame = ttk.LabelFrame(scrollable_frame, text="🤖 Smart Auto-Detection", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Label(detect_frame, text="One-click detection options:", 
                  style="Header.TLabel").pack(anchor=tk.W)
        
        # Detection buttons
        detect_btns = ttk.Frame(detect_frame)
        detect_btns.pack(fill=tk.X, pady=10)
        
        ttk.Button(detect_btns, text="👤 Detect Faces", style="Success.TButton",
                   command=self._auto_detect_faces).pack(fill=tk.X, pady=2)
        
        ttk.Button(detect_btns, text="👥 Detect All Faces in Video",
                   command=self._scan_all_faces).pack(fill=tk.X, pady=2)
        
        ttk.Button(detect_btns, text="🚗 Detect License Plates",
                   command=self._detect_license_plates).pack(fill=tk.X, pady=2)
        
        # Auto-track option
        self.auto_track_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detect_frame, text="🎯 Auto-track detected objects",
                        variable=self.auto_track_var).pack(anchor=tk.W, pady=5)
        
        # Detection sensitivity
        sens_frame = ttk.Frame(detect_frame)
        sens_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sens_frame, text="Detection Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.DoubleVar(value=1.2)
        ttk.Scale(sens_frame, from_=1.05, to=1.5, variable=self.sensitivity_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # ========== QUICK ACTIONS ==========
        quick_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Quick Actions", padding=10)
        quick_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Button(quick_frame, text="🔲 Blur Entire Frame (Current Time)",
                   command=self._blur_entire_frame).pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="📍 Mark Current Position as Start",
                   command=lambda: self._quick_set_time('start')).pack(fill=tk.X, pady=2)
        
        ttk.Button(quick_frame, text="📍 Mark Current Position as End",
                   command=lambda: self._quick_set_time('end')).pack(fill=tk.X, pady=2)
        
        # Preset blur buttons
        preset_frame = ttk.Frame(quick_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Label(preset_frame, text="Blur Preset:").pack(side=tk.LEFT)
        
        self.preset_var = tk.StringVar(value="Medium")
        for name in self.blur_presets:
            ttk.Radiobutton(preset_frame, text=name, variable=self.preset_var,
                           value=name, command=self._apply_preset).pack(side=tk.LEFT, padx=3)
        
        # ========== TIMING CONTROLS ==========
        timing_frame = ttk.LabelFrame(scrollable_frame, text="⏰ Region Settings", padding=10)
        timing_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Start time
        start_row = ttk.Frame(timing_frame)
        start_row.pack(fill=tk.X, pady=2)
        ttk.Label(start_row, text="Start (s):").pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value="0.0")
        ttk.Entry(start_row, textvariable=self.start_time_var, width=10).pack(side=tk.RIGHT)
        ttk.Button(start_row, text="⏱️ Now", 
                   command=lambda: self._set_time_from_slider('start')).pack(side=tk.RIGHT, padx=5)
        
        # End time
        end_row = ttk.Frame(timing_frame)
        end_row.pack(fill=tk.X, pady=2)
        ttk.Label(end_row, text="End (s):").pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value="0.0")
        ttk.Entry(end_row, textvariable=self.end_time_var, width=10).pack(side=tk.RIGHT)
        ttk.Button(end_row, text="⏱️ Now",
                   command=lambda: self._set_time_from_slider('end')).pack(side=tk.RIGHT, padx=5)
        
        # Blur strength
        blur_row = ttk.Frame(timing_frame)
        blur_row.pack(fill=tk.X, pady=5)
        ttk.Label(blur_row, text="Blur:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=51)
        self.blur_scale = ttk.Scale(blur_row, from_=5, to=151, variable=self.blur_var,
                                    orient=tk.HORIZONTAL, command=self._update_blur_label)
        self.blur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.blur_label = ttk.Label(blur_row, text="51", width=4)
        self.blur_label.pack(side=tk.RIGHT)
        
        # ========== REGIONS LIST ==========
        regions_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Blur Regions", padding=10)
        regions_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        columns = ("id", "type", "timing", "blur")
        self.regions_tree = ttk.Treeview(regions_frame, columns=columns, show="headings", height=5)
        self.regions_tree.heading("id", text="#")
        self.regions_tree.heading("type", text="Type")
        self.regions_tree.heading("timing", text="Timing")
        self.regions_tree.heading("blur", text="Blur")
        
        self.regions_tree.column("id", width=30)
        self.regions_tree.column("type", width=80)
        self.regions_tree.column("timing", width=120)
        self.regions_tree.column("blur", width=50)
        
        self.regions_tree.pack(fill=tk.X)
        self.regions_tree.bind("<<TreeviewSelect>>", self._on_region_select)
        
        # Region buttons
        region_btns = ttk.Frame(regions_frame)
        region_btns.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(region_btns, text="🗑️ Delete", 
                   command=self._delete_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="✏️ Update",
                   command=self._update_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="🔄 Re-track",
                   command=self._retrack_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="🧹 Clear",
                   command=self._clear_all_regions).pack(side=tk.RIGHT, padx=2)
        
        # ========== EXPORT ==========
        export_frame = ttk.LabelFrame(scrollable_frame, text="💾 Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(export_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(export_frame, text="Ready")
        self.progress_label.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="🚀 Export Blurred Video", style="Accent.TButton",
                   command=self._export_video).pack(fill=tk.X, pady=(10, 0))
        
        # Quick tips
        tips = ttk.Label(scrollable_frame,
            text="💡 Quick Tips:\n"
                 "• Click 'Detect Faces' for instant face blur\n"
                 "• Draw rectangles manually for custom areas\n"
                 "• Use arrow keys for frame-by-frame control\n"
                 "• Auto-track follows objects through video",
            wraplength=340, justify=tk.LEFT)
        tips.pack(fill=tk.X, pady=10, padx=5)
        
        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<space>", lambda e: self._toggle_preview())
        self.root.bind("<Home>", lambda e: self._seek_to(0))
        self.root.bind("<End>", lambda e: self._seek_to(self.duration))
        
    def _open_video(self):
        """Open a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"),
                ("All Files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self.video_path = file_path
        self.cap = cv2.VideoCapture(file_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        self.file_label.config(text=f"📄 {Path(file_path).name}\n"
                                    f"📐 {self.video_width}x{self.video_height} | "
                                    f"🎞️ {self.fps:.1f} FPS | ⏱️ {self.duration:.1f}s")
        
        self.timeline_slider.config(to=self.duration)
        self.end_time_var.set(f"{self.duration:.2f}")
        self.status_label.config(text="✅ Video loaded")
        
        self._clear_all_regions()
        self._show_frame(0)
        
    def _get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame without blur applied"""
        if self.cap is None:
            return None
            
        current_time = self.time_var.get()
        frame_number = int(current_time * self.fps)
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
        
    def _auto_detect_faces(self):
        """Detect faces in current frame and add blur regions"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
            
        if self.face_cascade is None:
            messagebox.showerror("Error", "Face detection model not loaded")
            return
            
        frame = self._get_current_frame()
        if frame is None:
            return
            
        self.status_label.config(text="🔍 Detecting faces...")
        self.root.update()
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        scale = self.sensitivity_var.get()
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
        
        # Also detect profile faces
        profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
        
        # Combine detections
        all_faces = list(faces) + list(profiles)
        
        if len(all_faces) == 0:
            self.status_label.config(text="❌ No faces detected")
            messagebox.showinfo("Info", "No faces detected in current frame.\n"
                                       "Try adjusting sensitivity or use 'Scan All Faces' for full video.")
            return
            
        # Get current timing settings
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time = self.time_var.get()
            end_time = self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        # Add regions for each face
        faces_added = 0
        current_frame = int(self.time_var.get() * self.fps)
        
        for (x, y, w, h) in all_faces:
            # Add padding around face
            padding = int(w * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(self.video_width - x, w + 2 * padding)
            h = min(self.video_height - y, h + 2 * padding)
            
            region = BlurRegion(
                x=x, y=y, width=w, height=h,
                start_time=start_time, end_time=end_time,
                blur_strength=blur_strength,
                mode=BlurMode.FACE
            )
            
            # Initialize tracking if enabled
            if self.auto_track_var.get():
                region.tracked_positions[current_frame] = (x, y, w, h)
                self._track_region_forward(region, frame, current_frame)
            
            self.blur_regions.append(region)
            faces_added += 1
            
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        
        self.status_label.config(text=f"✅ {faces_added} face(s) detected")
        
    def _scan_all_faces(self):
        """Scan entire video for faces"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
            
        if self.face_cascade is None:
            messagebox.showerror("Error", "Face detection model not loaded")
            return
            
        # Confirm with user
        if not messagebox.askyesno("Scan Video", 
            "This will scan the entire video for faces.\n"
            "This may take a while for long videos.\n\nContinue?"):
            return
            
        self.is_processing = True
        threading.Thread(target=self._scan_faces_thread, daemon=True).start()
        
    def _scan_faces_thread(self):
        """Background thread for face scanning"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample every N frames for efficiency
            sample_interval = max(1, int(self.fps / 2))  # ~2 samples per second
            
            detected_faces = []  # List of (frame_num, x, y, w, h)
            scale = self.sensitivity_var.get()
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_num % sample_interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
                    
                    for (x, y, w, h) in faces:
                        detected_faces.append((frame_num, x, y, w, h))
                        
                    # Update progress
                    progress = (frame_num / total_frames) * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    self.root.after(0, lambda f=frame_num: 
                        self.status_label.config(text=f"🔍 Scanning frame {f}/{total_frames}"))
                
                frame_num += 1
                
            cap.release()
            
            # Cluster faces into continuous regions
            self.root.after(0, lambda: self._process_detected_faces(detected_faces))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
        finally:
            self.is_processing = False
            
    def _process_detected_faces(self, detected_faces):
        """Process detected faces into blur regions"""
        if not detected_faces:
            self.status_label.config(text="❌ No faces found")
            messagebox.showinfo("Info", "No faces were detected in the video.")
            return
            
        # Simple clustering: group faces by approximate position
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        # Create regions from detections
        regions_added = 0
        
        # Group by spatial proximity
        used = set()
        for i, (f1, x1, y1, w1, h1) in enumerate(detected_faces):
            if i in used:
                continue
                
            # Find all detections close to this one
            group = [(f1, x1, y1, w1, h1)]
            used.add(i)
            
            for j, (f2, x2, y2, w2, h2) in enumerate(detected_faces):
                if j in used:
                    continue
                    
                # Check if faces overlap significantly
                cx1, cy1 = x1 + w1/2, y1 + h1/2
                cx2, cy2 = x2 + w2/2, y2 + h2/2
                
                if abs(cx1 - cx2) < (w1 + w2) / 2 and abs(cy1 - cy2) < (h1 + h2) / 2:
                    group.append((f2, x2, y2, w2, h2))
                    used.add(j)
                    
            # Create region from group
            frames = [g[0] for g in group]
            start_frame = min(frames)
            end_frame = max(frames)
            
            # Average position
            avg_x = int(np.mean([g[1] for g in group]))
            avg_y = int(np.mean([g[2] for g in group]))
            avg_w = int(np.mean([g[3] for g in group]))
            avg_h = int(np.mean([g[4] for g in group]))
            
            # Add padding
            padding = int(avg_w * 0.2)
            avg_x = max(0, avg_x - padding)
            avg_y = max(0, avg_y - padding)
            avg_w = min(self.video_width - avg_x, avg_w + 2 * padding)
            avg_h = min(self.video_height - avg_y, avg_h + 2 * padding)
            
            region = BlurRegion(
                x=avg_x, y=avg_y, width=avg_w, height=avg_h,
                start_time=start_frame / self.fps,
                end_time=end_frame / self.fps,
                blur_strength=blur_strength,
                mode=BlurMode.FACE
            )
            
            # Add tracking data
            for (f, x, y, w, h) in group:
                region.tracked_positions[f] = (x - padding, y - padding, w + 2*padding, h + 2*padding)
                
            self.blur_regions.append(region)
            regions_added += 1
            
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.progress_var.set(100)
        self.status_label.config(text=f"✅ {regions_added} face region(s) created")
        
    def _detect_license_plates(self):
        """Detect license plates (simplified - detects rectangular regions)"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
            
        frame = self._get_current_frame()
        if frame is None:
            return
            
        self.status_label.config(text="🔍 Detecting plates...")
        self.root.update()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # License plates typically have aspect ratio between 2:1 and 5:1
            if 2 < aspect_ratio < 5 and w > 60 and h > 20:
                plates.append((x, y, w, h))
                
        if not plates:
            self.status_label.config(text="❌ No plates detected")
            messagebox.showinfo("Info", "No license plate-like regions detected.\n"
                                       "Try drawing manually or adjusting the frame.")
            return
            
        # Get timing
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time = self.time_var.get()
            end_time = self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        for (x, y, w, h) in plates[:5]:  # Limit to 5 detections
            region = BlurRegion(
                x=x, y=y, width=w, height=h,
                start_time=start_time, end_time=end_time,
                blur_strength=blur_strength,
                mode=BlurMode.LICENSE_PLATE
            )
            self.blur_regions.append(region)
            
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.status_label.config(text=f"✅ {len(plates[:5])} plate region(s) detected")
        
    def _track_region_forward(self, region: BlurRegion, initial_frame: np.ndarray, start_frame: int):
        """Track a region forward through the video using optical flow"""
        if not self.auto_track_var.get():
            return
            
        # Use OpenCV's tracking
        try:
            tracker = cv2.TrackerCSRT_create()
        except:
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
            except:
                return  # Tracking not available
                
        bbox = (region.x, region.y, region.width, region.height)
        tracker.init(initial_frame, bbox)
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        end_frame = int(region.end_time * self.fps)
        frame_num = start_frame
        
        while frame_num < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                region.tracked_positions[frame_num] = (x, y, w, h)
            else:
                break
                
            frame_num += 1
            
            # Don't track too many frames to keep it responsive
            if frame_num - start_frame > 300:
                break
                
        cap.release()
        
    def _retrack_region(self):
        """Re-track the selected region"""
        selection = self.regions_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a region to re-track")
            return
            
        item = selection[0]
        idx = self.regions_tree.index(item)
        region = self.blur_regions[idx]
        
        frame = self._get_current_frame()
        if frame is None:
            return
            
        current_frame = int(self.time_var.get() * self.fps)
        region.tracked_positions.clear()
        region.tracked_positions[current_frame] = (region.x, region.y, region.width, region.height)
        
        self.status_label.config(text="🔄 Re-tracking...")
        self.root.update()
        
        self._track_region_forward(region, frame, current_frame)
        
        self._show_frame(self.time_var.get())
        self.status_label.config(text="✅ Region re-tracked")
        
    def _blur_entire_frame(self):
        """Add blur region covering the entire frame"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
            
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time = self.time_var.get()
            end_time = self.time_var.get() + 1
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        region = BlurRegion(
            x=0, y=0, width=self.video_width, height=self.video_height,
            start_time=start_time, end_time=end_time,
            blur_strength=blur_strength,
            mode=BlurMode.MANUAL
        )
        
        self.blur_regions.append(region)
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        
    def _quick_set_time(self, which: str):
        """Quick set start or end time"""
        current = self.time_var.get()
        if which == 'start':
            self.start_time_var.set(f"{current:.2f}")
        else:
            self.end_time_var.set(f"{current:.2f}")
            
    def _apply_preset(self):
        """Apply blur preset"""
        preset = self.preset_var.get()
        if preset in self.blur_presets:
            self.blur_var.set(self.blur_presets[preset])
            self._update_blur_label(None)
            
    def _show_frame(self, time_seconds: float):
        """Display a frame at the given time"""
        if self.cap is None:
            return
            
        frame_number = int(time_seconds * self.fps)
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        # Apply blur regions
        frame = self._apply_blur_regions(frame, time_seconds, frame_number)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 500
            
        scale_x = canvas_width / self.video_width
        scale_y = canvas_height / self.video_height
        self.scale_factor = min(scale_x, scale_y)
        
        new_width = int(self.video_width * self.scale_factor)
        new_height = int(self.video_height * self.scale_factor)
        
        self.canvas_offset_x = (canvas_width - new_width) // 2
        self.canvas_offset_y = (canvas_height - new_height) // 2
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to PhotoImage
        from PIL import Image, ImageTk
        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image)
        
        # Clear and redraw
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                 anchor=tk.NW, image=self.photo)
        
        # Draw region outlines
        self._draw_blur_regions(time_seconds, frame_number)
        
        # Update labels
        self.time_label.config(text=f"{self._format_time(time_seconds)} / {self._format_time(self.duration)}")
        self.frame_label.config(text=f"Frame: {frame_number} / {self.total_frames}")
        
    def _apply_blur_regions(self, frame: np.ndarray, current_time: float, frame_number: int) -> np.ndarray:
        """Apply blur to frame based on active regions"""
        result = frame.copy()
        
        for region in self.blur_regions:
            if region.contains_frame(current_time):
                # Get position (use tracking if available)
                if region.tracked_positions:
                    x, y, w, h = region.get_position_at_frame(frame_number)
                else:
                    x, y, w, h = region.x, region.y, region.width, region.height
                    
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame.shape[1], x + w)
                y2 = min(frame.shape[0], y + h)
                
                if x2 > x1 and y2 > y1:
                    roi = result[y1:y2, x1:x2]
                    blur_size = region.blur_strength
                    if blur_size % 2 == 0:
                        blur_size += 1
                    blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                    result[y1:y2, x1:x2] = blurred
                    
        return result
        
    def _draw_blur_regions(self, current_time: float, frame_number: int):
        """Draw blur region rectangles on canvas"""
        for i, region in enumerate(self.blur_regions):
            # Get position
            if region.tracked_positions:
                x, y, w, h = region.get_position_at_frame(frame_number)
            else:
                x, y, w, h = region.x, region.y, region.width, region.height
                
            # Convert to canvas coordinates
            x1 = int(x * self.scale_factor) + self.canvas_offset_x
            y1 = int(y * self.scale_factor) + self.canvas_offset_y
            x2 = int((x + w) * self.scale_factor) + self.canvas_offset_x
            y2 = int((y + h) * self.scale_factor) + self.canvas_offset_y
            
            # Color based on state
            if region.contains_frame(current_time):
                color = "#3fb950"  # Green when active
            else:
                color = "#6e7681"  # Gray when inactive
                
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            
            # Label with icon for type
            icon = {"face": "👤", "plate": "🚗", "track": "🎯", "manual": "🔲"}
            label = f"#{i+1} {icon.get(region.mode.value, '🔲')}"
            self.canvas.create_text(x1 + 5, y1 + 5, text=label,
                                   fill=color, anchor=tk.NW, font=("Segoe UI", 9, "bold"))
            
    def _on_mouse_down(self, event):
        if self.cap is None:
            return
        self.is_selecting = True
        self.selection_start = (event.x, event.y)
        
    def _on_mouse_drag(self, event):
        if not self.is_selecting or self.selection_start is None:
            return
            
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y
        
        self.selection_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#58a6ff", width=2, dash=(5, 5)
        )
        self.temp_rect = (x1, y1, x2, y2)
        
    def _on_mouse_up(self, event):
        if not self.is_selecting or self.selection_start is None:
            return
            
        self.is_selecting = False
        
        if self.temp_rect is None:
            return
            
        x1, y1, x2, y2 = self.temp_rect
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Convert to video coordinates
        video_x1 = int((x1 - self.canvas_offset_x) / self.scale_factor)
        video_y1 = int((y1 - self.canvas_offset_y) / self.scale_factor)
        video_x2 = int((x2 - self.canvas_offset_x) / self.scale_factor)
        video_y2 = int((y2 - self.canvas_offset_y) / self.scale_factor)
        
        # Clamp
        video_x1 = max(0, min(video_x1, self.video_width))
        video_y1 = max(0, min(video_y1, self.video_height))
        video_x2 = max(0, min(video_x2, self.video_width))
        video_y2 = max(0, min(video_y2, self.video_height))
        
        width = video_x2 - video_x1
        height = video_y2 - video_y1
        
        if width < 10 or height < 10:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            self.temp_rect = None
            return
            
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time = 0.0
            end_time = self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        region = BlurRegion(
            x=video_x1, y=video_y1, width=width, height=height,
            start_time=start_time, end_time=end_time,
            blur_strength=blur_strength,
            mode=BlurMode.MANUAL
        )
        
        # Auto-track if enabled
        if self.auto_track_var.get():
            frame = self._get_current_frame()
            if frame is not None:
                current_frame = int(self.time_var.get() * self.fps)
                region.tracked_positions[current_frame] = (video_x1, video_y1, width, height)
                self._track_region_forward(region, frame, current_frame)
        
        self.blur_regions.append(region)
        self._update_regions_list()
        
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        self.selection_rect = None
        self.temp_rect = None
        
        self._show_frame(self.time_var.get())
        
    def _update_regions_list(self):
        """Update the regions treeview"""
        for item in self.regions_tree.get_children():
            self.regions_tree.delete(item)
            
        mode_names = {
            BlurMode.MANUAL: "Manual",
            BlurMode.FACE: "Face",
            BlurMode.OBJECT_TRACK: "Tracked",
            BlurMode.LICENSE_PLATE: "Plate"
        }
        
        for i, region in enumerate(self.blur_regions):
            tracked = "📍" if region.tracked_positions else ""
            self.regions_tree.insert("", tk.END, values=(
                i + 1,
                f"{mode_names.get(region.mode, 'Manual')}{tracked}",
                f"{region.start_time:.1f}s → {region.end_time:.1f}s",
                region.blur_strength
            ))
            
    def _on_region_select(self, event):
        selection = self.regions_tree.selection()
        if selection:
            item = selection[0]
            idx = self.regions_tree.index(item)
            self.current_region_id = idx
            
            region = self.blur_regions[idx]
            self.start_time_var.set(f"{region.start_time:.2f}")
            self.end_time_var.set(f"{region.end_time:.2f}")
            self.blur_var.set(region.blur_strength)
            self._update_blur_label(None)
            
    def _delete_selected_region(self):
        selection = self.regions_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a region to delete")
            return
            
        item = selection[0]
        idx = self.regions_tree.index(item)
        del self.blur_regions[idx]
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        
    def _update_selected_region(self):
        selection = self.regions_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a region to update")
            return
            
        item = selection[0]
        idx = self.regions_tree.index(item)
        
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid time values")
            return
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        self.blur_regions[idx].start_time = start_time
        self.blur_regions[idx].end_time = end_time
        self.blur_regions[idx].blur_strength = blur_strength
        
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        
    def _clear_all_regions(self):
        self.blur_regions.clear()
        self._update_regions_list()
        if self.cap:
            self._show_frame(self.time_var.get())
            
    def _on_timeline_change(self, value):
        if self.cap:
            self._show_frame(float(value))
            
    def _set_time_from_slider(self, which: str):
        current_time = self.time_var.get()
        if which == 'start':
            self.start_time_var.set(f"{current_time:.2f}")
        else:
            self.end_time_var.set(f"{current_time:.2f}")
            
    def _seek_relative(self, delta: float):
        if self.cap is None:
            return
        new_time = max(0, min(self.time_var.get() + delta, self.duration))
        self.time_var.set(new_time)
        self._show_frame(new_time)
        
    def _seek_to(self, time: float):
        if self.cap is None:
            return
        self.time_var.set(time)
        self._show_frame(time)
        
    def _step_frame(self, delta: int):
        if self.cap is None:
            return
        current_frame = int(self.time_var.get() * self.fps)
        new_frame = max(0, min(current_frame + delta, self.total_frames - 1))
        new_time = new_frame / self.fps
        self.time_var.set(new_time)
        self._show_frame(new_time)
        
    def _toggle_preview(self):
        if self.cap is None:
            return
            
        if self.preview_running:
            self.preview_running = False
            self.play_btn.config(text="▶️ Play")
        else:
            self.preview_running = True
            self.play_btn.config(text="⏸️ Pause")
            threading.Thread(target=self._preview_loop, daemon=True).start()
            
    def _preview_loop(self):
        import time
        while self.preview_running:
            current_time = self.time_var.get()
            current_time += 1/self.fps
            
            if current_time >= self.duration:
                self.preview_running = False
                self.root.after(0, lambda: self.play_btn.config(text="▶️ Play"))
                break
                
            self.time_var.set(current_time)
            self.root.after(0, lambda t=current_time: self._show_frame(t))
            time.sleep(1/self.fps)
            
    def _update_blur_label(self, value):
        val = self.blur_var.get()
        if val % 2 == 0:
            val += 1
        self.blur_label.config(text=str(val))
        
    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
        
    def _export_video(self):
        if self.cap is None:
            messagebox.showerror("Error", "Please open a video first")
            return
            
        if not self.blur_regions:
            messagebox.showwarning("Warning", "No blur regions defined")
            return
            
        if self.is_processing:
            messagebox.showinfo("Info", "Export already in progress")
            return
            
        output_path = filedialog.asksaveasfilename(
            title="Save Blurred Video",
            defaultextension=".mp4",
            filetypes=[("MP4 Files", "*.mp4"), ("AVI Files", "*.avi")],
            initialfile=f"{Path(self.video_path).stem}_blurred.mp4"
        )
        
        if not output_path:
            return
            
        self.is_processing = True
        threading.Thread(target=self._export_thread, args=(output_path,), daemon=True).start()
        
    def _export_thread(self, output_path: str):
        try:
            cap = cv2.VideoCapture(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                  (self.video_width, self.video_height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = frame_count / self.fps
                frame = self._apply_blur_regions(frame, current_time, frame_count)
                out.write(frame)
                frame_count += 1
                
                progress = (frame_count / self.total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda f=frame_count:
                    self.progress_label.config(text=f"Processing: {f}/{self.total_frames}"))
                
            cap.release()
            out.release()
            
            self.root.after(0, lambda: self._export_complete(output_path))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Export failed: {str(e)}"))
        finally:
            self.is_processing = False
            
    def _export_complete(self, output_path: str):
        self.progress_var.set(100)
        self.progress_label.config(text="✅ Export complete!")
        self.status_label.config(text="✅ Video exported")
        messagebox.showinfo("Success", f"Video exported to:\n{output_path}")


def main():
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "Pillow", "--break-system-packages", "-q"])
    
    root = tk.Tk()
    app = SmartVideoBlurTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
