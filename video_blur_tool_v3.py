#!/usr/bin/env python3
"""
Video Blur Tool v3 - Ultimate Edition
Author: Red Coder
Features: Smart detection, object tracking, mouse-centric controls, quick presets
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
        
        frames = sorted(self.tracked_positions.keys())
        if not frames:
            return (self.x, self.y, self.width, self.height)
        
        if frame_num <= frames[0]:
            return self.tracked_positions[frames[0]]
        if frame_num >= frames[-1]:
            return self.tracked_positions[frames[-1]]
        
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


class UltimateVideoBlurTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Video Blur Tool v3 - Ultimate Edition")
        self.root.geometry("1350x900")
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
        self.profile_cascade = None
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
        
        # === v1 Mouse-centric state ===
        self.dragging_region: Optional[int] = None
        self.resize_handle: Optional[str] = None  # 'nw', 'ne', 'sw', 'se' for corners
        self.drag_start_pos: Optional[Tuple[int, int]] = None
        self.drag_start_region: Optional[dict] = None
        self.hovered_region: Optional[int] = None
        self.preset_size: Optional[Tuple[int, int]] = None
        self.quick_toolbar: Optional[tk.Toplevel] = None
        self.clicked_region_idx: Optional[int] = None
        
        # === v2 Detection state ===
        self.auto_track_var = None
        self.sensitivity_var = None
        
        # Blur presets
        self.blur_presets = {"Light": 21, "Medium": 51, "Heavy": 99, "Maximum": 151}
        
        # Size presets (from v1)
        self.size_presets = [
            ("👤 Face", 100, 120),
            ("🚗 Plate", 150, 40),
            ("📱 Phone", 80, 160),
            ("📝 Doc", 200, 280),
        ]
        
        self._setup_styles()
        self._create_ui()
        self._create_context_menus()
        
    def _load_detection_models(self):
        """Load face detection cascades"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.profile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_profileface.xml'
            )
        except Exception as e:
            print(f"Warning: Could not load face cascade: {e}")
            self.face_cascade = None
            
    def _setup_styles(self):
        """Configure custom styles - GitHub dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_dark = "#0d1117"
        bg_medium = "#161b22"
        bg_light = "#21262d"
        accent = "#58a6ff"
        accent_green = "#3fb950"
        accent_red = "#f85149"
        text_light = "#c9d1d9"
        
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=accent)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=text_light)
        
        style.configure("TLabelframe", background=bg_dark, foreground=text_light)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=accent, font=("Segoe UI", 10, "bold"))
        
        style.configure("TButton", background=bg_light, foreground=text_light,
            font=("Segoe UI", 10), padding=(12, 8))
        style.map("TButton", background=[("active", bg_medium)])
        
        style.configure("Accent.TButton", background=accent, foreground="#ffffff",
            font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#79c0ff")])
        
        style.configure("Success.TButton", background=accent_green, foreground="#ffffff",
            font=("Segoe UI", 10, "bold"))
        
        style.configure("Danger.TButton", background=accent_red, foreground="#ffffff")
        
        style.configure("TRadiobutton", background=bg_dark, foreground=text_light)
        style.configure("TCheckbutton", background=bg_dark, foreground=text_light)
        style.configure("TScale", background=bg_dark, troughcolor=bg_medium)
        
        style.configure("Treeview", background=bg_medium, foreground=text_light,
            fieldbackground=bg_medium, font=("Segoe UI", 9), rowheight=28)
        style.configure("Treeview.Heading", background=bg_light, foreground=text_light,
            font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", accent)])
        
    def _create_ui(self):
        """Create the main user interface"""
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
        
        # Mouse bindings - Combined from v1 and v2
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        
        # Timeline
        timeline_frame = ttk.Frame(left_panel)
        timeline_frame.pack(fill=tk.X, pady=10)
        
        self.time_var = tk.DoubleVar(value=0)
        self.timeline_slider = ttk.Scale(timeline_frame, from_=0, to=100,
                                         variable=self.time_var, orient=tk.HORIZONTAL,
                                         command=self._on_timeline_change)
        self.timeline_slider.pack(fill=tk.X, pady=(0, 5))
        
        time_info = ttk.Frame(timeline_frame)
        time_info.pack(fill=tk.X)
        self.time_label = ttk.Label(time_info, text="00:00.00 / 00:00.00")
        self.time_label.pack(side=tk.LEFT)
        self.frame_label = ttk.Label(time_info, text="Frame: 0 / 0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Playback controls
        playback_frame = ttk.Frame(left_panel)
        playback_frame.pack(fill=tk.X, pady=5)
        
        seek_btns = [("⏮️", -10), ("◀◀", -5), ("◀", -1), ("▶️ Play", "play"),
                     ("▶", 1), ("▶▶", 5), ("⏭️", 10)]
        
        for text, action in seek_btns:
            if action == "play":
                self.play_btn = ttk.Button(playback_frame, text=text, command=self._toggle_preview)
                self.play_btn.pack(side=tk.LEFT, padx=2)
            else:
                ttk.Button(playback_frame, text=text,
                          command=lambda a=action: self._seek_relative(a)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(playback_frame, text="← Frame", command=lambda: self._step_frame(-1)).pack(side=tk.RIGHT, padx=2)
        ttk.Button(playback_frame, text="Frame →", command=lambda: self._step_frame(1)).pack(side=tk.RIGHT, padx=2)
        
        # Right panel with scrollable controls
        self._create_right_panel(main_frame)
        
        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._step_frame(-1))
        self.root.bind("<Right>", lambda e: self._step_frame(1))
        self.root.bind("<space>", lambda e: self._toggle_preview())
        self.root.bind("<Home>", lambda e: self._seek_to(0))
        self.root.bind("<End>", lambda e: self._seek_to(self.duration))
        
    def _create_right_panel(self, main_frame):
        """Create the right panel with all controls"""
        right_panel = ttk.Frame(main_frame, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Scrollable frame
        canvas_scroll = tk.Canvas(right_panel, bg="#0d1117", highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = ttk.Frame(canvas_scroll)
        
        scrollable_frame.bind("<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw", width=380)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.LabelFrame(scrollable_frame, text="📁 Video File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=340)
        self.file_label.pack(fill=tk.X)
        ttk.Button(file_frame, text="📂 Open Video", style="Accent.TButton",
                   command=self._open_video).pack(fill=tk.X, pady=(10, 0))
        
        # === SMART DETECTION (from v2) ===
        detect_frame = ttk.LabelFrame(scrollable_frame, text="🤖 Smart Detection", padding=10)
        detect_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        ttk.Button(detect_frame, text="👤 Detect Faces", style="Success.TButton",
                   command=self._auto_detect_faces).pack(fill=tk.X, pady=2)
        ttk.Button(detect_frame, text="👥 Scan All Faces in Video",
                   command=self._scan_all_faces).pack(fill=tk.X, pady=2)
        ttk.Button(detect_frame, text="🚗 Detect License Plates",
                   command=self._detect_license_plates).pack(fill=tk.X, pady=2)
        
        self.auto_track_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(detect_frame, text="🎯 Auto-track detected objects",
                        variable=self.auto_track_var).pack(anchor=tk.W, pady=5)
        
        sens_frame = ttk.Frame(detect_frame)
        sens_frame.pack(fill=tk.X, pady=5)
        ttk.Label(sens_frame, text="Sensitivity:").pack(side=tk.LEFT)
        self.sensitivity_var = tk.DoubleVar(value=1.2)
        ttk.Scale(sens_frame, from_=1.05, to=1.5, variable=self.sensitivity_var,
                  orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # === QUICK PRESETS (from v1) ===
        presets_frame = ttk.LabelFrame(scrollable_frame, text="🎯 Quick Size Presets", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        presets_row = ttk.Frame(presets_frame)
        presets_row.pack(fill=tk.X)
        for name, w, h in self.size_presets:
            ttk.Button(presets_row, text=name,
                       command=lambda ww=w, hh=h: self._set_preset_mode(ww, hh)).pack(side=tk.LEFT, padx=2)
        
        # === TIMING CONTROLS ===
        timing_frame = ttk.LabelFrame(scrollable_frame, text="⏰ Region Settings", padding=10)
        timing_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # Start time row
        start_row = ttk.Frame(timing_frame)
        start_row.pack(fill=tk.X, pady=2)
        ttk.Label(start_row, text="Start (s):").pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value="0.0")
        ttk.Entry(start_row, textvariable=self.start_time_var, width=10).pack(side=tk.RIGHT)
        ttk.Button(start_row, text="⏺️ MARK IN",
                   command=lambda: self._set_time_from_slider('start')).pack(side=tk.RIGHT, padx=5)
        
        # End time row
        end_row = ttk.Frame(timing_frame)
        end_row.pack(fill=tk.X, pady=2)
        ttk.Label(end_row, text="End (s):").pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value="0.0")
        ttk.Entry(end_row, textvariable=self.end_time_var, width=10).pack(side=tk.RIGHT)
        ttk.Button(end_row, text="⏺️ MARK OUT",
                   command=lambda: self._set_time_from_slider('end')).pack(side=tk.RIGHT, padx=5)
        
        # Blur strength with presets
        blur_row = ttk.Frame(timing_frame)
        blur_row.pack(fill=tk.X, pady=5)
        ttk.Label(blur_row, text="Blur:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=51)
        self.blur_scale = ttk.Scale(blur_row, from_=5, to=151, variable=self.blur_var,
                                    orient=tk.HORIZONTAL, command=self._update_blur_label)
        self.blur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.blur_label = ttk.Label(blur_row, text="51", width=4)
        self.blur_label.pack(side=tk.RIGHT)
        
        # Blur presets
        preset_row = ttk.Frame(timing_frame)
        preset_row.pack(fill=tk.X, pady=2)
        self.preset_var = tk.StringVar(value="Medium")
        for name in self.blur_presets:
            ttk.Radiobutton(preset_row, text=name, variable=self.preset_var,
                           value=name, command=self._apply_blur_preset).pack(side=tk.LEFT, padx=3)
        
        # === REGIONS LIST ===
        regions_frame = ttk.LabelFrame(scrollable_frame, text="📋 Blur Regions", padding=10)
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
        ttk.Button(region_btns, text="🗑️ Delete", command=self._delete_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="✏️ Update", command=self._update_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="🔄 Re-track", command=self._retrack_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="🧹 Clear", command=self._clear_all_regions).pack(side=tk.RIGHT, padx=2)
        
        # === EXPORT ===
        export_frame = ttk.LabelFrame(scrollable_frame, text="💾 Export", padding=10)
        export_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(export_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(export_frame, text="Ready")
        self.progress_label.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="🚀 Export Blurred Video", style="Accent.TButton",
                   command=self._export_video).pack(fill=tk.X, pady=(10, 0))
        
        # Tips
        tips = ttk.Label(scrollable_frame,
            text="💡 Tips: Right-click for menu • Scroll wheel to adjust blur • Drag regions to move",
            wraplength=360, justify=tk.LEFT)
        tips.pack(fill=tk.X, pady=10, padx=5)

    # ==================== CONTEXT MENUS (from v1) ====================
    
    def _create_context_menus(self):
        """Create right-click context menus"""
        # Canvas context menu
        self.canvas_menu = tk.Menu(self.root, tearoff=0, bg="#21262d", fg="#c9d1d9")
        self.canvas_menu.add_command(label="📍 Set Start Time Here", command=self._set_start_from_current)
        self.canvas_menu.add_command(label="📍 Set End Time Here", command=self._set_end_from_current)
        self.canvas_menu.add_separator()
        self.canvas_menu.add_command(label="🧹 Clear All Regions", command=self._clear_all_regions)
        
        # Region context menu
        self.region_menu = tk.Menu(self.root, tearoff=0, bg="#21262d", fg="#c9d1d9")
        self.region_menu.add_command(label="📋 Duplicate Region", command=self._duplicate_clicked_region)
        self.region_menu.add_command(label="🎬 Apply to Whole Video", command=self._apply_whole_video)
        self.region_menu.add_command(label="▶️ Apply From Here", command=self._apply_from_here)
        self.region_menu.add_command(label="⏹️ Apply To Here", command=self._apply_to_here)
        self.region_menu.add_separator()
        self.region_menu.add_command(label="🔄 Re-track This Region", command=self._retrack_clicked_region)
        self.region_menu.add_separator()
        self.region_menu.add_command(label="🗑️ Delete Region", command=self._delete_clicked_region)

    def _on_right_click(self, event):
        """Handle right-click on canvas"""
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            self.clicked_region_idx = region_idx
            self.region_menu.post(event.x_root, event.y_root)
        else:
            self.canvas_menu.post(event.x_root, event.y_root)

    def _get_region_at(self, canvas_x, canvas_y) -> Optional[int]:
        """Get the index of the region at canvas coordinates"""
        if self.cap is None:
            return None
        
        current_frame = int(self.time_var.get() * self.fps)
        
        for i, region in enumerate(self.blur_regions):
            if region.tracked_positions:
                x, y, w, h = region.get_position_at_frame(current_frame)
            else:
                x, y, w, h = region.x, region.y, region.width, region.height
            
            # Convert region to canvas coordinates (v1 approach)
            x1 = int(x * self.scale_factor) + self.canvas_offset_x
            y1 = int(y * self.scale_factor) + self.canvas_offset_y
            x2 = int((x + w) * self.scale_factor) + self.canvas_offset_x
            y2 = int((y + h) * self.scale_factor) + self.canvas_offset_y
            
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return i
        return None

    def _get_resize_handle(self, canvas_x, canvas_y, region_idx) -> Optional[str]:
        """Check if mouse is on a resize handle (corners) of the region"""
        if self.cap is None or self.scale_factor == 0 or region_idx is None:
            return None
        
        region = self.blur_regions[region_idx]
        current_frame = int(self.time_var.get() * self.fps)
        
        if region.tracked_positions:
            x, y, w, h = region.get_position_at_frame(current_frame)
        else:
            x, y, w, h = region.x, region.y, region.width, region.height
        
        # Convert to canvas coordinates
        cx1 = int(x * self.scale_factor) + self.canvas_offset_x
        cy1 = int(y * self.scale_factor) + self.canvas_offset_y
        cx2 = int((x + w) * self.scale_factor) + self.canvas_offset_x
        cy2 = int((y + h) * self.scale_factor) + self.canvas_offset_y
        
        handle_size = 12  # pixels
        
        # Check each corner
        if abs(canvas_x - cx1) < handle_size and abs(canvas_y - cy1) < handle_size:
            return 'nw'
        if abs(canvas_x - cx2) < handle_size and abs(canvas_y - cy1) < handle_size:
            return 'ne'
        if abs(canvas_x - cx1) < handle_size and abs(canvas_y - cy2) < handle_size:
            return 'sw'
        if abs(canvas_x - cx2) < handle_size and abs(canvas_y - cy2) < handle_size:
            return 'se'
        
        return None

    # ==================== VIDEO OPERATIONS ====================
    
    def _open_video(self):
        """Open a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"), ("All Files", "*.*")]
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
            
        frame = self._apply_blur_regions(frame, time_seconds, frame_number)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 800, 500
            
        scale_x = canvas_width / self.video_width
        scale_y = canvas_height / self.video_height
        self.scale_factor = min(scale_x, scale_y)
        
        new_width = int(self.video_width * self.scale_factor)
        new_height = int(self.video_height * self.scale_factor)
        
        self.canvas_offset_x = (canvas_width - new_width) // 2
        self.canvas_offset_y = (canvas_height - new_height) // 2
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        from PIL import Image, ImageTk
        image = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image)
        
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y,
                                 anchor=tk.NW, image=self.photo)
        
        self._draw_blur_regions(time_seconds, frame_number)
        
        self.time_label.config(text=f"{self._format_time(time_seconds)} / {self._format_time(self.duration)}")
        self.frame_label.config(text=f"Frame: {frame_number} / {self.total_frames}")

    def _apply_blur_regions(self, frame: np.ndarray, current_time: float, frame_number: int) -> np.ndarray:
        """Apply blur to frame based on active regions"""
        result = frame.copy()
        
        for region in self.blur_regions:
            if region.contains_frame(current_time):
                if region.tracked_positions:
                    x, y, w, h = region.get_position_at_frame(frame_number)
                else:
                    x, y, w, h = region.x, region.y, region.width, region.height
                    
                x1, y1 = max(0, x), max(0, y)
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
        """Draw blur region rectangles on canvas with resize handles"""
        for i, region in enumerate(self.blur_regions):
            if region.tracked_positions:
                x, y, w, h = region.get_position_at_frame(frame_number)
            else:
                x, y, w, h = region.x, region.y, region.width, region.height
                
            x1 = int(x * self.scale_factor) + self.canvas_offset_x
            y1 = int(y * self.scale_factor) + self.canvas_offset_y
            x2 = int((x + w) * self.scale_factor) + self.canvas_offset_x
            y2 = int((y + h) * self.scale_factor) + self.canvas_offset_y
            
            if region.contains_frame(current_time):
                color = "#3fb950"
            else:
                color = "#6e7681"
                
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            
            # Draw resize handles at corners
            handle_size = 6
            for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                self.canvas.create_rectangle(
                    hx - handle_size, hy - handle_size,
                    hx + handle_size, hy + handle_size,
                    fill=color, outline="white", width=1)
            
            icon = {"face": "👤", "plate": "🚗", "track": "🎯", "manual": "🔲"}
            label = f"#{i+1} {icon.get(region.mode.value, '🔲')}"
            tracked = "📍" if region.tracked_positions else ""
            self.canvas.create_text(x1 + 5, y1 + 15, text=f"{label}{tracked}",
                                   fill=color, anchor=tk.NW, font=("Segoe UI", 9, "bold"))

    # ==================== MOUSE EVENT HANDLERS (from v1) ====================
    
    def _on_mouse_down(self, event):
        """Handle mouse press for region selection, dragging, or resizing"""
        if self.cap is None:
            return
        
        self._hide_quick_toolbar()
        
        # Preset mode - create region at click
        if self.preset_size:
            w, h = self.preset_size
            video_x = int((event.x - self.canvas_offset_x) / self.scale_factor) - w // 2
            video_y = int((event.y - self.canvas_offset_y) / self.scale_factor) - h // 2
            video_x = max(0, min(video_x, self.video_width - w))
            video_y = max(0, min(video_y, self.video_height - h))
            
            try:
                start_time = float(self.start_time_var.get())
                end_time = float(self.end_time_var.get())
            except ValueError:
                start_time, end_time = 0.0, self.duration
            
            blur_strength = self.blur_var.get()
            if blur_strength % 2 == 0:
                blur_strength += 1
            
            region = BlurRegion(x=video_x, y=video_y, width=w, height=h,
                start_time=start_time, end_time=end_time, blur_strength=blur_strength)
            self.blur_regions.append(region)
            self._update_regions_list()
            self._show_frame(self.time_var.get())
            self._show_quick_toolbar(event.x_root, event.y_root, len(self.blur_regions) - 1)
            
            self.preset_size = None
            self.canvas.config(cursor="crosshair")
            return
        
        # Check if clicking on existing region
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            region = self.blur_regions[region_idx]
            
            # Check for resize handle first
            handle = self._get_resize_handle(event.x, event.y, region_idx)
            if handle:
                self.resize_handle = handle
                self.dragging_region = region_idx
                self.drag_start_pos = (event.x, event.y)
                self.drag_start_region = {
                    'x': region.x, 'y': region.y,
                    'width': region.width, 'height': region.height
                }
                return
            
            # Otherwise start dragging (moving)
            self.dragging_region = region_idx
            self.resize_handle = None
            self.drag_start_pos = (event.x, event.y)
            self.drag_start_region = {
                'x': region.x, 'y': region.y,
                'width': region.width, 'height': region.height
            }
            return
            
        # Start new selection
        self.is_selecting = True
        self.selection_start = (event.x, event.y)

    def _on_mouse_drag(self, event):
        """Handle mouse drag for selection, region dragging, or resizing"""
        if self.dragging_region is not None and self.drag_start_pos and self.drag_start_region:
            region = self.blur_regions[self.dragging_region]
            
            if self.resize_handle:
                # Resize mode
                dx = int((event.x - self.drag_start_pos[0]) / self.scale_factor)
                dy = int((event.y - self.drag_start_pos[1]) / self.scale_factor)
                
                orig_x = self.drag_start_region['x']
                orig_y = self.drag_start_region['y']
                orig_w = self.drag_start_region['width']
                orig_h = self.drag_start_region['height']
                
                min_size = 20
                
                if self.resize_handle == 'se':
                    region.width = max(min_size, orig_w + dx)
                    region.height = max(min_size, orig_h + dy)
                elif self.resize_handle == 'sw':
                    new_w = max(min_size, orig_w - dx)
                    region.x = orig_x + orig_w - new_w
                    region.width = new_w
                    region.height = max(min_size, orig_h + dy)
                elif self.resize_handle == 'ne':
                    region.width = max(min_size, orig_w + dx)
                    new_h = max(min_size, orig_h - dy)
                    region.y = orig_y + orig_h - new_h
                    region.height = new_h
                elif self.resize_handle == 'nw':
                    new_w = max(min_size, orig_w - dx)
                    new_h = max(min_size, orig_h - dy)
                    region.x = orig_x + orig_w - new_w
                    region.y = orig_y + orig_h - new_h
                    region.width = new_w
                    region.height = new_h
                
                # Clamp to video bounds
                region.x = max(0, min(region.x, self.video_width - region.width))
                region.y = max(0, min(region.y, self.video_height - region.height))
                region.width = min(region.width, self.video_width - region.x)
                region.height = min(region.height, self.video_height - region.y)
            else:
                # Move mode
                dx = int((event.x - self.drag_start_pos[0]) / self.scale_factor)
                dy = int((event.y - self.drag_start_pos[1]) / self.scale_factor)
                
                new_x = max(0, min(self.drag_start_region['x'] + dx, self.video_width - region.width))
                new_y = max(0, min(self.drag_start_region['y'] + dy, self.video_height - region.height))
                
                region.x, region.y = new_x, new_y
            
            self._show_frame(self.time_var.get())
            return
        
        if not self.is_selecting or self.selection_start is None:
            return
            
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y
        
        self.selection_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#58a6ff", width=2, dash=(5, 5))
        self.temp_rect = (x1, y1, x2, y2)

    def _on_mouse_up(self, event):
        """Handle mouse release to finalize region selection, dragging, or resizing"""
        if self.dragging_region is not None:
            self.dragging_region = None
            self.resize_handle = None
            self.drag_start_pos = None
            self.drag_start_region = None
            self._update_regions_list()
            return
        
        if not self.is_selecting or self.selection_start is None:
            return
            
        self.is_selecting = False
        
        if self.temp_rect is None:
            return
            
        x1, y1, x2, y2 = self.temp_rect
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        
        video_x1 = int((x1 - self.canvas_offset_x) / self.scale_factor)
        video_y1 = int((y1 - self.canvas_offset_y) / self.scale_factor)
        video_x2 = int((x2 - self.canvas_offset_x) / self.scale_factor)
        video_y2 = int((y2 - self.canvas_offset_y) / self.scale_factor)
        
        video_x1 = max(0, min(video_x1, self.video_width))
        video_y1 = max(0, min(video_y1, self.video_height))
        video_x2 = max(0, min(video_x2, self.video_width))
        video_y2 = max(0, min(video_y2, self.video_height))
        
        width, height = video_x2 - video_x1, video_y2 - video_y1
        
        if width < 10 or height < 10:
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            self.temp_rect = None
            return
            
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time, end_time = 0.0, self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        region = BlurRegion(x=video_x1, y=video_y1, width=width, height=height,
            start_time=start_time, end_time=end_time, blur_strength=blur_strength)
        
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
        self._show_quick_toolbar(event.x_root, event.y_root, len(self.blur_regions) - 1)

    def _on_mouse_motion(self, event):
        """Update cursor based on hover state"""
        if self.cap is None:
            return
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            self.hovered_region = region_idx
            # Check for resize handles
            handle = self._get_resize_handle(event.x, event.y, region_idx)
            if handle in ('nw', 'se'):
                self.canvas.config(cursor="size_nw_se")
            elif handle in ('ne', 'sw'):
                self.canvas.config(cursor="size_ne_sw")
            else:
                self.canvas.config(cursor="fleur")  # Move cursor
        elif self.preset_size:
            self.canvas.config(cursor="target")
        else:
            self.canvas.config(cursor="crosshair")
            self.hovered_region = None

    def _on_canvas_scroll(self, event):
        """Handle scroll wheel for timeline or blur adjustment"""
        if self.cap is None:
            return
        
        delta = 1 if event.delta > 0 else -1
        
        if self.hovered_region is not None:
            region = self.blur_regions[self.hovered_region]
            new_blur = region.blur_strength + delta * 10
            region.blur_strength = max(5, min(151, new_blur))
            if region.blur_strength % 2 == 0:
                region.blur_strength += 1
            self._update_regions_list()
            self._show_frame(self.time_var.get())
        else:
            if event.state & 0x1:  # Shift key
                self._seek_relative(delta)
            else:
                self._step_frame(delta)

    # ==================== QUICK TOOLBAR (from v1) ====================
    
    def _show_quick_toolbar(self, x, y, region_idx):
        """Show quick action toolbar near the created region"""
        self._hide_quick_toolbar()
        
        self.clicked_region_idx = region_idx
        self.quick_toolbar = tk.Toplevel(self.root)
        self.quick_toolbar.overrideredirect(True)
        self.quick_toolbar.attributes('-topmost', True)
        self.quick_toolbar.configure(bg="#21262d")
        self.quick_toolbar.geometry(f"+{x+10}+{y+10}")
        
        frame = tk.Frame(self.quick_toolbar, bg="#21262d", padx=5, pady=5)
        frame.pack()
        
        btns = [("🎬 Whole", self._apply_whole_video), ("▶️ From", self._apply_from_here),
                ("⏹️ To", self._apply_to_here), ("🗑️", self._delete_clicked_region)]
        
        for text, cmd in btns:
            tk.Button(frame, text=text, command=cmd, bg="#30363d", fg="#c9d1d9",
                      relief=tk.FLAT, padx=8, pady=4).pack(side=tk.LEFT, padx=2)
        
        self.root.after(5000, self._hide_quick_toolbar)

    def _hide_quick_toolbar(self):
        """Hide the quick toolbar if visible"""
        if self.quick_toolbar:
            try:
                self.quick_toolbar.destroy()
            except:
                pass
            self.quick_toolbar = None

    # ==================== CONTEXT MENU ACTIONS ====================
    
    def _set_start_from_current(self):
        self.start_time_var.set(f"{self.time_var.get():.2f}")

    def _set_end_from_current(self):
        self.end_time_var.set(f"{self.time_var.get():.2f}")

    def _duplicate_clicked_region(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            orig = self.blur_regions[self.clicked_region_idx]
            new_region = BlurRegion(x=orig.x + 20, y=orig.y + 20, width=orig.width, height=orig.height,
                start_time=orig.start_time, end_time=orig.end_time, blur_strength=orig.blur_strength,
                mode=orig.mode)
            self.blur_regions.append(new_region)
            self._update_regions_list()
            self._show_frame(self.time_var.get())

    def _apply_whole_video(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].start_time = 0.0
            self.blur_regions[self.clicked_region_idx].end_time = self.duration
            self._update_regions_list()
            self._show_frame(self.time_var.get())
        self._hide_quick_toolbar()

    def _apply_from_here(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].start_time = self.time_var.get()
            self._update_regions_list()
            self._show_frame(self.time_var.get())
        self._hide_quick_toolbar()

    def _apply_to_here(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].end_time = self.time_var.get()
            self._update_regions_list()
            self._show_frame(self.time_var.get())
        self._hide_quick_toolbar()

    def _delete_clicked_region(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            del self.blur_regions[self.clicked_region_idx]
            self._update_regions_list()
            self._show_frame(self.time_var.get())
        self._hide_quick_toolbar()

    def _retrack_clicked_region(self):
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self._retrack_region_by_idx(self.clicked_region_idx)

    # ==================== SMART DETECTION (from v2) ====================
    
    def _auto_detect_faces(self):
        """Detect faces in current frame"""
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = self.sensitivity_var.get()
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
        if self.profile_cascade:
            profiles = self.profile_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
            all_faces = list(faces) + list(profiles)
        else:
            all_faces = list(faces)
        
        if len(all_faces) == 0:
            self.status_label.config(text="❌ No faces detected")
            messagebox.showinfo("Info", "No faces detected in current frame.")
            return
            
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time, end_time = self.time_var.get(), self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        current_frame = int(self.time_var.get() * self.fps)
        
        for (x, y, w, h) in all_faces:
            padding = int(w * 0.2)
            x, y = max(0, x - padding), max(0, y - padding)
            w = min(self.video_width - x, w + 2 * padding)
            h = min(self.video_height - y, h + 2 * padding)
            
            region = BlurRegion(x=x, y=y, width=w, height=h,
                start_time=start_time, end_time=end_time,
                blur_strength=blur_strength, mode=BlurMode.FACE)
            
            if self.auto_track_var.get():
                region.tracked_positions[current_frame] = (x, y, w, h)
                self._track_region_forward(region, frame, current_frame)
            
            self.blur_regions.append(region)
            
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.status_label.config(text=f"✅ {len(all_faces)} face(s) detected")

    def _scan_all_faces(self):
        """Scan entire video for faces"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
        if not messagebox.askyesno("Scan Video", "This will scan the entire video for faces.\nThis may take a while. Continue?"):
            return
        self.is_processing = True
        threading.Thread(target=self._scan_faces_thread, daemon=True).start()

    def _scan_faces_thread(self):
        """Background thread for face scanning"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, int(self.fps / 2))
            detected = []
            scale = self.sensitivity_var.get()
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % interval == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        detected.append((frame_num, x, y, w, h))
                    progress = (frame_num / total) * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                frame_num += 1
            cap.release()
            self.root.after(0, lambda: self._process_detected_faces(detected))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
        finally:
            self.is_processing = False

    def _process_detected_faces(self, detected):
        """Process detected faces into regions"""
        if not detected:
            self.status_label.config(text="❌ No faces found")
            return
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        # Simple clustering
        used = set()
        for i, (f1, x1, y1, w1, h1) in enumerate(detected):
            if i in used:
                continue
            group = [(f1, x1, y1, w1, h1)]
            used.add(i)
            for j, (f2, x2, y2, w2, h2) in enumerate(detected):
                if j in used:
                    continue
                if abs(x1 - x2) < w1 and abs(y1 - y2) < h1:
                    group.append((f2, x2, y2, w2, h2))
                    used.add(j)
            
            frames = [g[0] for g in group]
            region = BlurRegion(
                x=int(np.mean([g[1] for g in group])),
                y=int(np.mean([g[2] for g in group])),
                width=int(np.mean([g[3] for g in group])),
                height=int(np.mean([g[4] for g in group])),
                start_time=min(frames) / self.fps,
                end_time=max(frames) / self.fps,
                blur_strength=blur_strength, mode=BlurMode.FACE)
            for (f, x, y, w, h) in group:
                region.tracked_positions[f] = (x, y, w, h)
            self.blur_regions.append(region)
        
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.progress_var.set(100)
        self.status_label.config(text=f"✅ Face regions created")

    def _detect_license_plates(self):
        """Detect license plate regions"""
        if self.cap is None:
            messagebox.showwarning("Warning", "Please open a video first")
            return
        frame = self._get_current_frame()
        if frame is None:
            return
        
        self.status_label.config(text="🔍 Detecting plates...")
        self.root.update()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        plates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = w / h if h > 0 else 0
            if 2 < ratio < 5 and w > 60 and h > 20:
                plates.append((x, y, w, h))
        
        if not plates:
            self.status_label.config(text="❌ No plates detected")
            return
            
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time, end_time = self.time_var.get(), self.duration
        
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
        
        for (x, y, w, h) in plates[:5]:
            region = BlurRegion(x=x, y=y, width=w, height=h,
                start_time=start_time, end_time=end_time,
                blur_strength=blur_strength, mode=BlurMode.LICENSE_PLATE)
            self.blur_regions.append(region)
        
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.status_label.config(text=f"✅ {len(plates[:5])} plate(s) detected")

    def _track_region_forward(self, region: BlurRegion, initial_frame: np.ndarray, start_frame: int):
        """Track a region forward using CSRT tracker"""
        if not self.auto_track_var.get():
            return
        try:
            tracker = cv2.TrackerCSRT_create()
        except:
            try:
                tracker = cv2.legacy.TrackerCSRT_create()
            except:
                return
        
        bbox = (region.x, region.y, region.width, region.height)
        tracker.init(initial_frame, bbox)
        
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        end_frame = int(region.end_time * self.fps)
        frame_num = start_frame
        
        while frame_num < end_frame and frame_num - start_frame < 300:
            ret, frame = cap.read()
            if not ret:
                break
            success, bbox = tracker.update(frame)
            if success:
                region.tracked_positions[frame_num] = tuple(int(v) for v in bbox)
            else:
                break
            frame_num += 1
        cap.release()

    # ==================== REGION MANAGEMENT ====================
    
    def _update_regions_list(self):
        """Update the regions treeview"""
        for item in self.regions_tree.get_children():
            self.regions_tree.delete(item)
        
        mode_names = {BlurMode.MANUAL: "Manual", BlurMode.FACE: "Face",
                      BlurMode.OBJECT_TRACK: "Tracked", BlurMode.LICENSE_PLATE: "Plate"}
        
        for i, region in enumerate(self.blur_regions):
            tracked = "📍" if region.tracked_positions else ""
            self.regions_tree.insert("", tk.END, values=(
                i + 1,
                f"{mode_names.get(region.mode, 'Manual')}{tracked}",
                f"{region.start_time:.1f}s → {region.end_time:.1f}s",
                region.blur_strength))

    def _on_region_select(self, event):
        selection = self.regions_tree.selection()
        if selection:
            idx = self.regions_tree.index(selection[0])
            self.current_region_id = idx
            region = self.blur_regions[idx]
            self.start_time_var.set(f"{region.start_time:.2f}")
            self.end_time_var.set(f"{region.end_time:.2f}")
            self.blur_var.set(region.blur_strength)
            self._update_blur_label(None)

    def _delete_selected_region(self):
        selection = self.regions_tree.selection()
        if not selection:
            return
        idx = self.regions_tree.index(selection[0])
        del self.blur_regions[idx]
        self._update_regions_list()
        self._show_frame(self.time_var.get())

    def _update_selected_region(self):
        selection = self.regions_tree.selection()
        if not selection:
            return
        idx = self.regions_tree.index(selection[0])
        try:
            self.blur_regions[idx].start_time = float(self.start_time_var.get())
            self.blur_regions[idx].end_time = float(self.end_time_var.get())
            self.blur_regions[idx].blur_strength = self.blur_var.get()
        except ValueError:
            return
        self._update_regions_list()
        self._show_frame(self.time_var.get())

    def _retrack_region(self):
        selection = self.regions_tree.selection()
        if not selection:
            return
        idx = self.regions_tree.index(selection[0])
        self._retrack_region_by_idx(idx)

    def _retrack_region_by_idx(self, idx):
        frame = self._get_current_frame()
        if frame is None:
            return
        region = self.blur_regions[idx]
        current_frame = int(self.time_var.get() * self.fps)
        region.tracked_positions.clear()
        region.tracked_positions[current_frame] = (region.x, region.y, region.width, region.height)
        self._track_region_forward(region, frame, current_frame)
        self._update_regions_list()
        self._show_frame(self.time_var.get())
        self.status_label.config(text="✅ Region re-tracked")

    def _clear_all_regions(self):
        self.blur_regions.clear()
        self._update_regions_list()
        if self.cap:
            self._show_frame(self.time_var.get())

    # ==================== TIMELINE AND PLAYBACK ====================
    
    def _on_timeline_change(self, value):
        if self.cap:
            self._show_frame(float(value))

    def _set_time_from_slider(self, which: str):
        current = self.time_var.get()
        if which == 'start':
            self.start_time_var.set(f"{current:.2f}")
        else:
            self.end_time_var.set(f"{current:.2f}")

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
        current = int(self.time_var.get() * self.fps)
        new_frame = max(0, min(current + delta, self.total_frames - 1))
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
            current = self.time_var.get() + 1/self.fps
            if current >= self.duration:
                self.preview_running = False
                self.root.after(0, lambda: self.play_btn.config(text="▶️ Play"))
                break
            self.time_var.set(current)
            self.root.after(0, lambda t=current: self._show_frame(t))
            time.sleep(1/self.fps)

    def _update_blur_label(self, value):
        val = self.blur_var.get()
        if val % 2 == 0:
            val += 1
        self.blur_label.config(text=str(val))

    def _apply_blur_preset(self):
        preset = self.preset_var.get()
        if preset in self.blur_presets:
            self.blur_var.set(self.blur_presets[preset])
            self._update_blur_label(None)

    def _set_preset_mode(self, w, h):
        self.preset_size = (w, h)
        self.canvas.config(cursor="target")
        self.status_label.config(text=f"🎯 Click to place {w}x{h} region")

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"

    # ==================== EXPORT ====================
    
    def _export_video(self):
        if self.cap is None or not self.blur_regions:
            messagebox.showwarning("Warning", "No video or regions to export")
            return
        if self.is_processing:
            return
        output = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
            initialfile=f"{Path(self.video_path).stem}_blurred.mp4")
        if not output:
            return
        self.is_processing = True
        threading.Thread(target=self._export_thread, args=(output,), daemon=True).start()

    def _export_thread(self, output_path):
        try:
            cap = cv2.VideoCapture(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.video_width, self.video_height))
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_time = frame_num / self.fps
                frame = self._apply_blur_regions(frame, current_time, frame_num)
                out.write(frame)
                frame_num += 1
                progress = (frame_num / self.total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda f=frame_num: self.progress_label.config(text=f"Processing: {f}/{self.total_frames}"))
            
            cap.release()
            out.release()
            self.root.after(0, lambda: self._export_complete(output_path))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Export failed: {e}"))
        finally:
            self.is_processing = False

    def _export_complete(self, path):
        self.progress_var.set(100)
        self.progress_label.config(text="✅ Export complete!")
        self.status_label.config(text="✅ Video exported")
        messagebox.showinfo("Success", f"Video exported to:\n{path}")


def main():
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "Pillow", "-q"])
    
    root = tk.Tk()
    app = UltimateVideoBlurTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
