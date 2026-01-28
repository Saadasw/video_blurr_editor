#!/usr/bin/env python3
"""
Video Blur Tool - A professional video region blur application
Author: Red Coder
Description: Blur specific regions of a video for custom durations with an intuitive GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import threading
import os
from pathlib import Path


@dataclass
class BlurRegion:
    """Represents a blur region with timing information"""
    x: int
    y: int
    width: int
    height: int
    start_time: float  # in seconds
    end_time: float    # in seconds
    blur_strength: int = 51  # must be odd number
    
    def contains_frame(self, current_time: float) -> bool:
        """Check if current time falls within this region's active period"""
        return self.start_time <= current_time <= self.end_time
    
    def to_dict(self) -> dict:
        return {
            'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height,
            'start_time': self.start_time, 'end_time': self.end_time,
            'blur_strength': self.blur_strength
        }


class VideoBlurTool:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🎬 Video Blur Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a2e")
        
        # Video properties
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.video_width: int = 0
        self.video_height: int = 0
        self.duration: float = 0.0
        
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
        
        # Drag and resize state
        self.dragging_region: Optional[int] = None
        self.resize_handle: Optional[str] = None
        self.drag_start_pos: Optional[Tuple[int, int]] = None
        self.drag_start_region: Optional[dict] = None
        self.hovered_region: Optional[int] = None
        
        # Preset mode
        self.preset_size: Optional[Tuple[int, int]] = None
        
        # Quick toolbar
        self.quick_toolbar: Optional[tk.Toplevel] = None
        self.clicked_region_idx: Optional[int] = None
        
        self._setup_styles()
        self._create_ui()
        self._create_context_menus()
        
    def _setup_styles(self):
        """Configure custom styles for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_dark = "#1a1a2e"
        bg_medium = "#16213e"
        accent = "#e94560"
        accent_hover = "#ff6b6b"
        text_light = "#eaeaea"
        
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=text_light, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=accent)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground=text_light)
        
        style.configure("TButton", 
            background=accent, foreground="white", 
            font=("Segoe UI", 10, "bold"), padding=(15, 8))
        style.map("TButton",
            background=[("active", accent_hover), ("pressed", "#c73e54")])
        
        style.configure("Secondary.TButton",
            background=bg_medium, foreground=text_light,
            font=("Segoe UI", 10), padding=(12, 6))
        style.map("Secondary.TButton",
            background=[("active", "#1f3460")])
        
        style.configure("TScale", background=bg_dark, troughcolor=bg_medium)
        style.configure("TEntry", fieldbackground=bg_medium, foreground=text_light)
        
        style.configure("Treeview",
            background=bg_medium, foreground=text_light,
            fieldbackground=bg_medium, font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
            background=accent, foreground="white",
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
        
        # Video title
        ttk.Label(left_panel, text="📹 Video Preview", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # Canvas for video display
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="#0f0f23", highlightthickness=2, 
                                highlightbackground="#e94560", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for region selection
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_canvas_scroll)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        
        # Timeline slider
        timeline_frame = ttk.Frame(left_panel)
        timeline_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(timeline_frame, text="⏱️ Timeline:", style="Header.TLabel").pack(side=tk.LEFT)
        
        self.time_var = tk.DoubleVar(value=0)
        self.timeline_slider = ttk.Scale(timeline_frame, from_=0, to=100, 
                                         variable=self.time_var, orient=tk.HORIZONTAL,
                                         command=self._on_timeline_change)
        self.timeline_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.time_label = ttk.Label(timeline_frame, text="00:00.00 / 00:00.00")
        self.time_label.pack(side=tk.RIGHT)
        
        # Playback controls
        playback_frame = ttk.Frame(left_panel)
        playback_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(playback_frame, text="⏮️ -5s", style="Secondary.TButton",
                   command=lambda: self._seek_relative(-5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="⏪ -1s", style="Secondary.TButton",
                   command=lambda: self._seek_relative(-1)).pack(side=tk.LEFT, padx=2)
        
        self.play_btn = ttk.Button(playback_frame, text="▶️ Play", style="Secondary.TButton",
                                   command=self._toggle_preview)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(playback_frame, text="⏩ +1s", style="Secondary.TButton",
                   command=lambda: self._seek_relative(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(playback_frame, text="⏭️ +5s", style="Secondary.TButton",
                   command=lambda: self._seek_relative(5)).pack(side=tk.LEFT, padx=2)
        
        # Right panel - Controls
        right_panel = ttk.Frame(main_frame, width=350)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # File selection
        file_frame = ttk.LabelFrame(right_panel, text="📁 Video File", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.file_label = ttk.Label(file_frame, text="No file selected", wraplength=300)
        self.file_label.pack(fill=tk.X)
        
        ttk.Button(file_frame, text="📂 Open Video", command=self._open_video).pack(fill=tk.X, pady=(10, 0))
        
        # Region timing controls
        timing_frame = ttk.LabelFrame(right_panel, text="⏰ Region Timing", padding=10)
        timing_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start time with Mark In button
        start_row = ttk.Frame(timing_frame)
        start_row.pack(fill=tk.X, pady=2)
        ttk.Label(start_row, text="Start Time (s):").pack(side=tk.LEFT)
        self.start_time_var = tk.StringVar(value="0.0")
        self.start_entry = ttk.Entry(start_row, textvariable=self.start_time_var, width=10)
        self.start_entry.pack(side=tk.RIGHT)
        ttk.Button(start_row, text="⏺️ MARK IN", style="TButton",
                   command=lambda: self._set_time_from_slider('start')).pack(side=tk.RIGHT, padx=5)
        
        # End time with Mark Out button
        end_row = ttk.Frame(timing_frame)
        end_row.pack(fill=tk.X, pady=2)
        ttk.Label(end_row, text="End Time (s):").pack(side=tk.LEFT)
        self.end_time_var = tk.StringVar(value="0.0")
        self.end_entry = ttk.Entry(end_row, textvariable=self.end_time_var, width=10)
        self.end_entry.pack(side=tk.RIGHT)
        ttk.Button(end_row, text="⏺️ MARK OUT", style="TButton",
                   command=lambda: self._set_time_from_slider('end')).pack(side=tk.RIGHT, padx=5)
        
        # Blur strength
        blur_row = ttk.Frame(timing_frame)
        blur_row.pack(fill=tk.X, pady=5)
        ttk.Label(blur_row, text="Blur Strength:").pack(side=tk.LEFT)
        self.blur_var = tk.IntVar(value=51)
        blur_scale = ttk.Scale(blur_row, from_=5, to=151, variable=self.blur_var,
                               orient=tk.HORIZONTAL, command=self._update_blur_label)
        blur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.blur_label = ttk.Label(blur_row, text="51")
        self.blur_label.pack(side=tk.RIGHT)
        
        # Quick Presets section
        presets_frame = ttk.LabelFrame(right_panel, text="🎯 Quick Presets", padding=10)
        presets_frame.pack(fill=tk.X, pady=(0, 10))
        
        presets_row = ttk.Frame(presets_frame)
        presets_row.pack(fill=tk.X)
        
        presets = [
            ("👤 Face", 100, 120),
            ("🚗 Plate", 150, 40),
            ("📱 Phone", 80, 160),
            ("📝 Doc", 200, 280),
        ]
        for name, w, h in presets:
            ttk.Button(presets_row, text=name, style="Secondary.TButton",
                command=lambda w=w, h=h: self._set_preset_mode(w, h)).pack(side=tk.LEFT, padx=2, expand=True)
        
        # Blur regions list
        regions_frame = ttk.LabelFrame(right_panel, text="📋 Blur Regions", padding=10)
        regions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Treeview for regions
        columns = ("id", "position", "timing", "strength")
        self.regions_tree = ttk.Treeview(regions_frame, columns=columns, show="headings", height=6)
        self.regions_tree.heading("id", text="#")
        self.regions_tree.heading("position", text="Position")
        self.regions_tree.heading("timing", text="Timing")
        self.regions_tree.heading("strength", text="Blur")
        
        self.regions_tree.column("id", width=30)
        self.regions_tree.column("position", width=100)
        self.regions_tree.column("timing", width=100)
        self.regions_tree.column("strength", width=50)
        
        self.regions_tree.pack(fill=tk.BOTH, expand=True)
        self.regions_tree.bind("<<TreeviewSelect>>", self._on_region_select)
        
        # Region buttons
        region_btns = ttk.Frame(regions_frame)
        region_btns.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(region_btns, text="🗑️ Delete", style="Secondary.TButton",
                   command=self._delete_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="✏️ Update", style="Secondary.TButton",
                   command=self._update_selected_region).pack(side=tk.LEFT, padx=2)
        ttk.Button(region_btns, text="🧹 Clear All", style="Secondary.TButton",
                   command=self._clear_all_regions).pack(side=tk.RIGHT, padx=2)
        
        # Export section
        export_frame = ttk.LabelFrame(right_panel, text="💾 Export", padding=10)
        export_frame.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(export_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(export_frame, text="Ready to export")
        self.progress_label.pack(fill=tk.X)
        
        ttk.Button(export_frame, text="🚀 Export Blurred Video", 
                   command=self._export_video).pack(fill=tk.X, pady=(10, 0))
        
        # Instructions
        instructions = ttk.Label(right_panel, 
            text="💡 Instructions:\n"
                 "1. Open a video file\n"
                 "2. Navigate to start time\n"
                 "3. Draw a rectangle on the video\n"
                 "4. Set start/end times and blur strength\n"
                 "5. Add multiple regions if needed\n"
                 "6. Export the blurred video",
            wraplength=320, justify=tk.LEFT)
        instructions.pack(fill=tk.X, pady=10)
        
    def _open_video(self):
        """Open a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.wmv"),
                ("MP4 Files", "*.mp4"),
                ("AVI Files", "*.avi"),
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
            
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps
        
        # Update UI
        self.file_label.config(text=f"📄 {Path(file_path).name}\n"
                                    f"📐 {self.video_width}x{self.video_height}\n"
                                    f"🎞️ {self.fps:.2f} FPS | {self.duration:.2f}s")
        
        self.timeline_slider.config(to=self.duration)
        self.end_time_var.set(f"{self.duration:.2f}")
        
        # Clear existing regions
        self._clear_all_regions()
        
        # Show first frame
        self._show_frame(0)
        
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
            
        # Apply blur regions for preview
        frame = self._apply_blur_regions(frame, time_seconds)
        
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate scaling to fit canvas
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
        
        # Clear and redraw canvas
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_offset_x, self.canvas_offset_y, 
                                 anchor=tk.NW, image=self.photo)
        
        # Draw existing blur regions
        self._draw_blur_regions(time_seconds)
        
        # Update time label
        current_time = time_seconds
        self.time_label.config(text=f"{self._format_time(current_time)} / {self._format_time(self.duration)}")
        
    def _apply_blur_regions(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Apply blur to frame based on active regions"""
        result = frame.copy()
        
        for region in self.blur_regions:
            if region.contains_frame(current_time):
                x1 = max(0, region.x)
                y1 = max(0, region.y)
                x2 = min(frame.shape[1], region.x + region.width)
                y2 = min(frame.shape[0], region.y + region.height)
                
                if x2 > x1 and y2 > y1:
                    roi = result[y1:y2, x1:x2]
                    blur_size = region.blur_strength
                    if blur_size % 2 == 0:
                        blur_size += 1
                    blurred = cv2.GaussianBlur(roi, (blur_size, blur_size), 0)
                    result[y1:y2, x1:x2] = blurred
                    
        return result
        
    def _draw_blur_regions(self, current_time: float):
        """Draw blur region rectangles on canvas"""
        for i, region in enumerate(self.blur_regions):
            # Convert to canvas coordinates
            x1 = int(region.x * self.scale_factor) + self.canvas_offset_x
            y1 = int(region.y * self.scale_factor) + self.canvas_offset_y
            x2 = int((region.x + region.width) * self.scale_factor) + self.canvas_offset_x
            y2 = int((region.y + region.height) * self.scale_factor) + self.canvas_offset_y
            
            # Determine color based on whether region is active
            if region.contains_frame(current_time):
                color = "#e94560"  # Red when active
            else:
                color = "#4a4a6a"  # Gray when inactive
                
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1 + 5, y1 + 5, text=f"#{i+1}", 
                                   fill=color, anchor=tk.NW, font=("Segoe UI", 10, "bold"))
            
    def _on_mouse_down(self, event):
        """Handle mouse press for region selection"""
        if self.cap is None:
            return
        
        # Hide quick toolbar if visible
        self._hide_quick_toolbar()
        
        # Check for preset mode - create region at click
        if self.preset_size:
            w, h = self.preset_size
            video_x = int((event.x - self.canvas_offset_x) / self.scale_factor) - w // 2
            video_y = int((event.y - self.canvas_offset_y) / self.scale_factor) - h // 2
            
            # Clamp to video dimensions
            video_x = max(0, min(video_x, self.video_width - w))
            video_y = max(0, min(video_y, self.video_height - h))
            
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
                x=video_x, y=video_y, width=w, height=h,
                start_time=start_time, end_time=end_time,
                blur_strength=blur_strength
            )
            self.blur_regions.append(region)
            self._update_regions_list()
            self._show_frame(self.time_var.get())
            
            # Show quick toolbar
            self._show_quick_toolbar(event.x_root, event.y_root, len(self.blur_regions) - 1)
            
            self.preset_size = None
            self.canvas.config(cursor="crosshair")
            return
        
        # Check if clicking on an existing region to drag it
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            self.dragging_region = region_idx
            self.drag_start_pos = (event.x, event.y)
            region = self.blur_regions[region_idx]
            self.drag_start_region = {'x': region.x, 'y': region.y}
            return
            
        self.is_selecting = True
        self.selection_start = (event.x, event.y)
        
    def _on_mouse_drag(self, event):
        """Handle mouse drag for region selection or region dragging"""
        # Handle region dragging
        if self.dragging_region is not None and self.drag_start_pos and self.drag_start_region:
            dx = int((event.x - self.drag_start_pos[0]) / self.scale_factor)
            dy = int((event.y - self.drag_start_pos[1]) / self.scale_factor)
            
            region = self.blur_regions[self.dragging_region]
            new_x = self.drag_start_region['x'] + dx
            new_y = self.drag_start_region['y'] + dy
            
            # Clamp to video dimensions
            new_x = max(0, min(new_x, self.video_width - region.width))
            new_y = max(0, min(new_y, self.video_height - region.height))
            
            region.x = new_x
            region.y = new_y
            self._show_frame(self.time_var.get())
            return
        
        # Handle selection dragging
        if not self.is_selecting or self.selection_start is None:
            return
            
        # Delete previous selection rectangle
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
            
        x1, y1 = self.selection_start
        x2, y2 = event.x, event.y
        
        self.selection_rect = self.canvas.create_rectangle(
            x1, y1, x2, y2, outline="#00ff88", width=2, dash=(5, 5)
        )
        self.temp_rect = (x1, y1, x2, y2)
        
    def _on_mouse_up(self, event):
        """Handle mouse release to finalize region selection or dragging"""
        # Handle end of region dragging
        if self.dragging_region is not None:
            self.dragging_region = None
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
        
        # Normalize coordinates
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Convert from canvas to video coordinates
        video_x1 = int((x1 - self.canvas_offset_x) / self.scale_factor)
        video_y1 = int((y1 - self.canvas_offset_y) / self.scale_factor)
        video_x2 = int((x2 - self.canvas_offset_x) / self.scale_factor)
        video_y2 = int((y2 - self.canvas_offset_y) / self.scale_factor)
        
        # Clamp to video dimensions
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
            
        # Get timing values
        try:
            start_time = float(self.start_time_var.get())
            end_time = float(self.end_time_var.get())
        except ValueError:
            start_time = 0.0
            end_time = self.duration
            
        blur_strength = self.blur_var.get()
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        # Create blur region
        region = BlurRegion(
            x=video_x1,
            y=video_y1,
            width=width,
            height=height,
            start_time=start_time,
            end_time=end_time,
            blur_strength=blur_strength
        )
        
        self.blur_regions.append(region)
        self._update_regions_list()
        
        # Clear selection
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        self.selection_rect = None
        self.temp_rect = None
        
        # Refresh display
        self._show_frame(self.time_var.get())
        
        # Show quick toolbar for the new region
        self._show_quick_toolbar(event.x_root, event.y_root, len(self.blur_regions) - 1)
        
    def _update_regions_list(self):
        """Update the regions treeview"""
        # Clear existing items
        for item in self.regions_tree.get_children():
            self.regions_tree.delete(item)
            
        # Add regions
        for i, region in enumerate(self.blur_regions):
            self.regions_tree.insert("", tk.END, values=(
                i + 1,
                f"{region.x},{region.y} {region.width}x{region.height}",
                f"{region.start_time:.1f}s - {region.end_time:.1f}s",
                region.blur_strength
            ))
            
    def _on_region_select(self, event):
        """Handle region selection in treeview"""
        selection = self.regions_tree.selection()
        if selection:
            item = selection[0]
            idx = self.regions_tree.index(item)
            self.current_region_id = idx
            
            # Load region values into controls
            region = self.blur_regions[idx]
            self.start_time_var.set(f"{region.start_time:.2f}")
            self.end_time_var.set(f"{region.end_time:.2f}")
            self.blur_var.set(region.blur_strength)
            self._update_blur_label(None)
            
    def _delete_selected_region(self):
        """Delete the selected region"""
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
        """Update the selected region with current values"""
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
        """Clear all blur regions"""
        self.blur_regions.clear()
        self._update_regions_list()
        if self.cap:
            self._show_frame(self.time_var.get())
            
    def _on_timeline_change(self, value):
        """Handle timeline slider change"""
        if self.cap:
            self._show_frame(float(value))
            
    def _set_time_from_slider(self, which: str):
        """Set start or end time from current slider position"""
        current_time = self.time_var.get()
        if which == 'start':
            self.start_time_var.set(f"{current_time:.2f}")
        else:
            self.end_time_var.set(f"{current_time:.2f}")
            
    def _seek_relative(self, delta: float):
        """Seek relative to current position"""
        if self.cap is None:
            return
            
        new_time = self.time_var.get() + delta
        new_time = max(0, min(new_time, self.duration))
        self.time_var.set(new_time)
        self._show_frame(new_time)
        
    def _toggle_preview(self):
        """Toggle video preview playback"""
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
        """Preview playback loop"""
        while self.preview_running:
            current_time = self.time_var.get()
            current_time += 1/self.fps
            
            if current_time >= self.duration:
                self.preview_running = False
                self.root.after(0, lambda: self.play_btn.config(text="▶️ Play"))
                break
                
            self.time_var.set(current_time)
            self.root.after(0, lambda t=current_time: self._show_frame(t))
            
            import time
            time.sleep(1/self.fps)
            
    def _update_blur_label(self, value):
        """Update blur strength label"""
        val = self.blur_var.get()
        if val % 2 == 0:
            val += 1
        self.blur_label.config(text=str(val))
        
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS.ms"""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:05.2f}"
        
    def _export_video(self):
        """Export the blurred video"""
        if self.cap is None:
            messagebox.showerror("Error", "Please open a video first")
            return
            
        if not self.blur_regions:
            messagebox.showwarning("Warning", "No blur regions defined")
            return
            
        if self.is_processing:
            messagebox.showinfo("Info", "Export already in progress")
            return
            
        # Get output path
        output_path = filedialog.asksaveasfilename(
            title="Save Blurred Video",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 Files", "*.mp4"),
                ("AVI Files", "*.avi"),
                ("All Files", "*.*")
            ],
            initialfile=f"{Path(self.video_path).stem}_blurred.mp4"
        )
        
        if not output_path:
            return
            
        self.is_processing = True
        threading.Thread(target=self._export_thread, args=(output_path,), daemon=True).start()
        
    def _export_thread(self, output_path: str):
        """Export video in a separate thread"""
        try:
            # Open input video
            cap = cv2.VideoCapture(self.video_path)
            
            # Create output writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                  (self.video_width, self.video_height))
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = frame_count / self.fps
                
                # Apply blur regions
                frame = self._apply_blur_regions(frame, current_time)
                
                out.write(frame)
                frame_count += 1
                
                # Update progress
                progress = (frame_count / self.total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda f=frame_count: 
                    self.progress_label.config(text=f"Processing frame {f}/{self.total_frames}"))
                
            cap.release()
            out.release()
            
            self.root.after(0, lambda: self._export_complete(output_path))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Export failed: {str(e)}"))
        finally:
            self.is_processing = False
            
    def _export_complete(self, output_path: str):
        """Called when export is complete"""
        self.progress_var.set(100)
        self.progress_label.config(text="Export complete!")
        messagebox.showinfo("Success", f"Video exported to:\n{output_path}")
    
    # ==================== NEW MOUSE-CENTRIC FEATURES ====================
    
    def _create_context_menus(self):
        """Create right-click context menus"""
        # Canvas context menu
        self.canvas_menu = tk.Menu(self.root, tearoff=0, bg="#16213e", fg="#eaeaea",
                                   activebackground="#e94560", activeforeground="white")
        self.canvas_menu.add_command(label="📍 Set Start Time Here", command=self._set_start_from_current)
        self.canvas_menu.add_command(label="🏁 Set End Time Here", command=self._set_end_from_current)
        self.canvas_menu.add_separator()
        self.canvas_menu.add_command(label="🧹 Clear All Regions", command=self._clear_all_regions)
        
        # Region context menu
        self.region_menu = tk.Menu(self.root, tearoff=0, bg="#16213e", fg="#eaeaea",
                                   activebackground="#e94560", activeforeground="white")
        self.region_menu.add_command(label="📋 Duplicate Region", command=self._duplicate_clicked_region)
        self.region_menu.add_command(label="🎬 Apply to Whole Video", command=self._apply_whole_video)
        self.region_menu.add_command(label="▶️ Apply From Here", command=self._apply_from_here)
        self.region_menu.add_command(label="⏹️ Apply To Here", command=self._apply_to_here)
        self.region_menu.add_separator()
        self.region_menu.add_command(label="🗑️ Delete Region", command=self._delete_clicked_region)
    
    def _on_right_click(self, event):
        """Handle right-click context menu"""
        if self.cap is None:
            return
        
        # Check if clicking on a region
        region_idx = self._get_region_at(event.x, event.y)
        
        if region_idx is not None:
            self.clicked_region_idx = region_idx
            self.region_menu.tk_popup(event.x_root, event.y_root)
        else:
            self.canvas_menu.tk_popup(event.x_root, event.y_root)
    
    def _get_region_at(self, canvas_x: int, canvas_y: int) -> Optional[int]:
        """Get the index of region at canvas coordinates"""
        for i, region in enumerate(self.blur_regions):
            x1 = int(region.x * self.scale_factor) + self.canvas_offset_x
            y1 = int(region.y * self.scale_factor) + self.canvas_offset_y
            x2 = int((region.x + region.width) * self.scale_factor) + self.canvas_offset_x
            y2 = int((region.y + region.height) * self.scale_factor) + self.canvas_offset_y
            
            if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                return i
        return None
    
    def _set_start_from_current(self):
        """Set start time from current timeline position"""
        self.start_time_var.set(f"{self.time_var.get():.2f}")
    
    def _set_end_from_current(self):
        """Set end time from current timeline position"""
        self.end_time_var.set(f"{self.time_var.get():.2f}")
    
    def _duplicate_clicked_region(self):
        """Duplicate the clicked region"""
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            region = self.blur_regions[self.clicked_region_idx]
            new_region = BlurRegion(
                x=region.x + 20, y=region.y + 20,
                width=region.width, height=region.height,
                start_time=region.start_time, end_time=region.end_time,
                blur_strength=region.blur_strength
            )
            self.blur_regions.append(new_region)
            self._update_regions_list()
            self._show_frame(self.time_var.get())
    
    def _apply_whole_video(self):
        """Apply clicked region to whole video"""
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].start_time = 0.0
            self.blur_regions[self.clicked_region_idx].end_time = self.duration
            self._update_regions_list()
            self._show_frame(self.time_var.get())
    
    def _apply_from_here(self):
        """Apply clicked region from current time to end"""
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].start_time = self.time_var.get()
            self.blur_regions[self.clicked_region_idx].end_time = self.duration
            self._update_regions_list()
            self._show_frame(self.time_var.get())
    
    def _apply_to_here(self):
        """Apply clicked region from start to current time"""
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            self.blur_regions[self.clicked_region_idx].start_time = 0.0
            self.blur_regions[self.clicked_region_idx].end_time = self.time_var.get()
            self._update_regions_list()
            self._show_frame(self.time_var.get())
    
    def _delete_clicked_region(self):
        """Delete the clicked region"""
        if self.clicked_region_idx is not None and self.clicked_region_idx < len(self.blur_regions):
            del self.blur_regions[self.clicked_region_idx]
            self._update_regions_list()
            self._show_frame(self.time_var.get())
    
    def _on_canvas_scroll(self, event):
        """Handle scroll wheel on canvas"""
        if self.cap is None:
            return
        
        # Check if hovering over a region - adjust blur strength
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            delta = event.delta / 120
            region = self.blur_regions[region_idx]
            new_strength = region.blur_strength + int(delta * 10)
            new_strength = max(5, min(151, new_strength))
            if new_strength % 2 == 0:
                new_strength += 1
            region.blur_strength = new_strength
            self._update_regions_list()
            self._show_frame(self.time_var.get())
            return
        
        # Otherwise scrub timeline
        delta = event.delta / 120
        step = 1 / self.fps if not (event.state & 0x1) else 1.0  # Shift for 1 second
        new_time = self.time_var.get() + (delta * step)
        new_time = max(0, min(new_time, self.duration))
        self.time_var.set(new_time)
        self._show_frame(new_time)
    
    def _on_mouse_motion(self, event):
        """Handle mouse motion for hover effects and cursor changes"""
        if self.cap is None:
            return
        
        # Check for preset mode
        if self.preset_size:
            self.canvas.config(cursor="target")
            return
        
        # Check if over a region
        region_idx = self._get_region_at(event.x, event.y)
        if region_idx is not None:
            self.hovered_region = region_idx
            self.canvas.config(cursor="fleur")  # Move cursor
        else:
            self.hovered_region = None
            self.canvas.config(cursor="crosshair")
    
    def _set_preset_mode(self, width: int, height: int):
        """Set preset mode for quick region creation"""
        self.preset_size = (width, height)
        self.canvas.config(cursor="target")
    
    def _show_quick_toolbar(self, event_x_root: int, event_y_root: int, region_idx: int):
        """Show quick action toolbar after creating a region"""
        self._hide_quick_toolbar()
        
        self.quick_toolbar = tk.Toplevel(self.root)
        self.quick_toolbar.overrideredirect(True)
        self.quick_toolbar.geometry(f"+{event_x_root}+{event_y_root + 10}")
        self.quick_toolbar.configure(bg="#16213e")
        self.quick_toolbar.attributes('-topmost', True)
        
        self.clicked_region_idx = region_idx
        
        btn_frame = tk.Frame(self.quick_toolbar, bg="#16213e")
        btn_frame.pack(padx=2, pady=2)
        
        tk.Button(btn_frame, text="🎬 Whole", bg="#16213e", fg="white", bd=0,
                  command=lambda: [self._apply_whole_video(), self._hide_quick_toolbar()]).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_frame, text="▶️ From", bg="#16213e", fg="white", bd=0,
                  command=lambda: [self._apply_from_here(), self._hide_quick_toolbar()]).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_frame, text="⏹️ To", bg="#16213e", fg="white", bd=0,
                  command=lambda: [self._apply_to_here(), self._hide_quick_toolbar()]).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_frame, text="🗑️", bg="#e94560", fg="white", bd=0,
                  command=lambda: [self._delete_clicked_region(), self._hide_quick_toolbar()]).pack(side=tk.LEFT, padx=1)
        
        self.quick_toolbar.after(5000, self._hide_quick_toolbar)
    
    def _hide_quick_toolbar(self):
        """Hide the quick toolbar"""
        if self.quick_toolbar:
            try:
                self.quick_toolbar.destroy()
            except:
                pass
            self.quick_toolbar = None


def main():
    # Check for PIL
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Installing Pillow...")
        import subprocess
        subprocess.run(["pip", "install", "Pillow", "--break-system-packages", "-q"])
        from PIL import Image, ImageTk
    
    root = tk.Tk()
    app = VideoBlurTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
