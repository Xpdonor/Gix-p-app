import os
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER
import threading
import json
import glob
from .ai_model import AIModule
from .analyzer import apply_eq_and_write
from .recommender import MusicIndex

class GixpMusicPlayer(toga.App):
    def __init__(self, formal_name, app_id):
        super().__init__(formal_name, app_id)
        
        # Properties equivalent to Kivy properties
        self.music_dir = os.path.expanduser("~/Music")
        self.songs = []
        self.current_path = None
        self.sound = None
        self.ai_mode = True
        self.auto_eq = False
        self.output = "speaker"
        self.aim = AIModule()
        self.index = None
        
        # UI components
        self.cover_image = None
        self.now_label = None
        self.info_label = None
        self.song_list = None

    def startup(self):
        # Main container
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        # Top button row
        button_box = toga.Box(style=Pack(direction=ROW, height=50, padding=5))
        
        self.choose_btn = toga.Button('Choose Folder', on_press=self.open_folder_chooser, style=Pack(flex=1, padding=5))
        self.scan_btn = toga.Button('Scan & Index', on_press=self.scan_and_index, style=Pack(flex=1, padding=5))
        self.output_btn = toga.Button('Output', on_press=self.open_output_chooser, style=Pack(flex=1, padding=5))
        self.ai_btn = toga.Button('AI: ON', on_press=self.toggle_ai, style=Pack(flex=1, padding=5))
        
        button_box.add(self.choose_btn)
        button_box.add(self.scan_btn)
        button_box.add(self.output_btn)
        button_box.add(self.ai_btn)

        # Content area
        content_box = toga.Box(style=Pack(direction=ROW, flex=1))
        
        # Left side - Song list
        left_box = toga.Box(style=Pack(direction=COLUMN, width=300, padding=5))
        
        self.song_list = toga.DetailedList(
            on_select=self.play_track,
            style=Pack(flex=1)
        )
        
        self.info_label = toga.Label('Ready', style=Pack(height=30, padding=5))
        
        left_box.add(self.song_list)
        left_box.add(self.info_label)

        # Right side - Player controls
        right_box = toga.Box(style=Pack(direction=COLUMN, flex=1, padding=10))
        
        # Cover image
        self.cover_image = toga.ImageView(
            image=toga.Icon('resources/icon'),
            style=Pack(height=200, width=200, padding=10)
        )
        
        self.now_label = toga.Label('Stopped', style=Pack(height=40, padding=5, text_align=CENTER))
        
        # Control buttons
        control_box = toga.Box(style=Pack(direction=ROW, height=60, padding=10))
        
        self.prev_btn = toga.Button('⏮', on_press=self.prev_track, style=Pack(flex=1, padding=5))
        self.play_btn = toga.Button('▶️', on_press=self.toggle_play, style=Pack(flex=1, padding=5))
        self.next_btn = toga.Button('⏭', on_press=self.next_track, style=Pack(flex=1, padding=5))
        self.eq_btn = toga.Button('Auto-EQ: OFF', on_press=self.toggle_auto_eq, style=Pack(flex=1, padding=5))
        
        control_box.add(self.prev_btn)
        control_box.add(self.play_btn)
        control_box.add(self.next_btn)
        control_box.add(self.eq_btn)

        right_box.add(self.cover_image)
        right_box.add(self.now_label)
        right_box.add(control_box)

        content_box.add(left_box)
        content_box.add(right_box)

        main_box.add(button_box)
        main_box.add(content_box)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

        # Load initial data
        self.load_config()
        self.populate_from_dir()

    def load_config(self):
        cfg_path = "config.json"
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.music_dir = data.get("music_dir", self.music_dir)
                    self.auto_eq = data.get("auto_eq", False)
                    self.ai_mode = data.get("ai_mode", True)
                    self.output = data.get("output", "speaker")
            except:
                pass

    def save_config(self):
        cfg_path = "config.json"
        data = {
            "music_dir": self.music_dir,
            "auto_eq": self.auto_eq,
            "ai_mode": self.ai_mode,
            "output": self.output
        }
        try:
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except:
            pass

    def open_folder_chooser(self, widget=None):
        # BeeWare file dialog
        async def select_folder(widget):
            try:
                folder_path = await self.main_window.select_folder_dialog(
                    title="Choose music folder"
                )
                if folder_path:
                    self.music_dir = folder_path
                    self.save_config()
                    self.populate_from_dir()
            except Exception as e:
                print(f"Folder selection error: {e}")

        self.main_window.loop.create_task(select_folder(widget))

    def open_output_chooser(self, widget=None):
        # Simple output selection
        def set_output(device):
            self.output = device
            self.info_label.text = f"Output: {device}"
            self.save_config()

        # Create a simple dialog
        self.main_window.info_dialog("Output Device", "Choose output device in settings")

    def populate_from_dir(self):
        def _scan():
            exts = ['*.mp3', '*.flac', '*.wav', '*.m4a']
            files = []
            for e in exts:
                files += glob.glob(os.path.join(self.music_dir, '**', e), recursive=True)
            
            self.songs = sorted(files)
            
            # Update UI on main thread
            self.main_window.loop.call_soon_threadsafe(self._update_song_list)

        threading.Thread(target=_scan, daemon=True).start()

    def _update_song_list(self):
        song_titles = [os.path.basename(p) for p in self.songs]
        self.song_list.data = song_titles
        self.info_label.text = f"{len(self.songs)} tracks found"

    def scan_and_index(self, widget=None):
        def _build():
            self.info_label.text = "Indexing..."
            self.index = MusicIndex(self.songs)
            self.aim.build_index(self.songs)
            self.info_label.text = f"Indexed {len(self.songs)} tracks"

        threading.Thread(target=_build, daemon=True).start()

    def play_track(self, widget, row=None):
        if row is None or row >= len(self.songs):
            return
            
        path = self.songs[row]
        self.current_path = path
        self.now_label.text = os.path.basename(path)
        
        # TODO: Implement audio playback
        # For now, just update the UI
        self.info_label.text = f"Playing: {os.path.basename(path)}"

    def toggle_play(self, widget=None):
        # TODO: Implement play/pause functionality
        if self.current_path:
            self.info_label.text = "Playback toggled"

    def prev_track(self, widget=None):
        if not self.current_path or not self.songs:
            return
            
        try:
            idx = self.songs.index(self.current_path)
            if idx > 0:
                self.play_track(None, idx - 1)
        except ValueError:
            pass

    def next_track(self, widget=None):
        if not self.current_path or not self.songs:
            return
            
        try:
            idx = self.songs.index(self.current_path)
            if idx < len(self.songs) - 1:
                self.play_track(None, idx + 1)
        except ValueError:
            pass

    def toggle_ai(self, widget=None):
        self.ai_mode = not self.ai_mode
        self.ai_btn.text = f"AI: {'ON' if self.ai_mode else 'OFF'}"
        self.info_label.text = f"AI {'ON' if self.ai_mode else 'OFF'}"
        self.save_config()

    def toggle_auto_eq(self, widget=None):
        self.auto_eq = not self.auto_eq
        self.eq_btn.text = f"Auto-EQ: {'ON' if self.auto_eq else 'OFF'}"
        self.info_label.text = f"Auto-EQ {'ON' if self.auto_eq else 'OFF'}"
        self.save_config()

def main():
    return GixpMusicPlayer('Gixp Music Player', 'com.parham.gixp')
