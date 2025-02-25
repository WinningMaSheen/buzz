import os
import re
import enum
import logging
import datetime
import sounddevice
from enum import auto
from typing import Optional, Tuple, Any
from queue import Queue, Empty
from threading import Thread, Lock, Event
import time
import subprocess

from PyQt6.QtCore import QThread, Qt, QThreadPool, QTimer
from PyQt6.QtGui import QTextCursor, QCloseEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QMessageBox, QCheckBox,
    QGroupBox, QLabel, QSlider, QPushButton
)

from buzz.dialogs import show_model_download_error_dialog
from buzz.locale import _
from buzz.model_loader import (
    ModelDownloader,
    TranscriptionModel,
    ModelType,
)
from buzz.store.keyring_store import get_password, Key
from buzz.recording import RecordingAmplitudeListener
from buzz.settings.settings import Settings
from buzz.settings.recording_transcriber_mode import RecordingTranscriberMode
from buzz.transcriber.recording_transcriber import RecordingTranscriber
from buzz.transcriber.transcriber import (
    TranscriptionOptions,
    DEFAULT_WHISPER_TEMPERATURE,
    Task,
)
from buzz.translator import Translator
from buzz.widgets.audio_devices_combo_box import AudioDevicesComboBox
from buzz.widgets.audio_meter_widget import AudioMeterWidget
from buzz.widgets.model_download_progress_dialog import ModelDownloadProgressDialog
from buzz.widgets.record_button import RecordButton
from buzz.widgets.text_display_box import TextDisplayBox
from buzz.widgets.transcriber.transcription_options_group_box import (
    TranscriptionOptionsGroupBox,
)

REAL_CHARS_REGEX = re.compile(r'\w')
NO_SPACE_BETWEEN_SENTENCES = re.compile(r'([.!?。！？])([A-Z])')


class ImprovedTTSManager:
    """A TTS manager that uses system commands with a proper sequential queue and failsafe mechanisms"""
    
    def __init__(self):
        self.is_enabled = False
        self.current_process = None
        self.platform = None
        self.message_queue = []
        self.is_processing = False
        # Default speed settings for different platforms
        # macOS: normal = 175-200 wpm
        # Windows: normal = 0 (range -10 to 10)
        # Linux: normal = 175 wpm
        self.speed = 1.0  # Multiplier for default speed (1.0 = normal)
        self.process_timeout = 30  # Maximum time in seconds to wait for a TTS process
        self.max_queue_size = 50   # Maximum number of items in the queue
        self.process_start_time = None  # Track when current process started
        logging.debug("ImprovedTTSManager initialized")
        
        # Set up the queue check timer
        self.queue_timer = QTimer()
        self.queue_timer.timeout.connect(self._check_queue)
        self.queue_timer.setInterval(150)  # Check every 150ms (70% reduction)
    
    def _get_platform(self):
        """Detect the platform to determine which TTS method to use"""
        if self.platform is None:
            try:
                import platform
                self.platform = platform.system()
                logging.debug(f"Detected platform: {self.platform}")
            except Exception as e:
                logging.error(f"Error detecting platform: {str(e)}")
                self.platform = "Unknown"
        return self.platform
    
    def add_to_queue(self, text):
        """Add text to the speech queue with overflow protection"""
        if not self.is_enabled or not text or not text.strip():
            return
        
        try:
            text = text.strip()
            
            # Queue size management - if queue is too large, drop oldest items
            if len(self.message_queue) >= self.max_queue_size:
                overflow = len(self.message_queue) - self.max_queue_size + 1
                self.message_queue = self.message_queue[overflow:]
                logging.warning(f"TTS queue overflow: dropped {overflow} oldest items")
            
            logging.debug(f"Adding to TTS queue: {text[:50]}... (Queue size: {len(self.message_queue)})")
            
            # Add the text to our queue
            self.message_queue.append(text)
            
            # Make sure the queue timer is running
            if not self.queue_timer.isActive():
                self.queue_timer.start()
                
        except Exception as e:
            logging.error(f"Error in TTS add_to_queue: {str(e)}")
    
    def _check_queue(self):
        """Main timer-based method to check queue status and process items"""
        if not self.is_enabled:
            self.queue_timer.stop()
            return
            
        # If we have a current process, check if it's still running
        if self.current_process:
            # Check if process has completed
            if self.current_process.poll() is not None:
                logging.debug(f"TTS process completed with code {self.current_process.returncode}")
                self.current_process = None
                self.process_start_time = None
            # Check if process has been running too long
            elif self.process_start_time and time.time() - self.process_start_time > self.process_timeout:
                logging.warning("TTS process is taking too long - terminating")
                self._terminate_current_process()
            else:
                # Process is still running normally, nothing to do
                return
        
        # If we get here, there's no current process running
        
        # If queue is empty, stop processing
        if not self.message_queue:
            return
        
        # Get the next text from the queue and speak it
        text = self.message_queue.pop(0)
        self._speak_text(text)
    
    def _terminate_current_process(self):
        """Safely terminate the current TTS process"""
        if self.current_process:
            try:
                self.current_process.terminate()
            except Exception as e:
                logging.error(f"Error terminating TTS process: {str(e)}")
            finally:
                self.current_process = None
                self.process_start_time = None
    
    def _speak_text(self, text):
        """Speak text using the appropriate platform command with speed control and error handling"""
        if not text:
            return
            
        try:
            import subprocess
            platform = self._get_platform()
            
            if platform == "Darwin":  # macOS
                # Calculate rate based on speed multiplier (normal is ~175-200 wpm)
                rate = int(175 * self.speed)
                # Use the 'say' command on macOS with rate control
                self.current_process = subprocess.Popen(
                    ["say", "-r", str(rate), text], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            
            elif platform == "Windows":
                # Use PowerShell to speak text on Windows with rate control
                # Windows SpeechSynthesizer Rate range: -10 (slow) to 10 (fast), 0 is normal
                rate_value = int((self.speed - 1.0) * 10)  # Convert multiplier to Windows range
                rate_value = max(-10, min(10, rate_value))  # Ensure within valid range
                
                # Limit text length for PowerShell to avoid issues
                if len(text) > 500:
                    text = text[:497] + "..."
                
                text_escaped = text.replace('"', '`"').replace('$', '`$')
                powershell_cmd = (
                    f'Add-Type -AssemblyName System.Speech; '
                    f'$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                    f'$synth.Rate = {rate_value}; '
                    f'$synth.Speak("{text_escaped}"); '
                    f'$synth.Dispose()'  # Explicitly dispose to free resources
                )
                self.current_process = subprocess.Popen(
                    ["powershell", "-Command", powershell_cmd],
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
            
            elif platform == "Linux":
                # Try espeak on Linux if available, with speed control
                # espeak normal speed is ~175 wpm
                speed = int(175 * self.speed)
                try:
                    self.current_process = subprocess.Popen(
                        ["espeak", "-s", str(speed), text],
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL
                    )
                except FileNotFoundError:
                    # Try festival as a fallback
                    try:
                        text_escaped = text.replace('"', '\\"')
                        self.current_process = subprocess.Popen(
                            ["echo", text_escaped, "|", "festival", "--tts"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            shell=True
                        )
                    except FileNotFoundError:
                        logging.warning("Neither espeak nor festival found on Linux, TTS will not work")
                        return
            
            else:
                logging.warning(f"TTS not supported on platform: {platform}")
                return
            
            # Record process start time
            if self.current_process:
                self.process_start_time = time.time()
                logging.debug(f"TTS process started for text: {text[:50]}... (speed: {self.speed})")
            
        except Exception as e:
            logging.error(f"Error executing TTS command: {str(e)}")
            self.current_process = None
            self.process_start_time = None
    
    def set_enabled(self, enabled):
        """Enable or disable TTS"""
        was_enabled = self.is_enabled
        self.is_enabled = enabled
        logging.debug(f"TTS enabled changed from {was_enabled} to {enabled}")
        
        if not enabled:
            # Clear the queue when disabling
            self.message_queue.clear()
            
            if self.current_process is not None:
                self._terminate_current_process()
                
            # Stop the timer
            self.queue_timer.stop()
                
        elif enabled and not was_enabled:
            # Start the timer if we're enabling
            self.queue_timer.start()
    
    def set_speed(self, speed_multiplier):
        """Set the TTS speed multiplier (1.0 is normal speed)"""
        try:
            # Ensure speed is within a reasonable range (0.5 to 2.0)
            speed_multiplier = float(speed_multiplier)
            speed_multiplier = max(0.5, min(2.0, speed_multiplier))
            
            self.speed = speed_multiplier
            logging.debug(f"TTS speed set to {self.speed}")
            return True
        except Exception as e:
            logging.error(f"Error setting TTS speed: {str(e)}")
            return False
    
    def skip_current(self):
        """Skip the current item in the TTS queue"""
        try:
            # Terminate the current process if it exists
            self._terminate_current_process()
            logging.debug("Skipped current TTS item")
            return True
        except Exception as e:
            logging.error(f"Error skipping current TTS item: {str(e)}")
            return False
    
    def check_health(self):
        """Check the health of the TTS system"""
        # Make sure timer is running if enabled
        if self.is_enabled and not self.queue_timer.isActive():
            logging.warning("TTS health check: timer not active, restarting")
            self.queue_timer.start()
            return True
        return False
    
    def clear_queue(self):
        """Clear the TTS queue completely"""
        queue_size = len(self.message_queue)
        self.message_queue.clear()
        self._terminate_current_process()
        logging.debug(f"Cleared TTS queue ({queue_size} items)")
    
    def stop(self):
        """Clean up resources"""
        try:
            self.is_enabled = False
            self.message_queue.clear()
            
            if self.queue_timer.isActive():
                self.queue_timer.stop()
                
            self._terminate_current_process()
            logging.debug("TTS manager stopped")
        except Exception as e:
            logging.error(f"Error stopping TTS manager: {str(e)}")
    
    def __del__(self):
        self.stop()

class RecordingTranscriberWidget(QWidget):
    current_status: "RecordingStatus"
    transcription_options: TranscriptionOptions
    selected_device_id: Optional[int]
    model_download_progress_dialog: Optional[ModelDownloadProgressDialog] = None
    transcriber: Optional[RecordingTranscriber] = None
    model_loader: Optional[ModelDownloader] = None
    transcription_thread: Optional[QThread] = None
    recording_amplitude_listener: Optional[RecordingAmplitudeListener] = None
    device_sample_rate: Optional[int] = None

    class RecordingStatus(enum.Enum):
        STOPPED = auto()
        RECORDING = auto()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        flags: Optional[Qt.WindowType] = None,
        custom_sounddevice: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.sounddevice = custom_sounddevice or sounddevice

        if flags is not None:
            self.setWindowFlags(flags)

        layout = QVBoxLayout(self)

        self.translation_thread = None
        self.translator = None
        self.transcripts = []
        self.translations = []
        self.current_status = self.RecordingStatus.STOPPED
        self.setWindowTitle(_("Live Recording"))

        self.settings = Settings()
        self.transcriber_mode = list(RecordingTranscriberMode)[
            self.settings.value(key=Settings.Key.RECORDING_TRANSCRIBER_MODE, default_value=0)]

        default_language = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_LANGUAGE, default_value=""
        )

        model_types = [
            model_type
            for model_type in ModelType
            if model_type.is_available()
        ]
        default_model: Optional[TranscriptionModel] = None
        if len(model_types) > 0:
            default_model = TranscriptionModel(model_type=model_types[0])

        selected_model = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_MODEL,
            default_value=default_model,
        )

        if selected_model is None or selected_model.model_type not in model_types:
            selected_model = default_model

        openai_access_token = get_password(key=Key.OPENAI_API_KEY)

        self.transcription_options = TranscriptionOptions(
            model=selected_model,
            task=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_TASK,
                default_value=Task.TRANSCRIBE,
            ),
            language=default_language if default_language != "" else None,
            openai_access_token=openai_access_token,
            initial_prompt=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_INITIAL_PROMPT, default_value=""
            ),
            temperature=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_TEMPERATURE,
                default_value=DEFAULT_WHISPER_TEMPERATURE,
            ),
            word_level_timings=False,
            enable_llm_translation=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_ENABLE_LLM_TRANSLATION,
                default_value=False,
            ),
            llm_model=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_LLM_MODEL, default_value=""
            ),
            llm_prompt=self.settings.value(
                key=Settings.Key.RECORDING_TRANSCRIBER_LLM_PROMPT, default_value=""
            ),
        )

        self.audio_devices_combo_box = AudioDevicesComboBox(self)
        self.audio_devices_combo_box.device_changed.connect(self.on_device_changed)
        self.selected_device_id = self.audio_devices_combo_box.get_default_device_id()

        self.record_button = RecordButton(self)
        self.record_button.clicked.connect(self.on_record_button_clicked)

        self.transcription_text_box = TextDisplayBox(self)
        self.transcription_text_box.setPlaceholderText(_("Click Record to begin..."))

        self.translation_text_box = TextDisplayBox(self)
        self.translation_text_box.setPlaceholderText(_("Waiting for AI translation..."))

        self.transcription_options_group_box = TranscriptionOptionsGroupBox(
            default_transcription_options=self.transcription_options,
            model_types=model_types,
            parent=self,
        )
        self.transcription_options_group_box.transcription_options_changed.connect(
            self.on_transcription_options_changed
        )

        recording_options_layout = QFormLayout()
        recording_options_layout.addRow(_("Microphone:"), self.audio_devices_combo_box)

        self.audio_meter_widget = AudioMeterWidget(self)

        record_button_layout = QHBoxLayout()
        record_button_layout.addWidget(self.audio_meter_widget)
        record_button_layout.addWidget(self.record_button)

        layout.addWidget(self.transcription_options_group_box)
        layout.addLayout(recording_options_layout)
        layout.addLayout(record_button_layout)
        layout.addWidget(self.transcription_text_box)
        layout.addWidget(self.translation_text_box)

        if not self.transcription_options.enable_llm_translation:
            self.translation_text_box.hide()

        self.setLayout(layout)
        self.resize(450, 500)

        self.reset_recording_amplitude_listener()

        self.transcript_export_file = None
        self.translation_export_file = None
        self.export_enabled = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_EXPORT_ENABLED,
            default_value=False,
        )
        
        # Initialize TTS manager
        self.tts_manager = ImprovedTTSManager()
        
        # Add TTS controls in a group
        self.tts_group_box = QGroupBox(_("Text-to-Speech Controls"))
        tts_layout = QVBoxLayout()
        
        # TTS Enable checkbox
        self.tts_checkbox = QCheckBox(_("Enable Text-to-Speech"))
        self.tts_checkbox.setChecked(False)
        self.tts_checkbox.stateChanged.connect(self.on_tts_toggle)
        tts_layout.addWidget(self.tts_checkbox)
        
        # TTS Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel(_("Speed:")))
        
        self.tts_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.tts_speed_slider.setMinimum(50)  # 0.5x speed
        self.tts_speed_slider.setMaximum(200)  # 2.0x speed
        self.tts_speed_slider.setValue(100)  # 1.0x speed (normal)
        self.tts_speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.tts_speed_slider.setTickInterval(25)
        self.tts_speed_slider.valueChanged.connect(self.on_tts_speed_changed)
        speed_layout.addWidget(self.tts_speed_slider)
        
        self.tts_speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.tts_speed_label)
        tts_layout.addLayout(speed_layout)
        
        # TTS control buttons layout
        buttons_layout = QHBoxLayout()
        
        # Skip button
        self.tts_skip_button = QPushButton(_("Skip Current"))
        self.tts_skip_button.clicked.connect(self.on_tts_skip_clicked)
        self.tts_skip_button.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.tts_skip_button)
        
        # Clear queue button
        self.tts_clear_button = QPushButton(_("Clear Queue"))
        self.tts_clear_button.clicked.connect(self.on_tts_clear_clicked)
        self.tts_clear_button.setEnabled(False)  # Initially disabled
        buttons_layout.addWidget(self.tts_clear_button)
        
        tts_layout.addLayout(buttons_layout)
        
        self.tts_group_box.setLayout(tts_layout)
        layout.addWidget(self.tts_group_box)
        
        # Add a health check timer
        self.tts_health_timer = QTimer()
        self.tts_health_timer.timeout.connect(self.check_tts_health)
        self.tts_health_timer.start(10000)  # Check every 10 seconds

    def on_tts_toggle(self, state):
        try:
            enabled = state == Qt.CheckState.Checked.value
            logging.debug(f"TTS toggle requested: {enabled}")
            self.tts_manager.set_enabled(enabled)
            
            # Enable or disable control buttons based on TTS state
            self.tts_skip_button.setEnabled(enabled)
            self.tts_clear_button.setEnabled(enabled)
            
            if not enabled:
                self.tts_checkbox.setChecked(False)
            
        except Exception as e:
            logging.error(f"Error in TTS toggle: {str(e)}")
            self.tts_checkbox.setChecked(False)
            QMessageBox.warning(
                self,
                "TTS Error",
                "Failed to initialize text-to-speech. Please check your audio settings."
            )
    
    def on_tts_speed_changed(self, value):
        try:
            # Convert slider value (50-200) to speed multiplier (0.5-2.0)
            speed = value / 100.0
            self.tts_speed_label.setText(f"{speed:.1f}x")
            
            if self.tts_manager:
                success = self.tts_manager.set_speed(speed)
                if not success:
                    logging.warning("Failed to set TTS speed")
        except Exception as e:
            logging.error(f"Error setting TTS speed: {str(e)}")
    
    def on_tts_skip_clicked(self):
        try:
            if self.tts_manager:
                self.tts_manager.skip_current()
        except Exception as e:
            logging.error(f"Error skipping TTS item: {str(e)}")
            
    def on_tts_clear_clicked(self):
        try:
            if self.tts_manager:
                self.tts_manager.clear_queue()
        except Exception as e:
            logging.error(f"Error clearing TTS queue: {str(e)}")
            
    def check_tts_health(self):
        try:
            if self.tts_manager and self.tts_checkbox.isChecked():
                self.tts_manager.check_health()
        except Exception as e:
            logging.error(f"Error in TTS health check: {str(e)}")

    def on_transcription_options_changed(
        self, transcription_options: TranscriptionOptions
    ):
        self.transcription_options = transcription_options

        if self.transcription_options.enable_llm_translation:
            self.translation_text_box.show()
        else:
            self.translation_text_box.hide()

    def on_device_changed(self, device_id: int):
        self.selected_device_id = device_id
        self.reset_recording_amplitude_listener()

    def reset_recording_amplitude_listener(self):
        if self.recording_amplitude_listener is not None:
            self.recording_amplitude_listener.stop_recording()

        # Listening to audio will fail if there are no input devices
        if self.selected_device_id is None or self.selected_device_id == -1:
            return

        # Get the device sample rate before starting the listener as the PortAudio
        # function # fails if you try to get the device's settings while recording
        # is in progress.
        self.device_sample_rate = RecordingTranscriber.get_device_sample_rate(
            self.selected_device_id
        )
        logging.debug(f"Device sample rate: {self.device_sample_rate}")

        self.recording_amplitude_listener = RecordingAmplitudeListener(
            input_device_index=self.selected_device_id, parent=self
        )
        self.recording_amplitude_listener.amplitude_changed.connect(
            self.on_recording_amplitude_changed
        )
        self.recording_amplitude_listener.start_recording()

    def on_record_button_clicked(self):
        if self.current_status == self.RecordingStatus.STOPPED:
            self.start_recording()
            self.current_status = self.RecordingStatus.RECORDING
            self.record_button.set_recording()
        else:  # RecordingStatus.RECORDING
            self.stop_recording()
            self.set_recording_status_stopped()

    def start_recording(self):
        self.record_button.setDisabled(True)
        self.transcripts = []
        self.translations = []

        self.transcription_text_box.clear()
        self.translation_text_box.clear()

        if self.export_enabled:
            self.setup_for_export()

        model_path = self.transcription_options.model.get_local_model_path()
        if model_path is not None:
            self.on_model_loaded(model_path)
            return

        self.model_loader = ModelDownloader(model=self.transcription_options.model)
        self.model_loader.signals.progress.connect(self.on_download_model_progress)
        self.model_loader.signals.error.connect(self.on_download_model_error)
        self.model_loader.signals.finished.connect(self.on_model_loaded)
        QThreadPool().globalInstance().start(self.model_loader)

    def on_model_loaded(self, model_path: str):
        self.reset_recording_controls()
        self.model_loader = None

        self.transcription_thread = QThread()

        # TODO: make runnable
        self.transcriber = RecordingTranscriber(
            input_device_index=self.selected_device_id,
            sample_rate=self.device_sample_rate,
            transcription_options=self.transcription_options,
            model_path=model_path,
            sounddevice=self.sounddevice,
        )

        self.transcriber.moveToThread(self.transcription_thread)

        self.transcription_thread.started.connect(self.transcriber.start)
        self.transcription_thread.finished.connect(
            self.transcription_thread.deleteLater
        )

        self.transcriber.transcription.connect(self.on_next_transcription)

        self.transcriber.finished.connect(self.on_transcriber_finished)
        self.transcriber.finished.connect(self.transcription_thread.quit)
        self.transcriber.finished.connect(self.transcriber.deleteLater)

        self.transcriber.error.connect(self.on_transcriber_error)
        self.transcriber.error.connect(self.transcription_thread.quit)
        self.transcriber.error.connect(self.transcriber.deleteLater)

        if self.transcription_options.enable_llm_translation:
            self.translation_thread = QThread()

            self.translator = Translator(
                self.transcription_options,
                self.transcription_options_group_box.advanced_settings_dialog,
            )

            self.translator.moveToThread(self.translation_thread)

            self.translation_thread.started.connect(self.translator.start)
            self.translation_thread.finished.connect(
                self.translation_thread.deleteLater
            )

            self.translator.finished.connect(self.translation_thread.quit)
            self.translator.finished.connect(self.translator.deleteLater)

            self.translator.translation.connect(self.on_next_translation)

            self.translation_thread.start()

        self.transcription_thread.start()

    def on_download_model_progress(self, progress: Tuple[float, float]):
        (current_size, total_size) = progress

        if self.model_download_progress_dialog is None:
            self.model_download_progress_dialog = ModelDownloadProgressDialog(
                model_type=self.transcription_options.model.model_type, parent=self
            )
            self.model_download_progress_dialog.canceled.connect(
                self.on_cancel_model_progress_dialog
            )

        if self.model_download_progress_dialog is not None:
            self.model_download_progress_dialog.set_value(
                fraction_completed=current_size / total_size
            )

    def setup_for_export(self):
        export_folder = self.settings.value(
            key=Settings.Key.RECORDING_TRANSCRIBER_EXPORT_FOLDER,
            default_value="",
        )

        date_time_now = datetime.datetime.now().strftime("%d-%b-%Y %H-%M-%S")

        export_file_name_template = Settings().get_default_export_file_template()

        export_file_name = (
                export_file_name_template.replace("{{ input_file_name }}", "live recording")
                .replace("{{ task }}", self.transcription_options.task.value)
                .replace("{{ language }}", self.transcription_options.language or "")
                .replace("{{ model_type }}", self.transcription_options.model.model_type.value)
                .replace("{{ model_size }}", self.transcription_options.model.whisper_model_size or "")
                .replace("{{ date_time }}", date_time_now)
                + ".txt"
        )

        if not os.path.isdir(export_folder):
            self.export_enabled = False

        self.transcript_export_file = os.path.join(export_folder, export_file_name)
        self.translation_export_file = self.transcript_export_file.replace(".txt", ".translated.txt")

    def set_recording_status_stopped(self):
        self.record_button.set_stopped()
        self.current_status = self.RecordingStatus.STOPPED

    def on_download_model_error(self, error: str):
        self.reset_model_download()
        show_model_download_error_dialog(self, error)
        self.stop_recording()
        self.set_recording_status_stopped()
        self.record_button.setDisabled(False)

    @staticmethod
    def strip_newlines(text):
        return text.replace('\r\n', os.linesep).replace('\n', os.linesep)

    @staticmethod
    def filter_text(text: str):
        text = text.strip()

        if not REAL_CHARS_REGEX.search(text):
            return ""

        return text

    # Copilot magic implementation of a sliding window approach to find the longest common substring between two texts,
    # ignoring the initial differences.
    @staticmethod
    def find_common_part(text1: str, text2: str) -> str:
        len1, len2 = len(text1), len(text2)
        max_len = 0
        end_index = 0

        lcsuff = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    lcsuff[i][j] = lcsuff[i - 1][j - 1] + 1
                    if lcsuff[i][j] > max_len:
                        max_len = lcsuff[i][j]
                        end_index = i
                else:
                    lcsuff[i][j] = 0

        common_part = text1[end_index - max_len:end_index]

        return common_part if len(common_part) >= 5 else ""

    @staticmethod
    def merge_text_no_overlap(text1: str, text2: str) -> str:
        overlap_start = 0
        for i in range(1, min(len(text1), len(text2)) + 1):
            if text1[-i:] == text2[:i]:
                overlap_start = i

        return text1 + text2[overlap_start:]

    def process_transcription_merge(self, text: str, texts, text_box, export_file):
        texts.append(text)

        # Remove possibly errorous parts from overlapping audio chunks
        for i in range(len(texts) - 1):
            common_part = self.find_common_part(texts[i], texts[i + 1])
            if common_part:
                common_length = len(common_part)
                texts[i] = texts[i][:texts[i].rfind(common_part) + common_length]
                texts[i + 1] = texts[i + 1][texts[i + 1].find(common_part):]

        merged_texts = ""
        for text in texts:
            merged_texts = self.merge_text_no_overlap(merged_texts, text)

        merged_texts = NO_SPACE_BETWEEN_SENTENCES.sub(r'\1 \2', merged_texts)

        text_box.setPlainText(merged_texts)
        text_box.moveCursor(QTextCursor.MoveOperation.End)

        if self.export_enabled:
            with open(export_file, "w") as f:
                f.write(merged_texts)

    def on_next_transcription(self, text: str):
        text = self.filter_text(text)

        if len(text) == 0:
            return

        if self.translator is not None:
            self.translator.enqueue(text)

        if self.transcriber_mode == RecordingTranscriberMode.APPEND_BELOW:
            self.transcription_text_box.moveCursor(QTextCursor.MoveOperation.End)
            if len(self.transcription_text_box.toPlainText()) > 0:
                self.transcription_text_box.insertPlainText("\n\n")
            self.transcription_text_box.insertPlainText(text)
            self.transcription_text_box.moveCursor(QTextCursor.MoveOperation.End)

            if self.export_enabled:
                with open(self.transcript_export_file, "a") as f:
                    f.write(text + "\n\n")

        elif self.transcriber_mode == RecordingTranscriberMode.APPEND_ABOVE:
            self.transcription_text_box.moveCursor(QTextCursor.MoveOperation.Start)
            self.transcription_text_box.insertPlainText(text)
            self.transcription_text_box.insertPlainText("\n\n")
            self.transcription_text_box.moveCursor(QTextCursor.MoveOperation.Start)

            if self.export_enabled:
                with open(self.transcript_export_file, "r") as f:
                    existing_content = f.read()

                new_content = text + "\n\n" + existing_content

                with open(self.transcript_export_file, "w") as f:
                    f.write(new_content)

        elif self.transcriber_mode == RecordingTranscriberMode.APPEND_AND_CORRECT:
            self.process_transcription_merge(text, self.transcripts, self.transcription_text_box, self.transcript_export_file)

    def on_next_translation(self, text: str, _: Optional[int] = None):
        if len(text.strip()) == 0:
            return
        
        try:
            # Add text to TTS queue if TTS is enabled, with improved handling
            if self.tts_checkbox.isChecked():
                # Preprocess text for TTS - break long text into smaller chunks
                if len(text) > 300:  # If text is very long
                    sentences = re.split(r'(?<=[.!?。！？])\s+', text)
                    chunks = []
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 300:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Queue each chunk separately
                    for chunk in chunks:
                        self.tts_manager.add_to_queue(chunk)
                else:
                    self.tts_manager.add_to_queue(text)
                    
                # Enable clear button when there's text in the queue
                self.tts_clear_button.setEnabled(True)
        except Exception as e:
            logging.error(f"TTS error in translation: {str(e)}")

        # Rest of the existing translation display code...
        if self.transcriber_mode == RecordingTranscriberMode.APPEND_BELOW:
            self.translation_text_box.moveCursor(QTextCursor.MoveOperation.End)
            if len(self.translation_text_box.toPlainText()) > 0:
                self.translation_text_box.insertPlainText("\n\n")
            self.translation_text_box.insertPlainText(self.strip_newlines(text))
            self.translation_text_box.moveCursor(QTextCursor.MoveOperation.End)
            
            if self.export_enabled:
                with open(self.translation_export_file, "a") as f:
                    f.write(text + "\n\n")

        elif self.transcriber_mode == RecordingTranscriberMode.APPEND_ABOVE:
            self.translation_text_box.moveCursor(QTextCursor.MoveOperation.Start)
            self.translation_text_box.insertPlainText(self.strip_newlines(text))
            self.translation_text_box.insertPlainText("\n\n")
            self.translation_text_box.moveCursor(QTextCursor.MoveOperation.Start)

            if self.export_enabled:
                with open(self.translation_export_file, "r") as f:
                    existing_content = f.read()

                new_content = text + "\n\n" + existing_content

                with open(self.translation_export_file, "w") as f:
                    f.write(new_content)

        elif self.transcriber_mode == RecordingTranscriberMode.APPEND_AND_CORRECT:
            self.process_transcription_merge(text, self.translations, self.translation_text_box, self.translation_export_file)

    def stop_recording(self):
        if self.tts_manager:
            self.tts_manager.set_enabled(False)
            self.tts_checkbox.setChecked(False)
            self.tts_clear_button.setEnabled(False)
            self.tts_skip_button.setEnabled(False)
            
        if self.transcriber is not None:
            self.transcriber.stop_recording()

        if self.translator is not None:
            self.translator.stop()

        self.record_button.setDisabled(True)


    def on_transcriber_finished(self):
        self.reset_record_button()

    def on_transcriber_error(self, error: str):
        self.reset_record_button()
        self.set_recording_status_stopped()
        QMessageBox.critical(
            self,
            "",
            _("An error occurred while starting a new recording:")
            + error
            + ". "
            + _(
                "Please check your audio devices or check the application logs for more information."
            ),
        )

    def on_cancel_model_progress_dialog(self):
        if self.model_loader is not None:
            self.model_loader.cancel()
        self.reset_model_download()
        self.set_recording_status_stopped()
        self.record_button.setDisabled(False)

    def reset_model_download(self):
        if self.model_download_progress_dialog is not None:
            self.model_download_progress_dialog.canceled.disconnect(
                self.on_cancel_model_progress_dialog
            )
            self.model_download_progress_dialog.close()
            self.model_download_progress_dialog = None

    def reset_recording_controls(self):
        # Clear text box placeholder because the first chunk takes a while to process
        self.transcription_text_box.setPlaceholderText("")
        self.reset_record_button()
        self.reset_model_download()

    def reset_record_button(self):
        self.record_button.setEnabled(True)

    def on_recording_amplitude_changed(self, amplitude: float):
        self.audio_meter_widget.update_amplitude(amplitude)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.tts_manager:
            self.tts_manager.stop()
            
        if self.tts_health_timer:
            self.tts_health_timer.stop()
            
        if self.model_loader is not None:
            self.model_loader.cancel()

        self.stop_recording()
        if self.recording_amplitude_listener is not None:
            self.recording_amplitude_listener.stop_recording()
            self.recording_amplitude_listener.deleteLater()
            self.recording_amplitude_listener = None

        if self.translator is not None:
            self.translator.stop()

        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_LANGUAGE,
            self.transcription_options.language,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_TASK, self.transcription_options.task
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_TEMPERATURE,
            self.transcription_options.temperature,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_INITIAL_PROMPT,
            self.transcription_options.initial_prompt,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_MODEL, self.transcription_options.model
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_ENABLE_LLM_TRANSLATION,
            self.transcription_options.enable_llm_translation,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_LLM_MODEL,
            self.transcription_options.llm_model,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_LLM_PROMPT,
            self.transcription_options.llm_prompt,
        )

        return super().closeEvent(event)