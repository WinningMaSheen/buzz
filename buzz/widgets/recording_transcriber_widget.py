import os
import re
import enum
import logging
import datetime
import sounddevice
from enum import auto
from typing import Optional, Tuple, Any


from PyQt6.QtCore import QThread, Qt, QThreadPool
from PyQt6.QtGui import QTextCursor, QCloseEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QHBoxLayout, QMessageBox

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



import queue
import threading
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal
import pyttsx3

class TTSQueueManager(QObject):
    """
    Manages a queue of text-to-speech requests, ensuring they play in order
    and only one at a time.
    """
    tts_started = pyqtSignal(str)  # Emitted when TTS starts speaking a text
    tts_finished = pyqtSignal(str)  # Emitted when TTS finishes speaking a text
    tts_error = pyqtSignal(str)  # Emitted if there's an error

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.tts_queue = queue.Queue()
        self.engine = pyttsx3.init()
        self.is_speaking = False
        self.should_stop = False
        self.current_text = ""
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

        # Configure TTS engine
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Optional: Set a specific voice
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)  # Index 0 is usually default voice

    def add_to_queue(self, text: str) -> None:
        """Add text to the TTS queue."""
        if text and text.strip():
            self.tts_queue.put(text.strip())

    def _process_queue(self) -> None:
        """Main queue processing loop that runs in a separate thread."""
        while not self.should_stop:
            try:
                # Get the next text to speak (blocks until item is available)
                text = self.tts_queue.get(timeout=1.0)
                self.current_text = text
                self.is_speaking = True
                
                # Emit signal that we're starting to speak this text
                self.tts_started.emit(text)

                try:
                    # Speak the text
                    self.engine.say(text)
                    self.engine.runAndWait()
                    
                    # Emit signal that we finished speaking this text
                    self.tts_finished.emit(text)
                except Exception as e:
                    self.tts_error.emit(f"Error speaking text: {str(e)}")
                finally:
                    self.is_speaking = False
                    self.current_text = ""
                    self.tts_queue.task_done()

            except queue.Empty:
                # No items in queue, continue waiting
                continue
            except Exception as e:
                self.tts_error.emit(f"Queue processing error: {str(e)}")

    def clear_queue(self) -> None:
        """Clear all pending TTS requests."""
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except queue.Empty:
                break

    def stop(self) -> None:
        """Stop the TTS queue manager and cleanup."""
        self.should_stop = True
        self.clear_queue()
        if self.is_speaking:
            self.engine.stop()
        self.thread.join(timeout=2.0)
        self.engine.stop()

    def set_voice(self, voice_id: str) -> None:
        """Set the TTS voice by ID."""
        self.engine.setProperty('voice', voice_id)

    def set_rate(self, rate: int) -> None:
        """Set the speech rate (words per minute)."""
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """Set the speech volume (0.0 to 1.0)."""
        self.engine.setProperty('volume', volume)

    def get_available_voices(self) -> list:
        """Get list of available TTS voices."""
        return self.engine.getProperty('voices')

REAL_CHARS_REGEX = re.compile(r'\w')
NO_SPACE_BETWEEN_SENTENCES = re.compile(r'([.!?。！？])([A-Z])')

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

        # Initialize TTS Manager
        self.tts_manager = TTSQueueManager(self)
        self.tts_manager.tts_started.connect(self.on_tts_started)
        self.tts_manager.tts_finished.connect(self.on_tts_finished)
        self.tts_manager.tts_error.connect(self.on_tts_error)

        self.settings = Settings()
        self.transcriber_mode = list(RecordingTranscriberMode)[
            self.settings.value(key=Settings.Key.RECORDING_TRANSCRIBER_MODE, default_value=0)]

        # TTS Settings
        self.tts_enabled = self.settings.value(
            key=Settings.Key.TTS_ENABLED,
            default_value=True,
        )
        
        if self.tts_enabled:
            tts_rate = self.settings.value(
                key=Settings.Key.TTS_RATE,
                default_value=150,
            )
            tts_volume = self.settings.value(
                key=Settings.Key.TTS_VOLUME,
                default_value=0.9,
            )
            self.tts_manager.set_rate(tts_rate)
            self.tts_manager.set_volume(tts_volume)

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

        # Create TTS controls
        tts_group_box = QGroupBox("Text-to-Speech Settings", self)
        tts_layout = QVBoxLayout()

        # TTS Enable/Disable checkbox
        self.tts_enabled_checkbox = QCheckBox("Enable Text-to-Speech", self)
        self.tts_enabled_checkbox.setChecked(self.tts_enabled)
        self.tts_enabled_checkbox.toggled.connect(self.on_tts_enabled_changed)
        tts_layout.addWidget(self.tts_enabled_checkbox)

        # TTS Rate slider
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Speech Rate:"))
        self.tts_rate_slider = QSlider(Qt.Horizontal, self)
        self.tts_rate_slider.setMinimum(50)
        self.tts_rate_slider.setMaximum(300)
        self.tts_rate_slider.setValue(tts_rate)
        self.tts_rate_slider.valueChanged.connect(self.on_tts_rate_changed)
        rate_layout.addWidget(self.tts_rate_slider)
        tts_layout.addLayout(rate_layout)

        # TTS Volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.tts_volume_slider = QSlider(Qt.Horizontal, self)
        self.tts_volume_slider.setMinimum(0)
        self.tts_volume_slider.setMaximum(100)
        self.tts_volume_slider.setValue(int(tts_volume * 100))
        self.tts_volume_slider.valueChanged.connect(self.on_tts_volume_changed)
        volume_layout.addWidget(self.tts_volume_slider)
        tts_layout.addLayout(volume_layout)

        # TTS Voice selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice:"))
        self.tts_voice_combo = QComboBox(self)
        voices = self.tts_manager.get_available_voices()
        for voice in voices:
            self.tts_voice_combo.addItem(voice.name, voice.id)
        self.tts_voice_combo.currentIndexChanged.connect(self.on_tts_voice_changed)
        voice_layout.addWidget(self.tts_voice_combo)
        tts_layout.addLayout(voice_layout)

        tts_group_box.setLayout(tts_layout)

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
        layout.addWidget(tts_group_box)  # Add TTS controls
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

    def on_tts_enabled_changed(self, enabled: bool):
        self.tts_enabled = enabled
        self.settings.set_value(Settings.Key.TTS_ENABLED, enabled)

    def on_tts_rate_changed(self, value: int):
        self.tts_manager.set_rate(value)
        self.settings.set_value(Settings.Key.TTS_RATE, value)

    def on_tts_volume_changed(self, value: int):
        volume = value / 100.0
        self.tts_manager.set_volume(volume)
        self.settings.set_value(Settings.Key.TTS_VOLUME, volume)

    def on_tts_voice_changed(self, index: int):
        voice_id = self.tts_voice_combo.itemData(index)
        self.tts_manager.set_voice(voice_id)

    def on_tts_started(self, text: str):
        logging.debug(f"TTS started speaking: {text[:50]}...")

    def on_tts_finished(self, text: str):
        logging.debug(f"TTS finished speaking: {text[:50]}...")

    def on_tts_error(self, error: str):
        logging.error(f"TTS error: {error}")
        QMessageBox.warning(
            self,
            "TTS Error",
            f"An error occurred with text-to-speech: {error}"
        )

    # ... [Keep all existing methods from the original class]

    def on_next_translation(self, text: str, _: Optional[int] = None):
        if len(text) == 0:
            return

        # Add text to TTS queue if enabled
        if self.tts_enabled and text:
            self.tts_manager.add_to_queue(text)

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

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.model_loader is not None:
            self.model_loader.cancel()

        # Stop TTS
        if self.tts_manager:
            self.tts_manager.stop()

        self.stop_recording()
        if self.recording_amplitude_listener is not None:
            self.recording_amplitude_listener.stop_recording()
            self.recording_amplitude_listener.deleteLater()
            self.recording_amplitude_listener = None

        if self.translator is not None:
            self.translator.stop()

        # Save all settings
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_LANGUAGE,
            self.transcription_options.language,
        )
        self.settings.set_value(
            Settings.Key.RECORDING_TRANSCRIBER_TASK, 
            self.transcription_options.task
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
            Settings.Key.RECORDING_TRANSCRIBER_MODEL, 
            self.transcription_options.model
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
        self.settings.set_value(
            Settings.Key.TTS_ENABLED,
            self.tts_enabled,
        )
        self.settings.set_value(
            Settings.Key.TTS_RATE,
            self.tts_manager.engine.getProperty('rate'),
        )
        self.settings.set_value(
            Settings.Key.TTS_VOLUME,
            self.tts_manager.engine.getProperty('volume'),
        )

        return super().closeEvent(event)