"""Audio alarm backed by ``pygame.mixer``.

The alarm loops a sound file while the driver is judged drowsy and stops when
they recover. Audio can be disabled entirely (e.g. ``--no-audio`` or on a
headless machine); in that case the alarm becomes a no-op that still tracks its
logical on/off state.
"""

from __future__ import annotations

from pathlib import Path


class AudioAlarm:
    """A looping audio alarm with a graceful no-audio fallback."""

    def __init__(self, sound_path: str, enabled: bool = True) -> None:
        """Initialise the mixer and load the alarm sound.

        Args:
            sound_path: Path to a ``.wav``/``.ogg`` alarm sound.
            enabled: When ``False`` the alarm is a no-op (no mixer is opened).
        """
        self._enabled = enabled
        self._active = False
        self._sound: object | None = None

        if not self._enabled:
            return

        try:
            import pygame
        except ImportError:
            self._enabled = False
            return

        try:
            pygame.mixer.init()
        except pygame.error:
            # No audio device available (e.g. headless host): degrade quietly.
            self._enabled = False
            return

        if Path(sound_path).is_file():
            self._sound = pygame.mixer.Sound(sound_path)
        else:
            self._enabled = False

    @property
    def active(self) -> bool:
        """Whether the alarm is currently sounding."""
        return self._active

    def start(self) -> None:
        """Begin looping the alarm if it is not already active."""
        if not self._enabled or self._active or self._sound is None:
            self._active = self._enabled and self._sound is not None
            return
        self._sound.play(loops=-1)  # type: ignore[attr-defined]
        self._active = True

    def stop(self) -> None:
        """Silence the alarm if it is sounding."""
        if not self._enabled or not self._active or self._sound is None:
            self._active = False
            return
        self._sound.stop()  # type: ignore[attr-defined]
        self._active = False

    def close(self) -> None:
        """Stop the alarm and shut down the mixer."""
        self.stop()
        if not self._enabled:
            return
        import pygame

        pygame.mixer.quit()
