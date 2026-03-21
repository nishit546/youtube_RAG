import json
import logging
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "yt_chroma_db")


def _whisper_cache_path(video_id: str) -> str:
    folder = os.path.join(CACHE_DIR, video_id)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "whisper_transcript.json")


def _find_ffmpeg() -> str | None:
    """
    Find ffmpeg executable path, checking both the current PATH and the
    system-level PATH (so it works even if server started before ffmpeg install).
    Also checks common winget/chocolatey install locations on Windows.
    """
    # 1. Try current PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 2. Reload system + user PATH environment variables (Windows)
    try:
        import winreg
        def _get_reg_path(root, subkey, name):
            try:
                with winreg.OpenKey(root, subkey) as key:
                    return winreg.QueryValueEx(key, name)[0]
            except Exception:
                return ""
        sys_path = _get_reg_path(
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            "Path"
        )
        usr_path = _get_reg_path(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            "Path"
        )
        combined = sys_path + ";" + usr_path
        for directory in combined.split(";"):
            candidate = os.path.join(directory.strip(), "ffmpeg.exe")
            if os.path.isfile(candidate):
                return candidate
    except Exception:
        pass

    # 3. Common install locations as a last resort
    common_paths = [
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-full\bin\ffmpeg.exe"),
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path

    return None


def transcribe_with_whisper(video_id: str) -> list[dict]:
    """
    Download audio from YouTube and transcribe using Whisper.

    Returns:
        List of { "text": str, "start": float, "duration": float }
        — same format as YouTubeTranscriptApi entries.

    Raises:
        RuntimeError if yt-dlp or whisper are not installed,
        or if download / transcription fails.
    """
    whisper_cache = _whisper_cache_path(video_id)
    if os.path.exists(whisper_cache):
        logger.info(f"[whisper] Loading cached Whisper transcript for {video_id}")
        with open(whisper_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    # ── Check dependencies ────────────────────────────────────────────────────
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. Run: pip install yt-dlp  "
            "— needed for Whisper audio download."
        )

    try:
        import whisper  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper  "
            "— needed for speech-to-text fallback."
        )

    import whisper as whisper_lib
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        # ── Find ffmpeg (works even if installed after server start) ──────────
        ffmpeg_path = _find_ffmpeg()
        logger.info(f"[whisper] ffmpeg path: {ffmpeg_path}")

        # ── Download audio ────────────────────────────────────────────────────
        logger.info(f"[whisper] Downloading audio for {video_id}")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(tmpdir, "audio.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "96",
            }],
            "quiet": True,
        }
        if ffmpeg_path:
            # Pass the directory containing ffmpeg, not the full path to the binary
            ydl_opts["ffmpeg_location"] = os.path.dirname(ffmpeg_path)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            err_str = str(e).lower()
            if "ffprobe" in err_str or "ffmpeg" in err_str:
                raise RuntimeError(
                    "ffmpeg is not installed or not on PATH. "
                    "Install it with: winget install Gyan.FFmpeg  "
                    "(then restart your terminal so PATH is updated)"
                )
            raise RuntimeError(f"Audio download failed for {video_id}: {e}")



        # ── Transcribe with Whisper ───────────────────────────────────────────
        logger.info(f"[whisper] Transcribing audio for {video_id} (this may take a minute)")
        model = whisper_lib.load_model("base")
        result = model.transcribe(audio_path, verbose=False)

        # Convert Whisper segments → YouTubeTranscriptApi-compatible format
        entries = [
            {
                "text": seg["text"].strip(),
                "start": seg["start"],
                "duration": seg["end"] - seg["start"],
            }
            for seg in result.get("segments", [])
            if seg["text"].strip()
        ]

    # ── Cache the result ──────────────────────────────────────────────────────
    with open(whisper_cache, "w", encoding="utf-8") as f:
        json.dump(entries, f)

    logger.info(f"[whisper] Transcription complete for {video_id}: {len(entries)} segments")
    return entries
