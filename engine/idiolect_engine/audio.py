import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REFERENCE_RATE = 24000
# Longer sections tend to fade out toward the end.
SECTION_CHARACTERS = 200
# Each section's speech is brought to this loudness, within limits, so sections match.
TARGET_DB = -24.0
GAIN_LIMITS_DB = (-6.0, 12.0)
# Silence between sections: within a paragraph, and after one.
SENTENCE_PAUSE = 0.3
PARAGRAPH_PAUSE = 0.75
MAX_TAKE_SECONDS = 90
# Long enough to carry a speaking style, short enough to stay a quick lead-in;
# stretched, rarely, to reach the end of a sentence.
CLIP_SECONDS = (8, 15, 20)


def split_script(text: str, limit=SECTION_CHARACTERS):
    """
    Plan the sections to render, each with the pause that follows it. Paragraphs never share a
    section, so the model never reads across a paragraph break or a change of speaker; long ones
    split at sentences. Blank lines separate paragraphs; without any, each line is one.
    """
    if limit < 1:
        raise ValueError("Chunk limit must be positive")
    text = text.strip()
    paragraphs = re.split(r"\n\s*\n", text) if re.search(r"\n\s*\n", text) else text.split("\n")
    sections = []
    for paragraph in paragraphs:
        chunks = split_paragraph(re.sub(r"\s+", " ", paragraph).strip(), limit)
        sections += [{"text": chunk, "pause": SENTENCE_PAUSE} for chunk in chunks]
        if sections:
            sections[-1]["pause"] = PARAGRAPH_PAUSE
    if sections:
        sections[-1]["pause"] = 0.0
    return sections


def split_paragraph(text: str, limit):
    """Keep sentence boundaries where possible, and bound unpunctuated input too."""
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*", text)
    pieces = []
    for sentence in sentences:
        sentence = sentence.strip()
        while len(sentence) > limit:
            # Break a long sentence after a clause where possible, else between words.
            clause = max(
                sentence.rfind(mark, limit // 2, limit)
                for mark in (",", ";", ":", "，", " –", " —")
            )
            if clause >= 0:
                boundary = clause + 1
            else:
                boundary = sentence.rfind(" ", 0, limit + 1)
                if boundary < limit // 2:
                    boundary = limit
            pieces.append(sentence[:boundary].strip())
            sentence = sentence[boundary:].strip()
        if sentence:
            pieces.append(sentence)
    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > limit:
            chunks.append(current)
            current = ""
        current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "").replace("'", ""))


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, letter in enumerate(a, 1):
        current = [i]
        for j, other in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (letter != other))
            )
        previous = current
    return previous[-1]


def same_word(expected, heard):
    """The recognizer spells some words its own way; short words must match exactly."""
    if expected == heard:
        return True
    allowed = 2 if len(expected) >= 7 else 1 if len(expected) >= 4 else 0
    return (
        allowed > 0
        and abs(len(expected) - len(heard)) <= allowed
        and (edit_distance(expected, heard) <= allowed)
    )


def speech_end(text, words, size=3):
    """
    When the text's final words end in recognized speech, or None. The first place they appear,
    since the model sometimes trails off into extra syllables after the text. Mirrors the CLI's
    findEnding, which detects the end of a reading during setup.
    """
    ending = tokens(text)[-size:]
    heard = [(token, owner) for owner, word in enumerate(words) for token in tokens(word["text"])]
    if not ending:
        return None
    for at in range(len(heard)):
        if not same_word(ending[-1], heard[at][0]):
            continue
        matched, j = 0, at
        earliest = max(0, at - len(ending) - 2)
        for expected in reversed(ending):
            k = j
            while k >= earliest and not same_word(expected, heard[k][0]):
                k -= 1
            if k >= earliest:
                matched += 1
                j = k - 1
        if matched >= len(ending) - 1:
            return words[heard[at][1]]["end"]
    return None


def coverage(text, words):
    """The share of the text's words heard in order (longest common subsequence)."""
    expected = tokens(text)
    heard = [token for word in words for token in tokens(word["text"])]
    if not expected:
        return 1.0
    previous = [0] * (len(heard) + 1)
    for word in expected:
        current = [0]
        for j, other in enumerate(heard, 1):
            current.append(
                previous[j - 1] + 1 if same_word(word, other) else max(previous[j], current[j - 1])
            )
        previous = current
    return previous[-1] / len(expected)


def frame_energy(audio, sample_rate):
    """Loudness in 10 ms frames, and the frame length in samples."""
    frame = max(1, int(sample_rate * 0.01))
    frames = len(audio) // frame
    return np.sqrt(np.mean(audio[: frames * frame].reshape(frames, frame) ** 2, axis=1)), frame


def quiet_level(energy):
    """Below this, a frame is silence: soft consonants like "s" and "sh" stay above it."""
    return max(float(energy.max()) * 0.01, 1e-4) if len(energy) else 1e-4


def trim_silence(audio, sample_rate, before=0.12, after=0.1):
    """Cut leading and trailing silence, keeping a natural margin around soft first and last sounds."""
    energy, frame = frame_energy(audio, sample_rate)
    if len(energy) == 0:
        return audio
    loud = np.flatnonzero(energy > quiet_level(energy))
    if len(loud) == 0:
        return audio[:0]
    start = max(0, loud[0] * frame - int(before * sample_rate))
    end = min(len(audio), (loud[-1] + 1) * frame + int(after * sample_rate))
    return audio[start:end]


def decibels(audio):
    return 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-9) if len(audio) else -180.0


def speech_level(audio, sample_rate):
    """Loudness of the speech in a section, ignoring pauses."""
    energy, _ = frame_energy(audio, sample_rate)
    if len(energy) == 0 or energy.max() == 0:
        return -180.0
    speech = energy[energy > energy.max() * 0.1]
    return 20 * np.log10(np.sqrt(np.mean(speech**2)))


def ending_fade(audio, sample_rate, words, last=3):
    """How many decibels quieter a section's final words are than its typical word."""
    levels = [
        decibels(audio[int(word["start"] * sample_rate) : int(word["end"] * sample_rate)])
        for word in words
    ]
    if len(levels) <= last:
        return 0.0
    return float(np.median(levels) - np.mean(levels[-last:]))


def match_level(audio, sample_rate):
    """Bring a section's speech toward the target loudness, within the gain limits."""
    gain = np.clip(TARGET_DB - speech_level(audio, sample_rate), *GAIN_LIMITS_DB)
    return audio * float(10 ** (gain / 20))


def cut_after(audio, sample_rate, end, search=0.6, silence=0.06):
    """
    Cut after the speech that ends near `end` seconds. Recognized word timings are approximate,
    so cut at the first real silence after it, never inside a final consonant.
    """
    energy, frame = frame_energy(audio, sample_rate)
    level = quiet_level(energy)
    needed = int(silence * 100)
    first = int(end * 100)
    run = 0
    for index in range(first, min(len(energy), first + int(search * 100))):
        run = run + 1 if energy[index] < level else 0
        if run >= needed:
            return audio[: (index - needed + 1) * frame + int(0.05 * sample_rate)]
    # No silence before whatever follows: keep a generous margin instead.
    return audio[: min(len(audio), int((end + 0.3) * sample_rate))]


def read_take(source: Path):
    """Read one recorded script as mono, at its original rate, for training and clips."""
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as error:
        raise ValueError("A recording could not be read as audio.") from error
    audio = audio.mean(axis=1)
    duration = len(audio) / sample_rate
    if not 3 <= duration <= MAX_TAKE_SECONDS:
        raise ValueError(f"Each recording must last between 3 and {MAX_TAKE_SECONDS} seconds.")
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio**2)) < 0.0001:
        raise ValueError(
            "A recording is silent or too quiet. Try recording closer to the microphone."
        )
    return audio, sample_rate


# Titles that end with a full stop but not a sentence.
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "st.", "mt.", "jr.", "sr.", "vs.", "etc."}


def ends_sentence(text):
    text = text.rstrip("\"')”’")
    return text.endswith((".", "!", "?")) and text.split()[-1].lower() not in ABBREVIATIONS


def boundary(words, index):
    """How natural a place to end a clip: after a clause, then a pause."""
    text = words[index]["text"].rstrip("\"')”’")
    punctuation = 1 if text.endswith((",", ";", ":", "–", "—")) else 0
    gap = words[index + 1]["start"] - words[index]["end"] if index + 1 < len(words) else 1.0
    return punctuation + min(gap, 1.0)


def plan_clips(words, shortest=CLIP_SECONDS[0], longest=CLIP_SECONDS[1], stretch=CLIP_SECONDS[2]):
    """
    Split a take's timed words into clips that end at the end of a sentence. The model continues
    from a clip, so one ending mid-sentence makes it read new text as the rest of that sentence.
    """
    spans = []
    first = 0
    while first < len(words):

        def length(index, first=first):
            return words[index]["end"] - words[first]["start"]

        reachable = [i for i in range(first, len(words)) if length(i) <= stretch] or [first]
        sentences = [i for i in reachable if ends_sentence(words[i]["text"]) and length(i) >= 3]
        natural = [i for i in sentences if shortest <= length(i) <= longest]
        short = [i for i in sentences if length(i) < shortest]
        long = [i for i in sentences if length(i) > longest]
        if natural or short or long:
            end = max(natural) if natural else max(short) if short else min(long)
        else:
            # No sentence ends in reach: end at the most natural pause instead.
            within = [i for i in reachable if length(i) <= longest] or [first]
            end = max(within, key=lambda i: (boundary(words, i), i))
        if length(end) >= 3:
            spans.append((first, end))
        first = end + 1
    return spans


def cut_clips(audio, sample_rate, words):
    """Yield (24 kHz clip, its exact words) for each planned clip in a take."""
    for first, last in plan_clips(words):
        start = max(0.0, words[first]["start"] - 0.1)
        if first > 0:
            start = max(start, words[first - 1]["end"])
        end = words[last]["end"] + 0.25
        if last + 1 < len(words):
            end = min(end, words[last + 1]["start"])
        clip = audio[int(start * sample_rate) : int(end * sample_rate)]
        yield (
            resample(clip, sample_rate),
            " ".join(word["text"] for word in words[first : last + 1]),
        )


def resample(audio, sample_rate, rate=REFERENCE_RATE):
    if sample_rate == rate:
        return audio
    divisor = gcd(rate, sample_rate)
    return resample_poly(audio, rate // divisor, sample_rate // divisor).astype(np.float32)


def join_audio(paths, pauses, output: Path):
    """Join sections, each trimmed to its speech and level-matched, with the planned pause after each."""
    segments = []
    sample_rate = None
    for path, pause in zip(paths, pauses, strict=True):
        audio, sr = sf.read(path, dtype="float32")
        if sample_rate is not None and sr != sample_rate:
            raise ValueError("Generated sections have different sample rates")
        sample_rate = sr
        speech = match_level(trim_silence(audio, sr), sr)
        segments += [speech, np.zeros(int(sr * pause), dtype=np.float32)]
    if not any(len(segment) for segment in segments):
        raise ValueError("No audio was generated")
    combined = np.concatenate(segments)
    if not np.isfinite(combined).all():
        raise ValueError("The model produced invalid audio. Please try again.")
    peak = float(np.max(np.abs(combined)))
    if peak > 0.98:
        combined = combined * (0.98 / peak)
    sf.write(output, combined, sample_rate, subtype="PCM_16")
    return round(len(combined) / sample_rate, 2)
