import os
import warnings
import os
import re
warnings.filterwarnings("ignore")

import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence

SAFE_PATH = str(os.getenv('HOME')) + "/Coding/NTUA Dev/Thesis Umbrella"
CHA_PATH = os.path.join(SAFE_PATH, "CHA")
OUTPUT_PATH = CHA_PATH
SCIENTISTS = ["Lu", "Pitt"]
CATEGORIES = ["Control", "Dementia"]

INPUT_PATHS = [os.path.join(CHA_PATH, scientist, category) if "Pitt" not in scientist
               else os.path.join(CHA_PATH, scientist, category, "cookie")
               for scientist in SCIENTISTS for category in CATEGORIES] \
               + [os.path.join(CHA_PATH, "Pitt", category, "fluency") for category in CATEGORIES] \
               + [os.path.join(CHA_PATH, "Pitt", category, "sentence") for category in CATEGORIES] \
               + [os.path.join(CHA_PATH, "Pitt", category, "recall") for category in CATEGORIES]

PATTERNS = {
        # CHAT Code       : Replacement
        r'[^\x20-\x7E]'   : '',

        r'\[/\d*\]'       : '{CHA REPETITION}',
        r'\[//\]'         : '{CHA RETRACING}',
        r'\(\.\)'         : '{CHA SHORT PAUSE}',
        r'\(\.\.\)'       : '{CHA MEDIUM PAUSE}',
        r'\(\.\.\.\)'     : '{CHA LONG PAUSE}',
        r'\+\.\.\.'       : '{CHA TRAILING OFF}',
        r'&\+[a-zA-Z]*\s' : '{CHA PHONOLOGICAL FRAGMENT}',
    r'&\*[A-Z]{3}:[^\s]+' : '{CHA INTERPOSED WORD}',
        r'&\-[a-zA-Z]*\s' : '{CHA FILLER}',
        r'\([a-zA-Z]+\)'  : '{CHA NON COMPLETION OF WORD}',
        r'&=belches'      : '{CHA BELCHES}',
        r'&=hisses'       : '{CHA HISSES}',
        r'&=grunts'       : '{CHA GRUNTS}',
        r'&=whines'       : '{CHA WHINES}',
        r'&=coughs'       : '{CHA COUGHS}',
        r'&=hums'         : '{CHA HUMS}',
        r'&=roars'        : '{CHA ROARS}',
        r'&=whistles'     : '{CHA WHISTLES}',
        r'&=cries'        : '{CHA CRIES}',
        r'&=laughs'       : '{CHA LAUGHS}',
        r'&=sneezes'      : '{CHA SNEEZES}',
        r'&=whimpers'     : '{CHA WHIMPERS}',
        r'&=gasps'        : '{CHA GASPS}',
        r'&=moans'        : '{CHA MOANS}',
        r'&=sighs'        : '{CHA SIGHS}',
        r'&=yawns'        : '{CHA YAWNS}',
        r'&=groans'       : '{CHA GROANS}',
        r'&=mumbles'      : '{CHA MUMBLES}',
        r'&=sings'        : '{CHA SINGS}',
        r'&=yells'        : '{CHA YELLS}',
        r'&=growls'       : '{CHA GROWLS}',
        r'&=pants'        : '{CHA PANTS}',
        r'&=squeals'      : '{CHA SQUEALS}',
        r'&=vocalizes'    : '{CHA VOCALIZES}',
        r'\+\.\.\?'       : '{CHA TRAILING OFF QUESTION}',
        r'\+\!\?'         : '{CHA QUESTION WITH EXCLAMATION}',
        r'\+\/\.'         : '{CHA INTERRUPTION}',
        r'\+\/\?'         : '{CHA INTERRUPTION OF QUESTION}',
        r'\+\/\/\.'       : '{CHA SELF-INTERRUPTION}',
        r'\+\/\/\?'       : '{CHA SELF-INTERRUPTED QUESTION}',

        r'\([^)]*\)'      : "",
        r'(\w)\1\1'       : '',
        r'\[.*?\]'        : "",
        r'&-(\w+)'        : r'\1',
        r'&+(\w+)'        : r'\1',
        r'<(\w+)>'        : r'\1',
        r'\+...'          : "",
        r'\s+'            : ' ',
     r'[^A-Za-z\n \'{}?]' : '',
        r'\{'             : '[',
        r'\}'             : ']',
}

CHA_TOKENS = [

              '[CHA REPETITION]',
              '[CHA RETRACING]',
              '[CHA SHORT PAUSE]',
              '[CHA MEDIUM PAUSE]',
              '[CHA LONG PAUSE]',
              '[CHA TRAILING OFF]',
              '[CHA PHONOLOGICAL FRAGMENT]',
              '[CHA INTERPOSED WORD]',
              '[CHA FILLER]',
              '[CHA NON COMPLETION OF WORD]',
              '[CHA BELCHES]',
              '[CHA HISSES]',
              '[CHA GRUNTS]',
              '[CHA WHINES]',
              '[CHA COUGHS]',
              '[CHA HUMS]',
              '[CHA ROARS]',
              '[CHA WHISTLES]',
              '[CHA CRIES]',
              '[CHA LAUGHS]',
              '[CHA SNEEZES]',
              '[CHA WHIMPERS]',
              '[CHA GASPS]',
              '[CHA MOANS]',
              '[CHA SIGHS]',
              '[CHA YAWNS]',
              '[CHA GROANS]',
              '[CHA MUMBLES]',
              '[CHA SINGS]',
              '[CHA YELLS]',
              '[CHA GROWLS]',
              '[CHA PANTS]',
              '[CHA SQUEALS]',
              '[CHA VOCALIZES]',
              '[CHA TRAILING OFF QUESTION]',
              '[CHA QUESTION WITH EXCLAMATION]',
              '[CHA INTERRUPTION]',
              '[CHA INTERRUPTION OF QUESTION]',
              '[CHA SELF-INTERRUPTION]',
              '[CHA SELF-INTERRUPTED QUESTION]',

              ]

class ProcessingConfig:
    def __init__(self):
        self.input_paths = INPUT_PATHS
        self.output_path = OUTPUT_PATH
        self.patterns = PATTERNS

    def __repr__(self):
        return "Config Object for Processing CHA Files"

    def __str__(self):
        return "Config Object for Processing CHA Files"

    def __dict__(self):
        return {"input_paths": self.input_paths, "output_path": self.output_path, "patterns": self.patterns}

    def get_input_paths(self):
        return self.input_paths

    def get_output_path(self):
        return self.output_path

    def get_patterns(self):
        return self.patterns


def remove_fillers(text, fillers=None, replacement="[CHA FILLER]"):
    if fillers is None:
        fillers = ["uhm", "uh", "ah", "umm", "like"]
    for filler in fillers:
        text = text.replace(filler, replacement)
    return text

def replace_repeating_word(text):
    words = text.split()
    new_words = []
    for i in range(len(words)):
        if words[i] == words[i-1]:
            new_words.append("[CHA REPEATING]")
        if i == 0 or words[i] != words[i-1]:
            new_words.append(words[i])
    return " ".join(new_words)

def detect_silence(audio_path, min_silence_len=1000, silence_thresh=-40, token="[CHA SHORT PAUSE]"):
    model = whisper.load_model("base")
    audio = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    full_transcript = []
    for i, chunk in enumerate(chunks):
        chunk.export(f"temp_chunk_{i}.wav", format="wav")
        result = model.transcribe(f"temp_chunk_{i}.wav")
        full_transcript.append(result["text"])
        if i < len(chunks) - 1:
            full_transcript.append(token)

    os.system("rm temp_chunk_*.wav")
    return " ".join(full_transcript)

def condense_silence(text, token="[CHA SHORT PAUSE]"):
    wordl = re.split(r'(\[CHA SHORT PAUSE\])', text)
    wordl = [word for word in wordl if word]
    words = []
    for word in wordl:
        if word == ' ':
            continue
        words.append(word)
    new_words = []
    count = 0
    i = 0

    while i < len(words):
        if words[i] == token:
            count += 1
            if i + 1 < len(words) and words[i + 1] == token:
                i += 1
                continue
            if count == 2:
                new_words.append("[CHA MEDIUM PAUSE]")
            elif count >= 3:
                new_words.append("[CHA LONG PAUSE]")
            else:
                new_words.append("[CHA SHORT PAUSE]")
            count = 0
        else:
            new_words.append(words[i])
        i += 1
    return " ".join(new_words)
