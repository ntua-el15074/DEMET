CORE_PATH = "/core"
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
              # '[CHA NON COMPLETION OF WORD]',
              # '[CHA BELCHES]',
              # '[CHA HISSES]',
              # '[CHA GRUNTS]',
              # '[CHA WHINES]',
              # '[CHA COUGHS]',
              # '[CHA HUMS]',
              # '[CHA ROARS]',
              # '[CHA WHISTLES]',
              # '[CHA CRIES]',
              # '[CHA LAUGHS]',
              # '[CHA SNEEZES]',
              # '[CHA WHIMPERS]',
              # '[CHA GASPS]',
              # '[CHA MOANS]',
              # '[CHA SIGHS]',
              # '[CHA YAWNS]',
              # '[CHA GROANS]',
              # '[CHA MUMBLES]',
              # '[CHA SINGS]',
              # '[CHA YELLS]',
              # '[CHA GROWLS]',
              # '[CHA PANTS]',
              # '[CHA SQUEALS]',
              # '[CHA VOCALIZES]',
              # '[CHA TRAILING OFF QUESTION]',
              # '[CHA QUESTION WITH EXCLAMATION]',
              # '[CHA INTERRUPTION]',
              # '[CHA INTERRUPTION OF QUESTION]',
              # '[CHA SELF-INTERRUPTION]',
              # '[CHA SELF-INTERRUPTED QUESTION]',
            ]

class ModelBaseConfig:
    def __init__(self,model_name):
        self.model_name = model_name
        self.extra_tokens = CHA_TOKENS

    def get_model_name(self):
        return self.model_name

    def get_extra_tokens(self):
        return self.extra_tokens

