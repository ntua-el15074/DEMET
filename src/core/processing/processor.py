import os
import tqdm
import re
import pandas as pd
# from logger.logger import Logger
from core.processing.config import remove_fillers, replace_repeating_word, detect_silence, condense_silence


# logger = Logger(True)

class Processor:
    def __init__(self, ProcessingConfig):
        """
        input_paths: list of paths to input files
        output_path: path to save output files
        patterns: dictionary of patterns to replace in the text
        """
        # logger.print_and_log("Processor object created.", "green")
        self.input_paths = ProcessingConfig.get_input_paths()
        self.output_path = ProcessingConfig.get_output_path()
        self.patterns = ProcessingConfig.get_patterns()
        # logger.print(str(self.input_paths))

    def __str__(self):
        return "Processor Object for processing CHA files."

    def __repr__(self):
        return "Processor Object for processing CHA files."

    def __dict__(self):
        return {
            "input_paths": self.input_paths,
            "output_path": self.output_path,
            "patterns": self.patterns
        }

    def clean_cha_text(self, text):
        pattern = re.search(r"(\d+_\d+)", text)
        start, end = 0, 0
        duration = 0
        if pattern:
            start, end = map(int, pattern.group(1).split('_'))
            duration = end - start
            text = re.sub(pattern.group(1), '', text)
        text = re.sub(rf"\**PAR:", '', text)
        for pattern, replacement in self.patterns.items():
            text = re.sub(pattern, replacement, text)
        return duration, text


    def process(self,key="long"):
        os.makedirs(self.output_path, exist_ok=True)
        # if os.path.exists(os.path.join(self.output_path,"data.csv")):
            # logger.print_and_log("Data file already exists. Skipping processing.", "green")
        #     if os.path.exists(os.path.join(self.output_path, "train.csv")) and os.path.exists(os.path.join(self.output_path, "val.csv")) and os.path.exists(os.path.join(self.output_path, "test.csv")):
        #         logger.print_and_log("Train, val and test files already exist. Skipping splitting.", "green")
        #     else:
        #         self.split_data()
        #     return
        # logger.print_and_log("Processing CHA files.", "green")
        with open(os.path.join(self.output_path,"data.csv"), "w") as output_file:
            output_file.write("text,gt\n")
            for path in tqdm.tqdm(self.input_paths, desc="Processing CHA files"):
                for file in os.listdir(path):
                    if file.endswith(".cha"):
                        if key == "short":
                           # total_text = ""
                           # total_duration = 0
                           gt = 1 if "Dementia" in path else 0
                           with open(os.path.join(path, file), 'r') as f:
                               input_stream = f.read()
                               input_stream = re.sub(r"\r\n|\n|((\*|\%|\@)[A-Za-z]+\:)", r" \n\1", input_stream)
                               input_stream = input_stream.split("\n")
                               for split in input_stream:
                                   if re.match(rf"\*PAR:", split):
                                       _, text = self.clean_cha_text(split)
                                       text = text.strip()
                                       lt = text.split(" ")
                                       if len(lt) > 5:
                                           output_file.write(f"{text},{gt}\n")
                                       # total_text += text + " "
                               # output_file.write(f"{total_text},{gt}\n")
                       # elif key == "medium":
                        elif key == "medium":
                           total_text = ""
                           # total_duration = 0
                           gt = 1 if "Dementia" in path else 0
                           with open(os.path.join(path, file), 'r') as f:
                                input_stream = f.read()
                                input_stream = re.sub(r"\r\n|\n|((\*|\%|\@)[A-Za-z]+\:)", r" \n\1", input_stream)
                                input_stream = input_stream.split("\n")
                                for split in input_stream:
                                    if re.match(rf"\*PAR:", split):
                                        _, text = self.clean_cha_text(split)
                                        text = text.strip()
                                        lt = text.split(" ")
                                        # if len(lt) > 5:
                                        #     output_file.write(f"{text},{gt}\n")
                                        total_text += text + " "
                                        if len(total_text.split(" ")) > 20:
                                            output_file.write(f"{total_text},{gt}\n")
                                            total_text = ""
                                        else:
                                            continue
                                if len(total_text.split(" ")) > 5:
                                  output_file.write(f"{total_text},{gt}\n")


                                    # output_file.write(f"{total_text},{gt}\n")
                            # elif key == "medium":

                        else:
                            total_text = ""
                            # total_duration = 0
                            gt = 1 if "Dementia" in path else 0
                            with open(os.path.join(path, file), 'r') as f:
                                input_stream = f.read()
                                input_stream = re.sub(r"\r\n|\n|((\*|\%|\@)[A-Za-z]+\:)", r" \n\1", input_stream)
                                input_stream = input_stream.split("\n")
                                for split in input_stream:
                                    if re.match(rf"\*PAR:", split):
                                        _, text = self.clean_cha_text(split)
                                        text = text.strip()
                                        lt = text.split(" ")
                                        # if len(lt) > 5:
                                            # output_file.write(f"{text},{gt}\n")
                                        total_text += text + " "
                                output_file.write(f"{total_text},{gt}\n")

            self.split_data()
            # logger.print_and_log("CHA files processed.", "green")

    def split_data(self):
        # logger.print_and_log("Shuffling and splitting.", "green")
        df = pd.read_csv(os.path.join(self.output_path, "data.csv"))
        # df = df.sample(frac=1).reset_index(drop=True)
        df_balanced = pd.concat([
            df[df['gt'] == 0].sample(n=1200, random_state=42),
            df[df['gt'] == 1].sample(n=1200, random_state=42)
        ])
        df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
        df_balanced.to_csv(os.path.join(self.output_path, "shuffled_data.csv"), index=False)

    def transribe_audio(self, audio_path):
        transcription = detect_silence(audio_path)
        transcription = condense_silence(transcription)
        transcription = replace_repeating_word(transcription)
        transcription = remove_fillers(transcription)
        return transcription
