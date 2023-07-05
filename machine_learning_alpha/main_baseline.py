import os
import sys
from machine_learning import train_test_model


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


outpath = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena/results"

path = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena/datasets/readtext_concat"

directory = os.fsencode(path)
os.chdir(path)

model_names = ["svc", "lr", "knn", "rf"]

out_filename = "results_audio_data_readTextConcat.csv"
out_filepath = os.path.join(outpath, out_filename)

sys.stout = open(out_filepath, "w")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        chunked = "chunk" in filename

        for model_name in model_names:
            extra_print("___________________________________________")
            extra_print(f"Filename:{filename}")
            extra_print(f"Model :{model_name}")
            extra_print(f"Chunked? :{chunked}")
            extra_print("___________________________________________")
            train_test_model(
                filename=filename,
                model_name=model_name,
                isTranscript=False,
                chunked=chunked,
            )

sys.stdout.close
