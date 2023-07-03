import os
import sys
from machine_learning_transcripts import train_test_model


def extra_print(text):
    sys.stout.write(text + "\n")
    print(text)


outpath = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena/results"

path = "/Users/athena.kam/Documents/Thesis/codebase/thesis-2023-athena/datasets/transformed/google"
print(path)
asr = path.split("/")[-1]
directory = os.fsencode(path)
os.chdir(path)

model_names = ["svc", "lr", "knn", "rf"]
reducers = ["pca", "lda", "mrmr"]

out_filename = "results_transcripts_" + asr + ".csv"
out_filepath = os.path.join(outpath, out_filename)

sys.stout = open(out_filepath, "w")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        for model_name in model_names:
            for reducer in reducers:
                extra_print("___________________________________________")
                extra_print(f"Filename:{filename}")
                extra_print(f"Model :{model_name}")
                extra_print(f"Reducer:{reducer}")
                extra_print(f"Sentence?:{'sentence' in filename}")
                extra_print("___________________________________________")
                train_test_model(
                    filename=filename, model_name=model_name, reduce=reducer
                )

sys.stdout.close
