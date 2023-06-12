'''
code from: https://cloud.google.com/speech-to-text/docs/samples/speech-quickstart#speech_quickstart-python
'''
from google.cloud import speech
import os
import pandas as pd
import librosa
from time import ctime
# Define sample rate to be 16kHz 
def load_data(input_file):
  #reading the file
  speech, sample_rate = librosa.load(input_file)

  #make it 1-D
  if len(speech.shape) > 1: 
      speech = speech[:,0] + speech[:,1]
  #Resampling the audio at 16KHz
  if sample_rate !=16000:
    speech = librosa.resample(speech, orig_sr=sample_rate,target_sr = 16000)
  return speech

def asr_transcript(gcs_uri):
    transcript_vector = []
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    from google.cloud import speech

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=44100,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete\nStart time:")
    print(ctime())
    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        transcript_vector.append(result.alternatives[0].transcript)
        # The first alternative is the most likely one for this portion.
        # print(f"Transcript: {result.alternatives[0].transcript}")
        # print(f"Confidence: {result.alternatives[0].confidence}")
    print("\nFinish time: ",ctime(),"\n")
    return " ".join(transcript_vector)

# Try 
gs_base_directory = "gs://athena_thesis/SpontaneousDialogueOnly_FLAC"
curr_dir = os.getcwd()
directory_in_str = "dataset/26-29_09_2017_KCL/SpontaneousDialogueOnly_FLAC"
directory = os.fsencode(directory_in_str)

id_list = []
transcript_list = []
classification_list = []

for folder in os.listdir(directory):
    foldername = os.fsdecode(folder)
    
    if foldername == 'HC':
        folderpathname = os.path.join(directory,b'HC')
        classification = 0
    elif foldername == 'PD':
        folderpathname = os.path.join(directory,b'PD')
        classification = 1
    else:
        continue
        
    folderpath = os.fsencode(folderpathname)
    gs_folderpath = gs_base_directory+'/'+foldername
    for file in os.listdir(folderpath):
        filename = os.fsdecode(file)
        if filename.endswith('.flac'):
            gs_filepath = gs_folderpath+'/'+filename
            print('\nNow transcribing for: '+ filename)
            result = asr_transcript(gs_filepath)
            print(result)
            id_list.append(filename)
            transcript_list.append(result)
            classification_list.append(classification)

# Create a pd df
d = {"id":id_list,"transcripts":transcript_list,"classification":classification_list}
df = pd.DataFrame(d)
saveAs = os.path.join(curr_dir,'spontaneousDialogueOnly_google.csv')
df.to_csv(saveAs,index=False)