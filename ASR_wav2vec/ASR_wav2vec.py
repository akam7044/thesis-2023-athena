#Importing all the necessary packages
import pandas as pd
import os
import nltk
import librosa
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
#nltk.download("punkt")

# Constants 
MAX_TOKEN = 100000

#Loading the pre-trained model and the tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Directory drama 
curr_dir = os.getcwd()
directory_in_str = "dataset/26-29_09_2017_KCL/ReadTextOnly"
directory = os.fsencode(directory_in_str)

id_list = []
transcript_list = []
classification_list = []

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

# Corrects to sentance case
def correct_casing(input_sentence):
  sentences = nltk.sent_tokenize(input_sentence)
  return (' '.join([s.replace(s[0],s[0].capitalize(),1) for s in sentences]))

# Batch a vector 
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# Audio to transcript
def asr_transcript(input_file,MAX_TOKEN):
    speech_vector = []
    full_transcription = []

    speech = load_data(input_file)

    if len(speech)>MAX_TOKEN:
        for x in batch(speech,MAX_TOKEN):
            speech_vector.append(x)
    else:
        speech_vector.append(speech)

    for n in range(0,len(speech_vector)-1):
        #Tokenize
        input_values = tokenizer(speech_vector[n], return_tensors="pt").input_values
        #Take logits
        logits = model(input_values).logits
        #Take argmax
        predicted_ids = torch.argmax(logits, dim=-1)
        #Get the words from predicted word ids
        transcription = tokenizer.decode(predicted_ids[0])
        #Correcting the letter casing
        transcription = correct_casing(transcription.lower())

        full_transcription.append(transcription)

    
    return ' '.join(full_transcription)

# Try 
for folder in os.listdir(directory):
    foldername = os.fsdecode(folder)
    
    if foldername == 'HC':
        folderpathname = os.path.join(directory_in_str,'HC')
        classification = 0
    elif foldername == 'PD':
        folderpathname = os.path.join(directory_in_str,'PD')
        classification = 1
    else:
        continue
        
    folderpath = os.fsencode(folderpathname)
    for file in os.listdir(folderpath):
        filename = os.fsdecode(file)
        if filename.endswith('.wav'):
            filepath = os.path.join(folderpathname,filename)
            print('\nNow transcribing for: '+ filename)
            result = asr_transcript(filepath,MAX_TOKEN)
            print(result)
            id_list.append(filename)
            transcript_list.append(result)
            classification_list.append(classification)

# Create a pd df
d = {"id":id_list,"transcripts":transcript_list,"classification":classification_list}
df = pd.DataFrame(d)
saveAs = os.path.join(curr_dir,'readTextOnly_wav2vec.csv')
df.to_csv(saveAs,index=False)