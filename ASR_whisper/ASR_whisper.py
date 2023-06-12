import whisper
import os 
import pandas as pd

'''
whisper: https://github.com/openai/whisper
'''

model = whisper.load_model('base')
curr_dir = os.getcwd()
directory_in_str = 'dataset/26-29_09_2017_KCL/SpontaneousDialogueOnly'
directory = os.fsencode(directory_in_str)

id_list = []
transcript_list = []
classification_list = []

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
            result = model.transcribe(filepath,language = "en", fp16 = False)
            print(result['text'])

            id_list.append(filename)
            transcript_list.append(result['text'])
            classification_list.append(classification)

# Create a pd df
d = {"id":id_list,"transcripts":transcript_list,"classification":classification_list}
df = pd.DataFrame(d)
saveAs = os.path.join(curr_dir,'spontaneousDialougeOnly_whisper.csv')
df.to_csv(saveAs,index=False)
