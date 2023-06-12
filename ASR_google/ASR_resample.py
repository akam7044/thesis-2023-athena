from pydub import AudioSegment
import os

# Directory drama 
curr_dir = os.getcwd()
directory_in_str = "dataset/26-29_09_2017_KCL/SpontaneousDialogueOnly"
directory = os.fsencode(directory_in_str)
out_dir = "dataset/26-29_09_2017_KCL/SpontaneousDialogueOnly_FLAC"

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
    pd_dir = os.path.join(out_dir,'PD')
    os.mkdir(pd_dir)
    hc_dir = os.path.join(out_dir,'HC')
    os.mkdir(hc_dir)
    


def wav2flac(filepath,filename,out_folderpathname):
    filename_flac = filename.replace("wav","flac")
    export_filepath = os.path.join(out_folderpathname,filename_flac)
    if not os.path.isfile(export_filepath):
        audio = AudioSegment.from_wav(filepath)
        audio.export(export_filepath,format="flac")

for folder in os.listdir(directory):
    foldername = os.fsdecode(folder)
    
    if foldername == 'HC':
        folderpathname = os.path.join(directory_in_str,'HC')
        out_folderpathname = os.path.join(out_dir,'HC')
    elif foldername == 'PD':
        folderpathname = os.path.join(directory_in_str,'PD')
        out_folderpathname = os.path.join(out_dir,'PD')
    else:
        continue
        
    folderpath = os.fsencode(folderpathname)
    #out_folderpath = os.fsencode(out_folderpathname)
    for file in os.listdir(folderpath):
        filename = os.fsdecode(file)
        if filename.endswith('.wav'):
            filepath = os.path.join(folderpathname,filename)
            print('\nNow converting to FLAC for: '+ filename)
            result = wav2flac(filepath,filename,out_folderpathname)
            


# song = AudioSegment.from_wav("dataset/26-29_09_2017_KCL/ReadTextOnly/PD/ID27_pd_4_1_1_NorthWind.wav")
# song.export("testme2.flac",format = "flac")