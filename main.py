import youtube_dl
import turicreate as tc
import os

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
}

URL = 'https://www.youtube.com/watch?v=W-Y6CSXM1lU'
NAME = 'test_audio.mp3'
DIR = 'cut_audio'

#Download audio from youtube
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])

#Rename downloaded audio
if os.path.exists(NAME):
    os.remove(NAME)
    
for filename in os.listdir():
    if filename.endswith('.mp3'):
        os.rename(filename, NAME)

#Cut audio file into multiple
if os.path.exists(DIR):
    os.system('rm {}/*'.format(DIR))
else:
    os.mkdir(DIR)
    
CMD = 'aubiocut -i {} -c -r 16000 -o {}'.format(NAME, DIR)
os.system(CMD)

#Load prediction model
model = tc.load_model('audio.model')

#Upload audio to turicreate
audio = tc.load_audio(DIR)
audio['name'] = audio['path'].apply(lambda x: os.path.basename(x))

#Rename audio files with predicted class
prediction = model.predict(audio)

for i in range(len(prediction)):
    name = os.path.basename(audio[i]['name'])
    if prediction[i] == None:
        os.remove(audio[i]['path'])
    else:
        tmp = tc.load_audio(audio[i]['path'])
        if model.classify(tmp)['probability'][0] > 0.8:
            print(model.classify(tmp)['probability'][0])
            os.rename('{}/{}'.format(DIR, name), '{}/{}.{}.wav'.format(DIR, name.split('.wav')[0], prediction[i]))
        else:
            os.remove(audio[i]['path'])
            
print('\nNumber of audio (probability > 0.8): {}\n'.format(len(os.listdir(DIR))))