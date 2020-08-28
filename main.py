
import threading, requests, time
import io
import os
from google.cloud import speech

from google.cloud.speech import enums
from google.cloud.speech import types
import pandas as pd
from konlpy.tag import Twitter

from collections import Counter
import argparse
from moviepy.editor import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import wave
from array import array
from pydub import AudioSegment
from pydub.playback import play
twitter = Twitter()

xx=pd.read_csv("통합 문서1.csv",names = ['욕','빈칸'],encoding='cp949')
vect=CountVectorizer(tokenizer=twitter.nouns)
vect.fit(xx['욕'])

CHANNELS=2


videoclip = VideoFileClip("chulgoo4.mp4")
audioclip = videoclip.audio
audioclip.to_audiofile('chulgoo4_음성.wav')
wav_file = "chulgoo4_음성.wav"

with io.open(wav_file, 'rb') as audio_file:
    content = audio_file.read()
audio = types.RecognitionAudio(content=content)
config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=44100, audio_channel_count=CHANNELS,
    language_code='ko-KR',
    enable_word_time_offsets=True)
client = speech.SpeechClient.from_service_account_json("slol-7847930dcfd0.json")
operation = client.long_running_recognize(config, audio)

print('Waiting for operation to complete...')
result = operation.result(timeout=90)

word_list = []
start_time_list = []
end_time_list = []
for k in result.results:
    alternatives = k.alternatives
    for alternative in alternatives:
        print('Transcript: {}'.format(alternative.transcript))
        print('Confidence: {}'.format(alternative.confidence))

    for word_info in alternative.words:
        wo = word_info.word
        word_list.append(wo)
        start_time = word_info.start_time
        end_time = word_info.end_time
        start_time_list.append(start_time.seconds + start_time.nanos * 1e-9)
        end_time_list.append(end_time.seconds + end_time.nanos * 1e-9)
        print('Word: {}, start_time: {}, end_time: {}'.format(
            wo,
            start_time.seconds + start_time.nanos * 1e-9,
            end_time.seconds + end_time.nanos * 1e-9))

sentences_tag = []
for sentence in word_list:
    morph = twitter.pos(sentence)
    sentences_tag.append(morph)
    print("morph: ",morph)

noun_adj_list = []
for sentence1 in sentences_tag:
    for word, tag in sentence1:
        if tag in ['Noun', 'Adjective']:
            noun_adj_list.append(word)
            print("noun_adj_list: ",word)

word_vect = vect.transform(noun_adj_list)
feature_list = []
print("word_vect", word_vect)
for i in word_vect.toarray():

    for j in range(0, len(i)):
        if i[j] == 1:
            feature_list.append(vect.get_feature_names()[j])
            print("vect.get_feature_names()[j]",vect.get_feature_names()[j])

sound1 = AudioSegment.from_file(wav_file)
print(feature_list)
print(word_list)
for i in range(0, len(word_list)):
    for j in range(0, len(feature_list)):
        if feature_list[j] in word_list[i]:
            print(word_list[i],"욕시작시간",start_time_list[i])
            print(word_list[i],"욕끝시간{}",end_time_list[i])
            sound2 = AudioSegment.from_file("beep-1.wav")
            if end_time_list[i] - start_time_list[i] > 2.5:
                #sound2 = sound2[start_time_list[i] * 1000:(end_time_list[i] * 1000)]
                sound1 = sound1.overlay(sound2, position=(1000 * start_time_list[i]) + 2500, gain_during_overlay=-1000)

            else:
                #sound2 = sound2[start_time_list[i] * 1000:(end_time_list[i] * 1000)]
                sound1 = sound1.overlay(sound2, position=1000 * start_time_list[i], gain_during_overlay=-1000)


sound1.export(wav_file, format='wav')


videoclip2 = videoclip.set_audio(AudioFileClip(wav_file))
videoclip2.write_videofile('Chulgoo4_filter.avi', codec='libx264')

