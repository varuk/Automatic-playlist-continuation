# %% [code]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import re
from pickle import dump, dumps, load, loads
import math

# %% [code]
import gensim
from gensim.models import KeyedVectors
filename = '../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True) #word2vec model

# %% [code]
vocab = set(model.wv.vocab) #word2vec vocabulary

# %% [code]
documents = dict() #a dictionary of pid vs tracks statistics(ie term frequency)
dictionary = dict() #a dictionary of track and words vs their document frequency
titles = set() #a set of titles vocabulary
playlists = dict() #a dictionary of pid vs tracks list
lendocs = dict() #a dictionary of pid vs document lengths
avglen = 0 #average document length
df = pd.DataFrame() #a dataframe for storing metadata

# %% [code]

def findbyknn(query):
    # vector will be of 'collaborative','duration_ms','modified_at','num_albums','num_artist','num_edits','num_followers','num_tracks', name vector of 300. 
    query_vector = getknnvector(query)
    relevance = dict()
    for i in range(len(documents.keys())):
        data = df.loc[i]
        pid = data[0]
        playlist_vector = getknnvector(data[1:])
        sim = np.dot(playlist_vector, query_vector)/ (np.linalg.norm(playlist_vector)*np.linalg.norm(query_vector))  #cosine similarity
        tracks = playlists[pid]
        for track in tracks: 
            relevance[track] = relevance.get(track,0) + sim
    top_tracks = sorted(relevance.keys(), key=lambda x: (relevance[x]),reverse= True)
    return top_tracks[:500]
def getknnvector(playlist):
    name = playlist.get('name',"")
    vector = np.zeros(300)
    if name: #getting sentence representation by averaging individual word representations
        words = name.split()
        for word in words:
            if word in vocab:
                vector = vector + np.array(model[word])
    collab = 0
    if playlist.get('collaborative', False):
        collab = 1
    vector2 = np.array([collab,
        playlist.get('duration_ms',0),
        playlist.get('modified_at',0),
        playlist.get('num_albums',0),
        playlist.get('num_artists',0),
        playlist.get('num_edits',0),
        playlist.get('num_followers',0),
        playlist.get('num_tracks',0),
        ])
    return np.concatenate((vector2,vector), axis=0) #getting a 308x1 vector for finding similarity
def preprocess(text):
    text = re.sub('[\'\-,\+\;<>/\!\"\(\)\{\}\?]', ' ',text).lower().strip()
    text = deEmojify(text)
    return text
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF" 
        u"\U0001F1E0-\U0001F1FF" 
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)
def readchallengeset():
    k=0
    testdf = pd.DataFrame()
    file = open('../input/mpd-data/spotify_million_playlist_dataset_challenge/challenge_set.json')
    playlists = json.loads(file.read())['playlists']
    for playlist in playlists:
        
        pid = int(playlist['pid'])
        data = pd.Series([
        playlist.get('collaborative',False),
        playlist.get('duration_ms',0),
        playlist.get('modified_at',0),
        escapefunc(playlist.get('name',"")),
        playlist.get('num_albums',0),
        playlist.get('num_artists',0),
        playlist.get('num_edits',0),
        playlist.get('num_followers',0),
        playlist.get('num_tracks',0),
        ])
        if len(playlist['tracks']) == 0:
            tracks = findbyknn(data) #finding recommendation via similarity based method
        else: 
            query = "_".join(preprocess(playlist.get('name',"")).split())    #forming query for query expansion
            for track in playlist['tracks']:  
                query = query + "_" + preprocess(track['track_name']).strip()
            tracks = findbyqe(query) #finding recommendation via query expansion based methods
        list1 = [pid]
        list1.extend(tracks)
        testdf = testdf.append(pd.Series(list1), ignore_index=True)
        if k%100==0:
            print(k)
        k = k + 1
    testdf.set_index(1)
    testdf.rename(inplace = True, columns ={0: "pid"})
    testdf = testdf.astype({"pid": 'int32'})
    return testdf
def readmpd():
    global avglen
    global df
    k = 0
    filenames = os.listdir('../input/mpd-data/spotify_million_playlist_dataset/data')
    
    columns = ['pid','collaborative','duration_ms','modified_at','name','num_albums','num_artist','num_edits','num_followers','num_tracks']       
    for filename in sorted(filenames):
        file = open('../input/mpd-data/spotify_million_playlist_dataset/data' + '/' + filename)
        playlistss = json.loads(file.read())['playlists']
        for playlist in playlistss:
            pid = playlist['pid']
            playlists[pid] = []
            documents[pid] = dict()
            data = pd.Series([           #forming pandas series of metadata
            playlist['pid'],
            playlist['collaborative'],
            playlist['duration_ms'],
            playlist['modified_at'],
            escapefunc(playlist['name']),
            playlist['num_albums'],
            playlist['num_artists'],
            playlist['num_edits'],
            playlist['num_followers'],
            playlist['num_tracks'],
            ])
            df = df.append(data, ignore_index=True)
            words = preprocess(playlist['name']).strip().split() 
            for word in words: #processing playlist titles 
                titles.add(word)
                lendocs[pid] = lendocs.get(pid,0) + 1
                documents[pid][word] = documents[pid].get(word,0) + 1
                if documents[pid][word] ==1:
                    dictionary[word] = dictionary.get(word,0) + 1
            for track in playlist['tracks']: #processing playlist tracks
                lendocs[pid] = lendocs.get(pid,0) + 1
                
                playlists[pid].append(track['track_uri'])
                documents[pid][track['track_uri']] = documents[pid].get(track['track_uri'],0) + 1
                if documents[pid][track['track_uri']] ==1:
                    dictionary[track['track_uri']] = dictionary.get(track['track_uri'],0) + 1
            avglen = avglen + lendocs[pid]
        k = k +1
        if k==2:
            break
    avglen = avglen / len(documents.keys())      
    df.columns = columns
def bm25(query, m=100, k1 = .9, b = .75): #bm25 for intial set
    global avglen
    query = query.split("_")
    query = query[0].split() + query[1:]
    pids = documents.keys()
    N = len(pids)
    doc_relevance = dict()
    for pid in pids: #finding RSV for each document
        tfs = documents[pid]
        Ld = lendocs[pid]
        for q in query:
            df = dictionary.get(q,1)
            tfd = tfs.get(q,0)
            doc_relevance[pid] = doc_relevance.get(pid,0) + math.log(N/(df))*(k1 +1)*tfd/(k1*((1-b) + b*(Ld/avglen)) + tfd)
    topresults = sorted(doc_relevance.keys(), key=lambda x: (doc_relevance[x]),reverse= True)
    return topresults[:m]
def preprocessquery(query):
    tokens = query.split("_")
    dict = {}
    for token in tokens:
        dict[token] = dict.get(token,0) + 1
    return dict
def findbyqe(query, m=10, k=40): #query expansion
    topresults = bm25(query,k) #initial results
    p = {}
    N = len(documents.keys())
    term_scores = {}
    for pid in topresults:
        terms = documents[pid].keys()
        for term in terms:
            p[term] = p.get(term,0) + 1
    All_terms = p.keys()
    for term in All_terms: #finding term scores
        df = dictionary[term]
        ui = (df+.5)/N
        pi = (p[term]+.5)/(k+1)
        wt = math.log(pi/(1-pi)) - math.log(ui/(1-ui))
        term_scores[term] = wt*p[term]
    top_tracks = sorted(term_scores.keys(), key=lambda x: (term_scores[x]),reverse= True)
    answers = []
    query_tracks = set(query.split("_"))
    for track in top_tracks: #finding top 500 tracks
        if len(answers) >= 500:
            break
        if track not in titles and track not in query_tracks:
            answers.append(track)
        
    return answers     
def escapefunc(st): #encodes title in ""
    return "\"%s\"" % st.replace("\"", "\"\"")      

if __name__ == "__main__":
    readmpd()
    testdf = readchallengeset() 
    testdf.to_csv('output.csv', index = True) #output file for recommended songs 
