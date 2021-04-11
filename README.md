# Automatic-playlist-continuation
Project description
The task is to create a playlist continuation model which, given a seed playlist title and/or initial set of tracks in a playlist, can predict the subsequent tracks in that playlist, thereby continuing the playlist automatically. This is a form of the more general task of sequential recommendation. The model is to be trained using Spotify Million Playlist Dataset.
Input: A user-created playlist which is represented by playlist metadata(such as last edit time, number of playlist edits, and more) and k seed tracks where k=0,1,5,10,25 or 100. 
Output: A list of 500 recommended candidate tracks, ordered by relevance in decreasing order.
Approach taken
Research has shown that the recommendation system problem can be modeled as an ad hoc retrieval task. [1] 
Case 1: Input playlist contains some songs.
Thus we aim to use pseudo-relevance feedback, a standard ad hoc retrieval method as a collaborative filtering mechanism to filter out the relevant songs for playlist continuation. 
We generate a set of pseudo documents by treating each playlist as a document. The tracks in the playlist are considered as terms. 
The input playlist will be considered as a query. Using pseudo relevance feedback, we will add more terms to the query ie more songs to the input playlist which will act in a way similar to collaborative filtering. 
Case 2: there is no song in the input playlist. 
we will use the title of the playlist as a cue to generate more songs. The tracks from the playlist having titles similar to the input playlist title will be given more preference. The track which appears in more number of such similar playlists is more relevant. This can also be modeled as an ad hoc retrieval task. 
A similarity score will be generated for each playlist title and the query. This score will be added to the total score of each track in the playlist. In the end, the tracks will be sorted in decreasing order by their total scores and the more relevant tracks will be given as output. 
Dataset:
Spotify released a dataset of one million user-created playlists from the Spotify platform, dubbed the Million Playlist Dataset (MPD). The dataset contains 1,000,000 playlists, including playlist titles and track titles, created by users on the Spotify platform between January 2010 and October 2017. It is 5.5 gb in size and has been downloaded by the team already. 
A challenge set consisting of 10000 playlists is available for validation purposes. 
Source: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

