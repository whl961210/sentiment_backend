import pandas as pd
import googleapiclient.discovery
import os
import re

def get_youtube_comments(video_id):
    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = os.getenv('YOUTUBE_API_KEY')  # Use an environment variable for the API key

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=DEVELOPER_KEY)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        text = comment['textDisplay']

        # Regex to match non-English characters
        if re.match("^[a-zA-Z0-9\s,.'-?!]*$", text):
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                text
            ])

    return pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
