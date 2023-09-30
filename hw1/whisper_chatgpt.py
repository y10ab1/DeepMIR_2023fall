import openai  # Assume you have access to the OpenAI library
import torch  # Importing PyTorch for any additional processing you might need

# key
openai.api_key_path = "/home/yuehpo/coding/DeepMIR_2023fall/hw1/OPENAI_KEY.txt"




def extract_lyrics(audio_file):
    # Use Whisper or another ASR system to extract lyrics from the audio clip
    # Assume whisper_asr is a function you've defined to handle this
    lyrics = openai.Audio.transcribe("whisper-1", audio_file).text
    print(lyrics)
    return lyrics

def identify_artist(lyrics):
    # Use ChatGPT to attempt to identify the artist based on the lyrics
    prompt = f"Who is the artist of the song with the following lyrics?\n\n{lyrics}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant to classify artist of the song from the provided lyric."},
            {"role": "user", "content": f"{prompt}"},
    ]
)
    # Assume the artist name is in the text of the response
    # You may need to process the response to extract the artist name
    print(response)
    artist = response['choices'][0]['message']['content']
    return artist

def pipeline(audio_clip):
    lyrics = extract_lyrics(audio_clip)
    # save lyrics
    f = open("lyrics.txt", "w")
    f.write(lyrics)
    # artist = identify_artist(lyrics)
    return None


if __name__ == '__main__':

    # Assume audio_clip is a variable holding your audio data
    audio_file= open("/home/yuehpo/coding/DeepMIR_2023fall/hw1/artist20/train/radiohead/Pablo_Honey/09-Prove_Yourself.mp3", "rb")

    artist_prediction = pipeline(audio_file)
    print(artist_prediction)

