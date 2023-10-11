import openai  # Assume you have access to the OpenAI library
import torch  # Importing PyTorch for any additional processing you might need
import torchaudio
from transformers import pipeline
from datasets import load_dataset
# key
openai.api_key_path = "/home/yuehpo/coding/DeepMIR_2023fall/hw1/OPENAI_KEY.txt"




def extract_lyrics(audio_file):
    # Use Whisper or another ASR system to extract lyrics from the audio clip
    # Assume whisper_asr is a function you've defined to handle this
    

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    chunk_length_s=30,
    device=device,
    )

    
    sample, sr = torchaudio.load(audio_file)    
    
    
    sample = sample.squeeze().numpy()
    

    prediction = pipe(sample, batch_size=8)["text"]

    # we can also return timestamps for the predictions
    prediction = pipe(sample, batch_size=8, return_timestamps=True)["chunks"]
    
    return prediction

def identify_artist(lyrics):
    # Use ChatGPT to attempt to identify the artist based on the lyrics
    artists = [aerosmith, beatles
                ,creedence_clearwater_revival
                ,cure
                ,dave_matthews_band
                ,depeche_mode
                ,fleetwood_mac
                ,garth_brooks
                ,green_day
                ,led_zeppelin
                ,madonna
                ,metallica
                ,prince
                ,queen
                ,radiohead
                ,roxette
                ,steely_dan
                ,suzanne_vega
                ,tori_amos
                ,u2]
    prompt = f"Now, lets guess which artist is the artist of the song with the following lyrics with timestamp.\n\n{lyrics}\n\nYou can choose from the following artists:\n{artists}"
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

def pipeline_whisper_chatgpt(audio_clip):
    lyrics = extract_lyrics(audio_clip)
    # save lyrics
    f = open("lyrics.txt", "w")
    f.write(lyrics)
    # artist = identify_artist(lyrics)
    return None


if __name__ == '__main__':

    # Assume audio_clip is a variable holding your audio data
    audio_file = "/home/yuehpo/coding/DeepMIR_2023fall/hw1/artist20/train/radiohead/Pablo_Honey/09-Prove_Yourself.mp3"

    lyric = "\n".join(extract_lyrics(audio_file))
    print(lyric)

