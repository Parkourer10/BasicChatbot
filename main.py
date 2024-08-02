from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os
import threading
import queue
import time
import pyttsx3
import torch

# Configuration
Model = 'tiny'
English = True
Translate = False
SampleRate = 44100
BlockSize = 30
Threshold = 0.1
Vocals = [50, 1000]
EndBlocks = 40

# Chat model setup
template = """
Answer the questions below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model="AI")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', 'english_rp+f3')

class StreamHandler:
    def __init__(self):
        self.padding = 0
        self.prevblock = self.buffer = np.zeros((0, 1))
        self.fileready = False
        self.last_text = None
        self.transcription_queue = queue.Queue()
        self.transcribing = threading.Event()
        print("Loading Whisper Model..", end='', flush=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(f'{Model}{".en" if English else ""}', device=self.device)
        print(" Done.")
        self.stop_event = threading.Event()

    def callback(self, indata, frames, time, status):
        if not any(indata):
            return
        freq = np.argmax(np.abs(np.fft.rfft(indata[:, 0]))) * SampleRate / frames
        if np.sqrt(np.mean(indata**2)) > Threshold and Vocals[0] <= freq <= Vocals[1]:
            if self.padding < 1:
                self.buffer = self.prevblock.copy()
            self.buffer = np.concatenate((self.buffer, indata))
            self.padding = EndBlocks
        else:
            self.padding -= 1
            if self.padding > 1:
                self.buffer = np.concatenate((self.buffer, indata))
            elif self.padding < 1 < self.buffer.shape[0] > SampleRate:
                self.fileready = True
                write('dictate.wav', SampleRate, self.buffer)
                self.buffer = np.zeros((0, 1))
            elif self.padding < 1 < self.buffer.shape[0] < SampleRate:
                self.buffer = np.zeros((0, 1))
            else:
                self.prevblock = indata.copy()

    def transcribe_audio(self):
        while not self.stop_event.is_set():
            self.transcribing.wait()
            if self.fileready:
                print("\nTranscribing..")
                try:
                    result = self.model.transcribe(
                        'dictate.wav',
                        fp16=False,
                        language='en' if English else '',
                        task='translate' if Translate else 'transcribe'
                    )
                    transcription_text = result['text']
                    self.transcription_queue.put(transcription_text)
                except Exception as e:
                    print(f"Error during transcription: {e}")
                finally:
                    self.fileready = False
                    if os.path.exists('dictate.wav'):
                        os.remove('dictate.wav')
            self.transcribing.clear()
            time.sleep(0.1)

    def listen(self):
        print("Listening.. (Ctrl+C to Quit)")
        with sd.InputStream(channels=1, callback=self.callback, blocksize=int(SampleRate * BlockSize / 1000), samplerate=SampleRate):
            while not self.stop_event.is_set():
                self.transcribing.set()
                if not self.transcription_queue.empty():
                    return self.transcription_queue.get()

def chat_response(context, user_input):
    result = chain.invoke({"context": context, "question": user_input})
    print("AI: ", result)
    engine.say(result)
    engine.runAndWait()
    return result

def main():
    handler = StreamHandler()
    context = ""
    print('Welcome!')

    transcription_thread = threading.Thread(target=handler.transcribe_audio, daemon=True)
    transcription_thread.start()

    try:
        while True:
            user_input = handler.listen()
            print('You: ', user_input)
            if user_input:
                if user_input.lower() == "exit":
                    break
                # Run chat model processing in a separate thread for parallel execution
                def process_chat():
                    nonlocal context
                    result = chat_response(context, user_input)
                    context += f"\nUser: {user_input}\nAI: {result}"

                chat_thread = threading.Thread(target=process_chat)
                chat_thread.start()
                chat_thread.join()  # Ensure chat thread completes before continuing

    except (KeyboardInterrupt, SystemExit):
        print("Stopping...")
    finally:
        handler.stop_event.set()
        transcription_thread.join()
        print("Quitting..")
        if os.path.exists('dictate.wav'):
            os.remove('dictate.wav')

if __name__ == '__main__':
    main()
