import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
from script import phrases

# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(use_deepspeed=True, kv_cache=True)




# This is the text that will be spoken.

# Here's something for the poetically inclined.. (set text=)
"""
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,"""

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "high_quality"




print(torch.cuda.is_available())


# Pick one of the voices from the output above
voice = 'oussama'

# Load it and send it through Tortoise.
voice_samples, conditioning_latents = load_voice(voice)
# for content in ['teaser', 'intro']:
for index, text in enumerate(phrases):
    print(f" File index is the following: {index}")
    gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                            preset=preset)
    torchaudio.save(f'/app/output/generated_{index}.wav', gen.squeeze(0).cpu(), 24000)

# Tortoise can also generate speech using a random voice. The voice changes each time you execute this!
# (Note: random voices can be prone to strange utterances)
# gen = tts.tts_with_preset(text, voice_samples=None, conditioning_latents=None, preset=preset)
# torchaudio.save('/app/output/generated3.wav', gen.squeeze(0).cpu(), 24000)


# voice_samples, conditioning_latents = load_voices(['pat', 'william'])

# gen = tts.tts_with_preset("They used to say that if man was meant to fly, he’d have wings. But he did fly. He discovered he had to.", 
#                           voice_samples=None, conditioning_latents=None, preset=preset)
# torchaudio.save('/app/output/captain_kirkard2.wav', gen.squeeze(0).cpu(), 24000)
