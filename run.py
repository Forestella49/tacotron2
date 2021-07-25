from detect import img2char
from syn_mod import Synthesizer
import cv2
from utils.audio import save_wav
import numpy as np

if __name__=="__main__":
  img = cv2.imread(r'C:\Users\김규림\pythonProject\tacotron2\test2.PNG')#이미지 루트
  texts = img2char(img) #텍스트
  load_path=r"C:\Users\김규림\pythonProject\tacotron2\checkpoint"#프리트레인 모델
  num_speakers =1
  checkpoint_step =None
  sample_path = r"C:\Users\김규림\pythonProject\tacotron2\sample"
  speaker_id = 0
  base_alignment_path = None
  is_korean = True

  synthesizer.load(load_path,num_speakers,checkpoint_step,inference_prenet_dropout=False)
  audio = np.zeros(1)
  for text in texts:
      audio = np.append(audio,synthesizer.Synthesize(texts=[text],base_path=sample_path,speaker_ids=[speaker_id],
                                   attention_trim=True,base_alignment_path=base_alignment_path,isKorean=is_korean))
      audio = np.append(audio,np.zeros(5000))
    
  save_wav(audio,'/content/hi.wav',16000)
