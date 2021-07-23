from detect import img2char
from synthesizer import Synthesizer
import cv2
from utils.audio import save_wav

if __name__=="__main__":
  img = cv2.imread('')#이미지 루트
  texts = img2char(img) #텍스트
  load_path=""#프리트레인 모델
  num_speakers =1
  checkpoint_step =None
  sample_path = "logdir-tacotron2/generate"
  speaker_id = 0
  base_alignment_path = None
  is_korean = True
  for text in texts:
  synthesizer = Synthesizer()
  synthesizer.load(load_path,num_speakers,checkpoint_step,inference_prenet_dropout=False)
  for text in texts:
      audio = np.append(audio,synthesizer.synthesize(texts=[text],base_path=config.sample_path,speaker_ids=[config.speaker_id],
                                   attention_trim=True,base_alignment_path=config.base_alignment_path,isKorean=config.is_korean))
      audio = np.append(audio,np.zeros(5000))
    
  save_wav(audio,'/content/hi.wav',24000)
