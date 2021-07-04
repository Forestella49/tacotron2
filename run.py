from detect import img2char
from synthesizer import Synthesizer
import cv2

if __name__=="__main__":
  img = cv2.imread('')#이미지 루트
  text = img2char(img) #텍스트
  load_path=""#프리트레인 모델
  num_speakers =2
  checkpoint_step =None
  sample_path = "logdir-tacotron2/generate"
  speaker_id = 0
  base_alignment_path = None
  is_korean = True
  synthesizer = Synthesizer()
  synthesizer.load(load_path,num_speakers,checkpoint_step,inference_prenet_dropout=False)

  audio = synthesizer.synthesize(texts=[text],base_path=sample_path,speaker_ids=[speaker_id],
                                   attention_trim=True,base_alignment_path=base_alignment_path,isKorean=is_korean)[0]


#python synthesizer.py 
#--load_path logdir-tacotron2/moon+son_2019-02-27_00-21-42 
#--num_speakers 2 
#--speaker_id 0 
#--text "오스트랄로피테쿠스 아파렌시스는 멸종된 사람족 종으로, 현재에는 뼈 화석이 발견되어 있다."
