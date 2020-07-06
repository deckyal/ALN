# ALN (Adversarial based Latent Network)

This the Source code for our adversarial based neural network for affect estimations in the wild. 

This source code holds the image augmentations techniques used on our other repositories : https://github.com/deckyal/FADeNN. 

Requirements : 
  1. Python 2.7
  2. Pytorch 0.4
  3. OpenSmile : https://www.audeering.com/opensmile/

Preparations : 
  1. Extract the LLD sound features using opensmiles. 

Trainings : 
  1. chal-main.py : To train the discriminator only using l2 loss
  2. chal-main_gan_single.py : To train using both Generator and Discriminators. 
  Please observe the required parameters to tweak the configurations.
  
Tests : 
  The validation code can be seen on the chal-man_gan_single.py, do_test().

Citation : 
[D. Aspandi, A. Mallol-Ragolta, B. Schuller and X. Binefa,  "Latent-Based Adversarial Neural Networks for Facial Affect Estimations," in 2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020) (FG), Buenos Aires, undefined, AR, 2020 pp. 348-352.]
