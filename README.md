# ALN
This the Source code for our adversarial based neural network for affect estimations in the wild. 

The requirements : 
  1. Python 2.7
  2. Pytorch 0.4
  3. OpenSmile : https://www.audeering.com/opensmile/

Preparations : 
  1. Extract the LLD sound features using opensmiles. 

Trainings : 
  1. chal-main.py : To train the discriminator only using l2 loss
  2. chal-main_gan_single.py : To train using both Generator and Discriminators 
  Please observe the required parameters to tweak the configurations.
  
Tests : 
  The validation code can be seen on the chal-man_gan_single.py, do_test().
