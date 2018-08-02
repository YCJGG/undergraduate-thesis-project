import os

lr = 1e-2*3.33
k = 0
while(k<9):
    lr = lr/3.33
    os.system('CUDA_VISIBLE_DEVICES=4,5 python fine_tune.py ' + '--lrC='+str(lr))
    k += 1
       
