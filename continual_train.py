import train
import numpy as np
import os

def continual_train(data_folder="UTKFace", split=(0.6, 0.3, 0.1), batch_norm=False, probabilistic=False, num_workers=0, epochs=100,
                    output_folder="models/", lrs=[1e-04],lr_reduction_points=[50], batch_size=4):


    j=0
    while True:
        np.random.shuffle(lr_reduction_points)
        np.random.shuffle(lrs)
        lr_red=lr_reduction_points[0]
        lr=lrs[0]
        dir=output_folder+"Instance"+str(j)+"_LR_"+str(lr)+"/"
        os.mkdir(output_folder+"Instance"+str(j)+"_LR_"+str(lr))
        train.train(data_folder,split,
                    batch_norm=batch_norm,
                    probabilistic=probabilistic,
                    num_workers=num_workers,
                    epochs=epochs,
                    output_folder=dir,
                    lr=lr,batch_size=batch_size,lr_red_int=lr_red)


        j+=1


if __name__ == "__main__":
    n=input("n")
    data_folder="/data/datasets/UTK/UTKFace/"
    output_folder = "/data/Liam/UTKTesting/Prob"+str(int(n))+"/"
    continual_train(data_folder=data_folder,output_folder=output_folder,probabilistic=True,lrs=[1e-04,1e-03,1e-05],lr_reduction_points=[20,30,40])