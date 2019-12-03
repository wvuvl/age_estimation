import torch as t
from dlutils.tracker import LossTracker
import Losses
import numpy as np
import os




def testing(model,testing_dl,output_folder):

    os.mkdir(output_folder+"test/")
    with t.no_grad():
        test=[]
        tracker=LossTracker(output_folder+"test/")
        for sample in testing_dl:
            inp=sample[0].to(t.device("cuda"))
            x=model.eval()(inp)
            y = sample[1].to(t.device("cuda"))
            if model.probabilistic:
                loss = Losses.prob_loss(x, y)
            else:
                loss = Losses.reg_loss(x, y)
            d = {"Test Loss": loss}
            tracker.update(d)
            entry=[x,y,loss.cpu().numpy().item()]
            test+=entry
    tracker.register_means(1)
    test=np.asarray(test)
    np.save(output_folder+"test/"+"test.npy",test)

    log="Testing Loss: "+str()

    with open(output_folder +"test/"+ "Description.txt", "w") as file:
        file.write(log)










