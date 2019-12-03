import torch as t
import Models
import Losses
import Dataloader
from dlutils.tracker import LossTracker
from dlutils.progress_bar import ProgressBar
import Logging
import time
import evaluation
'''
    TO DO:
        -Add Testing at end    
'''


def train(data_folder="UTKFace", split=(0.6, 0.3, 0.1), batch_norm=False, probabilistic=False, num_workers=0, epochs=100,
          output_folder="models/", lr=1e-04, batch_size=4,lr_red_int=50):
    train_dl, test_dl, val_dl = Dataloader.generate_dataloaders(data_folder, split, num_workers=num_workers,
                                                                batch_size=batch_size)

    Logging.log(data_folder, split, batch_norm, probabilistic, epochs, output_folder, lr, batch_size,lr_red_int)

    tracker = LossTracker(output_folder=output_folder)

    model = Models.Regress(probabilistic=probabilistic, batch_norm=batch_norm)
    model.to('cuda')
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    checkpoint = {'model': model,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    n = len(train_dl)
    print("\nTOTAL EPOCHS: " + str(epochs))
    print("TOTAL BATCHES: " + str(n) + "\n\n")
    for epoch in range(0, 100):
        print("################################################################")
        print("##  \t\t\t\t\t\t\t\t\t\t\t\t\t\t    ##")
        print("##  \t-----------------" + ("EPOCH " + str(epoch)).center(16) + "----------------\t    ##")
        print("##  \t\t\t\t\t\t\t\t\t\t\t\t\t\t    ##")
        print("################################################################\n\n")
        if epoch % lr_red_int == 0 and epoch != 0:
            print("adjusting lr...")
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

        # TRAIN ONE EPOCH
        p = ProgressBar(n, fill="~")
        i = 1
        j = 1
        start = time.time()
        for sample in train_dl:
            if i % (n / 10) < 1 or i == 1:
                k = i - j
                p.increment(k)
                j = i
            inp = sample[0].to(t.device("cuda"))
            x = model(inp)
            y = sample[1].to(t.device("cuda"))
            if probabilistic:
                loss = Losses.prob_loss(x, y)
            else:
                loss = Losses.reg_loss(x, y)
            d = {"Training Loss": loss}
            tracker.update(d)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        tot = time.time() - start
        p.increment(p.total_iterations - p.current_iteration)
        ips = (n * batch_size / tot)
        print("\nThroughput " + str(ips)[0:8] + " images/s\n")
        # Validate and record
        print("Validating...\n")
        with t.no_grad():
            for sample in val_dl:
                inp = sample[0].to(t.device("cuda"))
                x = model.eval()(inp)
                y = sample[1].to(t.device("cuda"))
                if probabilistic:
                    loss = Losses.prob_loss(x, y)
                else:
                    loss = Losses.reg_loss(x, y)
                d = {"Validation Loss": loss}
                tracker.update(d)

        print(x)
        print("Done.\n")
        tracker.register_means(epoch + 1)
        tracker.plot()
        vloss = tracker.state_dict()["means_over_epochs"]["Validation Loss"][-1]
        checkpoint["Validation Loss"] = vloss
        print("Validation Loss: "+str(vloss)+"\n\n")
        if epoch%25==0:
            t.save(checkpoint,output_folder+"Epoch_"+str(epoch))

    evaluation.testing(model,test_dl,output_folder)

if __name__ == "__main__":
    data_folder="/data/datasets/UTK/UTKFace/"#input("What is the data folder path? ")
    num_workers=8#input("Num workers? ")
    output_folder="/data/Liam/UTKTesting/Model1/"
    train(data_folder=data_folder,output_folder=output_folder)