def log(folder, split, batch_norm, probabilistic, epochs, output_folder, lr, batch_size):
    params = [str(lr),
              str(probabilistic),
              str(split),
              str(batch_norm),
              str(epochs),
              str(batch_size),
              str(folder),
              str(output_folder)]
    log = ""
    for entry in params:
        log += "," + entry

    with open(output_folder + "Description.txt", "w") as file:
        file.write(log)
