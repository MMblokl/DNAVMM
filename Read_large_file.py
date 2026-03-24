def get_sequences(file="./csv/BIOSCAN_5M_Insect_Dataset_metadata.csv"):
    with open(file, 'r') as inputfile:
        line = inputfile.readline()
        while len(line) != 0:
            print(line[11], "\n")
            line = inputfile.readline().split(',')
        inputfile.close()



if __name__ == "__main__":
    get_sequences()