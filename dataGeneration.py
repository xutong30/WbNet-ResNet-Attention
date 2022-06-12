import os
import shutil
import pandas as pd
import csv


# generating .csv files for the Wingbeats dataset
def wingbeats_dataset(root):
    species_number = {}
    with open('dataInfo_Wingbeats.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for paths in os.listdir(root):
            count = 0
            for root_dir, cur_dir, files in os.walk(root + paths):

                #       open a csv file
                index = paths.split('. ')
                filename = index[0] + '.csv'

                for name in files:
                    fname = root_dir + "/" + name
                    writer.writerow([fname, index[0], index[1]])
                count += len(files)
            species_number.update({paths: count})

    print(species_number)

    species_num = list(species_number.values())

    file = 'dataInfo_Wingbeats.csv'
    df = pd.read_csv(file, header=None)

    previous = 0
    total = 0

    for i, num in enumerate(species_num):
        total += num
        if i == 0:
            train = df[previous:total].sample(frac=0.80)
        else:
            temp = df[previous:total].sample(frac=0.80)
            train = train.append(temp)

        previous += num

    train.to_csv('trainData_Wingbeats.csv', header=["Fname", "Genera", "Species"], index=False)
    vali = df.drop(train.index, axis=0)
    vali.to_csv('valiData_Wingbeats.csv', header=["Fname", "Genera", "Species"], index=False)


# generating .csv files for the Abuzz dataset
def abuzz_dataset(root):
    species_number = {}
    with open('dataInfo_Abuzz.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for paths in os.listdir(root):
            count = 0
            for root_dir, cur_dir, files in os.walk(root + paths):

                #       open a csv file
                index = paths.split('. ')
                filename = index[0] + '.csv'

                for name in files:
                    fname = root_dir + "/" + name
                    writer.writerow([fname, index[0], index[1]])
                count += len(files)
            species_number.update({paths: count})

    print(species_number)

    species_num = list(species_number.values())

    file = 'dataInfo_Abuzz.csv'
    df = pd.read_csv(file, header=None)

    previous = 0
    total = 0

    for i, num in enumerate(species_num):
        total += num
        if i == 0:
            train = df[previous:total].sample(frac=0.80)
        else:
            temp = df[previous:total].sample(frac=0.80)
            train = train.append(temp)

        previous += num

    train.to_csv('trainData_Abuzz.csv', header=["Fname", "Genera", "Species"], index=False)
    vali = df.drop(train.index, axis=0)
    vali.to_csv('valiData_Abuzz.csv', header=["Fname", "Genera", "Species"], index=False)


if __name__ == "__main__":
    wingbeats_dataset('./Wingbeats/')
    abuzz_dataset('./Abuzz/')


