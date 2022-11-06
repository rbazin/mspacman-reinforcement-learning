import csv


def read_csv(filename):
    """function for reading the feature data from a csv file"""
    feature_arr = []
    file = open(filename, "r")
    csv_data = csv.reader(file, delimiter=",")
    for row in csv_data:
        features = []
        for i in range(0, len(row)):
            features.append(float(row[i]))
        feature_arr.append(features)
    return feature_arr


def write_csv(filename, data):
    """function for writing data into a csv file"""
    file = open(filename, "w")
    csv_writer = csv.writer(file, delimiter=",")
    csv_writer.writerow(data)
    file.close()
