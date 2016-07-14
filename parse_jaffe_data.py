import sys

def parse():
    file = open("data_ratings.txt", "r")

    ratings = []
    for line in file:
        line_vector = line.split()
        line_vector = line_vector[1:]
        ratings.append(line_vector)

    ratings.sort(key = lambda x: x[6])

    #print(ratings)
    for rating in ratings:
        del rating[-1:]

    ratings = [ [float(val) for val in rating] for rating in ratings ]

    return ratings
    
if __name__ == '__main__':
    file = open("data_ratings.txt", "r")

    ratings = []
    for line in file:
        line_vector = line.split()
        line_vector = line_vector[1:]
        ratings.append(line_vector)

    ratings.sort(key = lambda x: x[6])
    
    for rating in ratings:
        print(rating[-1:])
    
    name = ''
    count = 0
    for rating in ratings:
        name2 = rating[-1:][0][0:5]
        if name != name2:
            print(name, count)
            name = name2
            count = 1
        else:
            count = count + 1
    
    print(name, count)