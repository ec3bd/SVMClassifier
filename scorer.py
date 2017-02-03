import csv

with open('Book2.csv', 'r') as w:
    read = csv.reader(w)
    read = list(read)
    with open('predictions.txt', 'r') as f:
        pred = csv.reader(f)
        pred = list(pred)


correct = 0.0
total = 0.0
for x in range(0, len(read)):
    if(read[x][15] == pred[x][0]):
        correct+=1.0
    total+=1.0

print(correct/total)