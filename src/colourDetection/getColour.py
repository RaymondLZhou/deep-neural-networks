def getColourName(R, G, B, csv):
    minimum = 10000

    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i, "R"])) + abs(G- int(csv.loc[i, "G"]))+ abs(B- int(csv.loc[i, "B"]))

        if(d <= minimum):
            minimum = d
            cname = csv.loc[i, "colour_name"]

    return cname
