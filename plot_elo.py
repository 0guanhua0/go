import csv

import matplotlib.pyplot as plt
from whr import Base


def plot_elo():
    game = []
    with open("whr_history.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            game.append(
                {"p1": row[0], "p2": row[1], "winner": row[2], "time": int(row[3])}
            )

    game.sort(key=lambda x: x["time"])
    start_time = game[0]["time"]
    model = []
    model_hour = {}
    for g in game:
        for x in (g["p1"], g["p2"]):
            if x not in model_hour:
                model.append(x)
                model_hour[x] = int((g["time"] - start_time) / 3600.0)

    whr = Base()
    for g in game:
        whr.create_game(g["p1"], g["p2"], g["winner"], model_hour[g["p2"]])
    whr.iterate_until_converge(verbose=False)

    elo = []
    hour = []
    for m in model:
        rating = whr.ratings_for_player(m)
        h = model_hour[m]
        elo.append(min(rating, key=lambda x: abs(x[0] - h))[1])
        hour.append(h)

    plt.style.use("dark_background")
    plt.xlabel("hour", color="white")
    plt.ylabel("elo", color="white")
    plt.scatter(hour, elo, color="white")
    plt.box(False)
    plt.savefig("elo.png", facecolor="black")


if __name__ == "__main__":
    plot_elo()
