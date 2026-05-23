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
    model = []
    seen = set()
    for g in game:
        if g["p1"] not in seen:
            model.append(g["p1"])
            seen.add(g["p1"])
        if g["p2"] not in seen:
            model.append(g["p2"])
            seen.add(g["p2"])

    model_to_day = {m_id: i for i, m_id in enumerate(model)}
    whr = Base()

    for g in game:
        whr.create_game(g["p1"], g["p2"], g["winner"], model_to_day[g["p2"]])

    whr.iterate_until_converge(verbose=False)

    elos = []
    for m_id in model:
        rating = whr.ratings_for_player(m_id)
        day = model_to_day[m_id]
        elo = min(rating, key=lambda x: abs(x[0] - day))[1]
        elos.append(elo)

    plt.style.use("dark_background")
    plt.xlabel("model", color="white")
    plt.ylabel("elo", color="white")
    plt.xticks([])
    plt.scatter(range(len(elos)), elos, color="white")
    plt.box(False)
    plt.savefig("elo.png", facecolor="black")


if __name__ == "__main__":
    plot_elo()
