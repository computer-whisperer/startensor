from mpyq import MPQArchive
import json
import os
import queue
import threading


def sort_file(filename):
   # file = open(filename, 'r')
    archive = MPQArchive(filename)
    jsondat = archive.read_file("replay.gamemetadata.json")
   # file.close()
    dat = json.loads(jsondat)
    winner = None
    looser = None

    for player in dat["Players"]:
        if player is None:
            continue
        if player["Result"] == "Win":
            winner = player
        if player["Result"] == "Loss":
            looser = player
        if "MMR" not in player:
            player["MMR"] = 0

    if dat["Duration"] < 20:
        os.renames(filename, "short/{}".format(filename))
    elif winner is None or looser is None:
        os.renames(filename, "missingplayer/{}".format(filename))
    else:
        new_fname = "{}/{}/{}/{}/{}".format(winner["AssignedRace"], looser["AssignedRace"], winner["MMR"] - winner["MMR"]%100, looser["MMR"] - looser["MMR"]%100, filename)
        os.renames(filename, new_fname)

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        try:
            sort_file(item)
        except Exception as e:
            print(e)
        q.task_done()

q = queue.Queue()
threads = []
for i in range(4):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

for filename in os.listdir("."):
    if filename.endswith(".SC2Replay"):
        q.put(filename)

q.join()

for t in threads:
    q.put(None)

for t in threads:
    t.join()
