from mpyq import MPQArchive
import json
import os

def sort_file(filename):
    archive = MPQArchive(filename)
    jsondat = archive.read_file("replay.gamemetadata.json")
    dat = json.loads(jsondat)
    winner = None
    looser = None
    for player in dat["Players"]:
        if player["Result"] == "Win":
            winner = player
        else:
            looser = player
    if not (winner is not None and looser is not None and "MMR" in winner and "MMR" in looser):
        os.renames(filename, "naughty/{}".format(filename))
    new_fname = "{}/{}/{}/{}/{}".format(winner["AssignedRace"], looser["AssignedRace"], winner["MMR"] - winner["MMR"]%100, looser["MMR"] - looser["MMR"]%100, filename)
    os.renames(filename, new_fname)

for filename in os.scandir("."):
    if filename.path.endswith(".SC2Replay"):
        try:
            sort_file(filename.path)
        except Exception as e:
            print(e)
