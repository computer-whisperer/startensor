#include <sc2api/sc2_api.h>
#include <sc2lib/sc2_lib.h>

#include <iostream>

using namespace sc2;

class Bot : public Agent {
public:
    virtual void OnGameStart() final {
        std::cout << "Hello, World!" << std::endl;
    }

    virtual void OnStep() final {
        uint32_t game_loop = Observation()->GetGameLoop();

        if (game_loop % 100 == 0) {
            sc2::Units units = Observation()->GetUnits(sc2::Unit::Alliance::Self);
            for (auto& it_unit : units) {
                sc2::Point2D target = sc2::FindRandomLocation(Observation()->GetGameInfo());
                Actions()->UnitCommand(it_unit, sc2::ABILITY_ID::SMART, target);
            }
}
    }
};

int main(int argc, char* argv[]) {
    Coordinator coordinator;
    coordinator.LoadSettings(argc, argv);

    Bot bot;
    Bot bot2;
    coordinator.SetParticipants({
        CreateParticipant(Race::Terran, &bot),
        CreateParticipant(Race::Zerg, &bot2)
    });

    coordinator.LaunchStarcraft();
    coordinator.StartGame("/home/christian/StarCraftII/Maps/Ladder2017Season3/AcolyteLE.SC2Map");

    while (coordinator.Update()) {
    }
    std::cout << "Saving replay";
    bot.Control()->SaveReplay("/home/christian/StarCraftII/Replays/AcolyteLE2.SC2Replay");

    return 0;
}
