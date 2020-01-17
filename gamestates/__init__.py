from gamestates.battleselect import BattleSelect
from gamestates.battleconfirm import BattleConfirm
from gamestates.mode_trainingtower import TrainingTower
from gamestates.trainingtowerselect import TrainingTowerSelect

from enum import Enum, auto
class Gamestate(Enum):
    BATTLE_SELECT = auto()