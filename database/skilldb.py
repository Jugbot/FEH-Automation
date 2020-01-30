import os
import shelve
from jsonobject import JsonObject
import json
from constants import *


class SkillDB():
    def __init__(self, translationdb):
        if CLEAN:
            self.db = shelve.open(SKILL_CACHE_NAME, flag='n')
        self.db = shelve.open(SKILL_CACHE_NAME)
        if len(self.db) == 0:
            for path in os.listdir(SKILL_DATA):
                with open(path, 'r', encoding="utf-8") as f:
                    skills = json.load(f)
                    for skill in skills:
                        if skill.id_tag in self.db:
                            print(*self.db[skill.id_tag].items(), sep='\n')
                            print(*skill.items(), sep='\n')
                        else:
                            self.db[skill.id_tag] = {}
                        self.db[skill.id_tag].update(skill)
            # also provide readable name as key
            for key in self.db:
                name_id = self.db[key]["name_id"]
                self.db[translationdb.data(name_id)] = self.db[key]

    def data(self, name):
        return JsonObject(self.db[name])


if __name__ == "__main__":
    from database.translationdb import TranslationDB
    sdb = SkillDB(TranslationDB())