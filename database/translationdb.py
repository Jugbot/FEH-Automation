import os
import json
import shelve
from jsonobject import JsonObject
from constants import *


class TranslationDB():
    def __init__(self):
        if CLEAN:
            self.db = shelve.open(TRANSLATION_CACHE_NAME, flag='n')
        self.db = shelve.open(TRANSLATION_CACHE_NAME)
        
        if len(self.db) == 0:
            for root, dirs, files in os.walk(TRANSLATION_DATA):
                for json_file in files:
                    with open(os.path.join(root, json_file), 'r', encoding="utf-8") as f:
                        translations = json.load(f)
                        for o in translations:
                            self.db[o["key"]] = o["value"]

    def data(self, name):
        return self.db[name]


if __name__ == "__main__":
    t = TranslationDB()