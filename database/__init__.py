from database.elementdb import ElementDB, Selection
from database.mapdb import MapDB
from database.translationdb import TranslationDB
from database.skilldb import SkillDB

mapDB = MapDB()
elemDB = ElementDB()
translationDB = TranslationDB()
skillDB = SkillDB(translationDB)
