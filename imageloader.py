import os
import xml.etree.ElementTree as ET
import json
import cv2 
import numpy

def walkPList(node):
    if node.tag == "dict":
        collection = {}
        for i, child in enumerate(node):
            if child.tag == "key":
                collection[child.text] = walkPList(node[i+1])
        return collection
    elif node.tag == "array":
        array = []
        for child in node:
            array.append(walkPList(child))
        return array
    elif node.tag == "string":
        if node.text[0] == "{":
            return json.loads(node.text.replace("{", "[").replace("}", "]"))
        return node.text
    elif node.tag == "integer":
        return int(node.text)
    elif node.tag == "false":
        return False
    elif node.tag == "true":
        return True
    elif node.tag == "plist":
        return walkPList(node[0])
    else:
        print("unhandled tag: ", node.tag)
        return node.text

def parsePList(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    return walkPList(root)

assets = []
for root, dirs, files in os.walk("./templates/UI"):
    for fname in files:
        name, ext = fname.split('.')
        if ext == "plist":
            assets.append({
                'source': cv2.imread(root + name + ".png", cv2.IMREAD_UNCHANGED),
                'plist': parsePList(root + fname)
            })

templates = {}
def loadTemplates(asset):
    for name, data in asset['plist']['frames'].items():
        position, size = data['textureRect']
        x1, y1 = position
        sx, sy = size
        if data['textureRotated']:
            x2, y2 = numpy.add(position, [sy, sx])
        else:
            x2, y2 = numpy.add(position, [sx, sy])
        img = asset['source'][y1:y2, x1:x2]
        if data['textureRotated']:
            img = numpy.rot90(img)
        templates[name] = img
        cv2.imwrite("./imgdump/" + name, img)
        # cv2.imshow(name, templates[name])
        # cv2.waitKey()

for asset in assets:
    loadTemplates(asset)
