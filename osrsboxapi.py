from osrsbox import items_api as itm
import time
import os
import base64

categories = ['amulets', 'axes', 'bodies', 'boots', 'bows', 'capes', 'gloves',
              'helms', 'legs', 'rings', 'shields', 'staffs', 'swords']

mapping = {
    'amulets': ['amulet', 'necklace'],
    'axes': ['axe', 'mace', 'hammer'],
    'bodies': ['body', 'top', 'shirt', 'chestplate'],
    'boots': ['boots'],
    'bows': ['bow'],
    'capes': ['cloak', 'cape'],
    'gloves': ['gloves', 'vambraces'],
    'helms': ['helm', 'hood', 'hat', 'mask', 'coif', 'mitre'],
    'legs': ['legs', 'skirt'],
    'rings': ['ring'],
    'shields': ['shield'],
    'staffs': ['staff'],
    'swords': ['sword', 'dagger', 'defender', 'scimitar', 'spear', 'halberd']
}

def save_image(imgdata, filename, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    filename += '.jpg'
    filepath = os.path.join(path, filename)
    imgdata = base64.b64decode(imgdata)
    with open(filepath, 'wb') as f:
        f.write(imgdata)

def get_images():
    items = itm.load()
    for item in items:
        equipable = item.equipable_by_player
        if equipable:
            filename = item.name.lower()
            if "(" in filename:
                #Skips item variants e.g. Dragon dagger (p++)
                pass
            else:
                imgdata = item.icon
                for category in categories:
                    for keyword in mapping[category]:
                        if keyword in filename:
                            path = os.path.join('./images/', category)
                            save_image(imgdata, filename, path)

if __name__ == '__main__':
    get_images()
