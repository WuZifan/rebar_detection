import os

for path in os.listdir('./images'):
    if len(path) == 17:
        os.remove(os.path.join('./images/',path))
        os.remove('./labels/'+path.replace('.jpg','.txt'))