import os

def safe_listdr(p):
    try:
        return os.listdir(p)
    except:
        return []