import time
import os
from tqdm import tqdm

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

a = [1,2,3]
cls()
print("Retreving Price Data")
for i in tqdm(a, desc = 'tqdm() Progress Bar'):
    time.sleep(1)
    