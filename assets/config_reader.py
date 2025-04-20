import sys
sys.dont_write_bytecode = True

import json

def config_reader():
    try:
        mk_file_path =r"C:\Users\ASUS\OneDrive\Desktop\worker\assets\config.mk"
        with open(mk_file_path,'r') as file:
            data=json.load(file)
        return data
    except Exception as e:
        print(e)        

# data = config_reader()
# print(data)