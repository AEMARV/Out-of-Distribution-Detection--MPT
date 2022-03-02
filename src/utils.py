from typing import List, Tuple
import os
import torch
import sys
from src.globals import *

def softmax(x1,x2):
    m = torch.maximum(x1,x2)
    x1 = x1 - m
    x2=  x2 - m
    out = (x1.exp() + x2.exp()).log()
    out = out  + m
    return out

def force_replace_path(pathlist: List[str]) -> str:
    full_path = ''
    exists = True
    for path in pathlist:
        full_path = os.path.join(full_path, path)
        if not os.path.exists(full_path):
            os.mkdir(full_path)
            exists = False
    if exists:
        os.rmdir(full_path)
        os.mkdir(full_path)
    return full_path


def hasnan(t):
    if isinstance(t, torch.Tensor):
        if torch.isnan(t).sum() > 0:
            print("HASSSNAN")
            return True
        else:
            return False
    if t != t:
        return True
    return False


def hasinf(t):
    if (t == -float('inf')).sum() > 0:
        print("HASSSINF")
        return True
    if (t == float('inf')).sum() > 0:
        print("HASSSINF")
        return True
    return False


def boolprompt(question):
    answer = ''
    while (answer.lower != 'n' or answer.lower() != 'y'):
        answer = input(question+' [y]/[n]')
        if answer[0].lower() == 'y':
            return True
        elif answer[0].lower() == 'n':
            return False
        else:
            print('The answer is defaulted to yes')
            return True

def prob_wrapper(out):
    if type(out) is tuple:
        return out
    else:
        out = (out,0)
        return out

def copy_code(dst_path):
    import shutil
    src_dst_path= os.path.join(dst_path,'src')
    os.makedirs(src_dst_path,exist_ok=True)
    srcpath = os.path.join(PATH_ROOT, 'src')
    runpath = os.path.join(PATH_ROOT, 'run.py')
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    shutil.copytree(srcpath, src_dst_path)
    for file in os.listdir('.'):
        if file.split('.')[-1]=='py':
            filepath = os.path.join('.',file)
            shutil.copy(filepath,dst_path)
    return

def dict_filename(dictionary):
    print(dictionary)
    string = ""
    for key,val in dictionary.items():
        string = string + key
        string = string+ "="
        string = string+ str(val)
        string = string + "_"
    string= string+"exp"
    return string


def dict_lambda(*dict_list, f=lambda x: x):
    ''' Applies the lambda function on the list of the dictionary, per key/val'''
    ret_dict = dict_list[0]
    for key in dict_list[0]:
        vals = []
        for this_dict in (dict_list):
            vals = vals + [this_dict[key]]
        ret_dict[key] = f(*vals)
    return ret_dict