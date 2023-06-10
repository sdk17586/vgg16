import os
import cv2
import sys
import json
import time
import traceback
import numpy as np

from loguru import logger
from PIL import Image, ImageDraw


basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(basePath)


# 디렉토리 중복 체크 및 생성
def makeDir(path):

    '''
        정보 1. 디렉토리에 폴더가 있는지 확인 및 생성
    '''

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def makeClassInfo(pretrained, originWeightPath, classInfo):

    '''
        정보 1. classes.json, classes.names 를 만들기 위해 리스트를 로직
    '''

    classIdList = []
    classNameList = []
    colorList = []
    preNumClasses = 0
    preChannel = 0

    try:
        if pretrained:
            originClassNameList = []
            with open(os.path.join(originWeightPath, "classes.json"), "r") as jsonFile:
                classesJson = json.load(jsonFile)

            for classesInfo in classesJson["classInfo"]:
                originClassNameList.append(classesInfo["className"])
            
            preNumClasses = len(originClassNameList)
            preChannel = int(classesJson["imageInfo"]["imageChannel"])

        for _class in classInfo:
            classIdList.append(_class["classId"].replace("'", ""))
            classNameList.append(_class["className"].replace("'", ""))
            colorList.append(_class["color"].replace("'", ""))

        classIdList = list(dict.fromkeys(classIdList))
        classNameList = list(dict.fromkeys(classNameList))
        colorList = list(dict.fromkeys(colorList))

    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

        sys.exit(1)

    return classIdList, classNameList, colorList, preChannel, preNumClasses


# classes.names, classes.json 생성
def makeClasses(weightPath, classNameList, classIdList, colorList, imageSize, grayScale, purposeType):

    '''
        정보 1. classes.json, classes.names 를 만들기 위한 로직
    '''

    try:
        with open(os.path.join(weightPath, "classes.names"), "w") as f:
            f.writelines('\n'.join(classNameList))

        classesJsonFile = os.path.join(weightPath, "classes.json")

        classInfo = []

        for index in range(len(classNameList)):
            result = {
                "classId": classIdList[index],
                "className": classNameList[index],
                "color": colorList[index]
            }
            classInfo.append(result)

            saveJsonData = {
                "imageInfo": {
                    "imageSize": imageSize,
                    "imageChannel": grayScale,
                },
                "classInfo": classInfo,
                "purposeType": purposeType
            }

        with open(classesJsonFile, "w") as f:
            json.dump(saveJsonData, f)
    
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

        sys.exit(1)