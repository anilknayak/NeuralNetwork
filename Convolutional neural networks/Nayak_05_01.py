import tensorflow as tf
import cv2 as cv
import sys
import real_time_face_recognition as rlfr
import argparse

if __name__ == '__main__':
    rlfr.main(rlfr.parse_arguments(sys.argv[1:]))