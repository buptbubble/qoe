import cv2
import numpy as np

mv = np.array([1,1],dtype=np.float32)

magi = cv2.magnitude(mv,mv)
phase = cv2.phase(mv,mv)

print "x",np.tan(phase)*magi
print "y",(1/np.tan(phase))*magi
print magi,phase