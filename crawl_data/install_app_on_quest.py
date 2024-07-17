# 1. Connect Android phone and Quest 2
# 2. Open Meta Quest App in Android phone
# 3. Go to the All Apps pages on Meta Quest App

import os
cmd='''
adb shell input swipe 743 1088 743 788
sleep 10
adb shell input tap 805 1709
sleep 10
adb shell input tap 561 1205
adb shell input tap 414 1843
sleep 7
adb shell input tap 913 1075
sleep 3
adb shell input keyevent 4
sleep 10

adb shell input tap 554 1709
sleep 10
adb shell input tap 561 1205
adb shell input tap 414 1843
sleep 7
adb shell input tap 913 1075
sleep 3
adb shell input keyevent 4
sleep 10

adb shell input tap 180 1709
sleep 10
adb shell input tap 561 1205
adb shell input tap 414 1843
sleep 7
adb shell input tap 913 1075
sleep 3
adb shell input keyevent 4
sleep 10
'''

while True:
    os.system(cmd)