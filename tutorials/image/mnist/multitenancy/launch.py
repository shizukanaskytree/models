import threading 
import conv1, conv2

if __name__ == '__main__':
    t1 = threading.Thread(target=conv1.conv1_main, name="t1")
    t1.start()
    t1.join()


