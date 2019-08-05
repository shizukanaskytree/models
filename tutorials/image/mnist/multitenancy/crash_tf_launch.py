import threading 
import conv 
import crash_tf_conv 

if __name__ == '__main__':
    mnist01 = conv.Mnist()
    mnist02 = crash_tf_conv.Mnist()

    # Note: target is only the name of the function!
    t1 = threading.Thread(target=mnist01.run, name="t1")
    t2 = threading.Thread(target=mnist02.run, name="t2")
    t1.start()
    t2.start()
    t1.join()
    t2.join()


