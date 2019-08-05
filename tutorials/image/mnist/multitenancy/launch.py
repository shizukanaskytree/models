import threading 
import conv 

if __name__ == '__main__':

    print('Thread name: ',threading.current_thread().getName(),
          ' thread id: ', threading.current_thread().ident)

    mnist01 = conv.Mnist()
    mnist02 = conv.Mnist()

    # Note: target is only the name of the function!
    t1 = threading.Thread(target=mnist01.run, name="t1")
    t2 = threading.Thread(target=mnist02.run, name="t2")
    t1.start()
    t2.start()
    t1.join()
    t2.join()


