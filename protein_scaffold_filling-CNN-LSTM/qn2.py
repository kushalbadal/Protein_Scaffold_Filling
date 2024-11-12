import multiprocessing
import time
import random
def producer(buffer, lock):
    while True:
        item = random.randint(1, 10)
        with lock:
            buffer.append(item)
            print(f"Produced {item}. Buffer size: {len(buffer)}")
        time.sleep(random.uniform(0.1, 0.5))
def consumer(buffer, lock):
    while True:
        with lock:
            if len(buffer) > 0:
                item = buffer.pop(0)
                print(f"Consumed {item}. Buffer size: {len(buffer)}")
            else:
                print("Buffer empty. Waiting.")
        time.sleep(random.uniform(0.2, 0.6))
if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        buffer = manager.list()
        lock = multiprocessing.Lock()
        p = multiprocessing.Process(target=producer, args=(buffer, lock))
        c = multiprocessing.Process(target=consumer, args=(buffer, lock))
        p.start()
        c.start()
        p.join()
        c.join()
