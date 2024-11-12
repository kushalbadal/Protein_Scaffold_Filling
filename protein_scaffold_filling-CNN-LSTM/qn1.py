import multiprocessing
import time
def child_process(child_number):
    time_to_run = (child_number + 10) / 1000
    print(f"Child Process {child_number} starting for {time_to_run*1000} ms")
    time.sleep(time_to_run)
    print(f"Child Process {child_number} finished")
def main():
    processes = []
    for i in range(1, 11):
        process = multiprocessing.Process(target=child_process, args=(i,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
    print("All child processes are complete. Exiting parent process.")

if __name__ == "__main__":
    main()
