from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import multiprocessing
from tqdm import tqdm

def func(x):
    return x+1


def main():

    print(multiprocessing.cpu_count(), 'CPUs')

    timer = time.perf_counter()

    a = [i for i in range(100_000)]

    print(time.perf_counter() - timer, 'seconds')

    with ProcessPoolExecutor() as executor:
        a = list(tqdm(executor.map(func, a), total=len(a)))
    print(time.perf_counter() - timer, 'seconds for process')

    with ThreadPoolExecutor() as executor:
        a = list(tqdm(executor.map(func, a), total=len(a)))
    print(time.perf_counter() - timer, 'seconds for thread')


if __name__ == '__main__':
    main()
