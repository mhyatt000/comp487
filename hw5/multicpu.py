from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing
from tqdm import tqdm

def func(x):
    return x+1


def main():

    print(multiprocessing.cpu_count(), 'CPUs')

    timer = time.perf_counter()

    a = [i for i in range(1_000_000)]

    print(time.perf_counter() - timer, 'seconds')

    with ProcessPoolExecutor() as executor:

        a = list(tqdm(executor.map(func, a), total=len(a)))

    print(time.perf_counter() - timer, 'seconds')


if __name__ == '__main__':
    main()
