from concurrent.futures import ProcessPoolExecutor
import time 


def main():
    
    a = [i for i in range(1_000_000)]

    with ProcessPoolExecutor as executor:


if __name__ == '__main__':
    main()
