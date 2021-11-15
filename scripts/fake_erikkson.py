#!/usr/bin/env python

import numpy.random as nrand

def main():
    error_l2 = 0.16
    error_h1 = 0.54
    output = nrand.normal(size=9)

    print(error_l2, error_h1)
    print(' '.join(map(str, output)))

if __name__ == '__main__':
    main()
    