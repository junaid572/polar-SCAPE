# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:29:49 2025

@author: junai
"""

import numpy as np
from polar_codes.polar_code import PolarCode
from polar_codes.channels.bsc_channel import BscChannel


def bsc_polar16(K, n, p, verbose = False):
    """
        k = either an integer, which is information size 
        TODO: or a list of bool, which is data to be transmitted of size len(k)
        n = 2**n is the codeword length
        p = bit-flip probability of 
    """
    
    channel = BscChannel(p)
    fix_channel = BscChannel(p)

    bfp = p
    code = PolarCode(n=n, K=K, construction_method='PW', channel=fix_channel, CRC_len=16)

    # Third, we generate informational bits
    u_message = np.asarray([0 if np.random.random_sample() > 0.5 else 1 for _ in range(0, K)], dtype='uint8')
    # print('In message = {}'.format(u_message))

    # Fourth, we encode our message (apply polar transform to it)
    x_message = code.encode(u_message)
    # print('Message sent to a channel = {}'.format(x_message))

    # Fifth, we modulate message, transmit it over the channel and then demodulate (it is a channel pipeline)
    to_message = channel.modulate(x_message)
    from_message = channel.transmit(to_message)
    y_message = channel.demodulate(from_message)


    fsc_u_est_message = code.decode(y_message, decoding_method='SCL', list_size=32)
    # print(fsc_u_est_message, len(u_message))
    if fsc_u_est_message is None:
        return np.nan, np.nan, np.nan

    dec_message = code.get_message_info_bits(fsc_u_est_message)

    ##### Re encoding ######
    recoded_msg = code.encode(dec_message)
    ber_re = sum(recoded_msg != y_message)/(2**n)

    # print(sum(dec_message != u_message))

    ber = sum(dec_message != u_message)/K
    ber_un = sum(x_message != y_message)/(2**n)
    if verbose:
        print('\n*********************\n')
        print("Coded BER: \t\t\t", ber)
        print("Uncoded BER: \t\t", ber_un)
        print("Recoded BER: \t\t", ber_re)
        print("Capacity: \t\t\t", 1 + bfp*np.log2(bfp) + (1 - bfp)*np.log2(1 - bfp))
        print("Capacity (estd.): \t", 1 + ber_re*np.log2(ber_re) + (1 - ber_re)*np.log2(1 - ber_re))
        print("Code Rate: \t\t\t", K/(2**n))
    
        print('\n*********************\n')
    return ber, ber_un, ber_re

def bsc_polar8(K, n, p, verbose = False):
    """
        k = either an integer, which is information size 
        TODO: or a list of bool, which is data to be transmitted of size len(k)
        n = 2**n is the codeword length
        p = bit-flip probability of 
    """
    
    channel = BscChannel(p)
    fix_channel = BscChannel(p)

    bfp = p
    code = PolarCode(n=n, K=K, construction_method='PW', channel=fix_channel, CRC_len=8)

    # Third, we generate informational bits
    u_message = np.asarray([0 if np.random.random_sample() > 0.5 else 1 for _ in range(0, K)], dtype='uint8')
    # print('In message = {}'.format(u_message))

    # Fourth, we encode our message (apply polar transform to it)
    x_message = code.encode(u_message)
    # print('Message sent to a channel = {}'.format(x_message))

    # Fifth, we modulate message, transmit it over the channel and then demodulate (it is a channel pipeline)
    to_message = channel.modulate(x_message)
    from_message = channel.transmit(to_message)
    y_message = channel.demodulate(from_message)


    fsc_u_est_message = code.decode(y_message, decoding_method='SCL', list_size=32)
    # print(fsc_u_est_message, len(u_message))
    if fsc_u_est_message is None:
        return np.nan, np.nan, np.nan

    dec_message = code.get_message_info_bits(fsc_u_est_message)

    ##### Re encoding ######
    recoded_msg = code.encode(dec_message)
    ber_re = sum(recoded_msg != y_message)/(2**n)

    # print(sum(dec_message != u_message))

    ber = sum(dec_message != u_message)/K
    ber_un = sum(x_message != y_message)/(2**n)
    if verbose:
        print('\n*********************\n')
        print("Coded BER: \t\t\t", ber)
        print("Uncoded BER: \t\t", ber_un)
        print("Recoded BER: \t\t", ber_re)
        print("Capacity: \t\t\t", 1 + bfp*np.log2(bfp) + (1 - bfp)*np.log2(1 - bfp))
        print("Capacity (estd.): \t", 1 + ber_re*np.log2(ber_re) + (1 - ber_re)*np.log2(1 - ber_re))
        print("Code Rate: \t\t\t", K/(2**n))
    
        print('\n*********************\n')
    return ber, ber_un, ber_re

def bsc_polar0(K, n, p, verbose = False):
    """
        k = either an integer, which is information size 
        TODO: or a list of bool, which is data to be transmitted of size len(k)
        n = 2**n is the codeword length
        p = bit-flip probability of 
    """
    
    channel = BscChannel(p)
    fix_channel = BscChannel(p)

    bfp = p
    code = PolarCode(n=n, K=K, construction_method='PW', channel=fix_channel, CRC_len=0)

    # Third, we generate informational bits
    u_message = np.asarray([0 if np.random.random_sample() > 0.5 else 1 for _ in range(0, K)], dtype='uint8')
    # print('In message = {}'.format(u_message))

    # Fourth, we encode our message (apply polar transform to it)
    x_message = code.encode(u_message)
    # print('Message sent to a channel = {}'.format(x_message))

    # Fifth, we modulate message, transmit it over the channel and then demodulate (it is a channel pipeline)
    to_message = channel.modulate(x_message)
    from_message = channel.transmit(to_message)
    y_message = channel.demodulate(from_message)


    fsc_u_est_message = code.decode(y_message, decoding_method='SCL', list_size=32)
    # print(fsc_u_est_message, len(u_message))
    if fsc_u_est_message is None:
        return np.nan, np.nan, np.nan

    dec_message = code.get_message_info_bits(fsc_u_est_message)

    ##### Re encoding ######
    recoded_msg = code.encode(dec_message)
    ber_re = sum(recoded_msg != y_message)/(2**n)

    # print(sum(dec_message != u_message))

    ber = sum(dec_message != u_message)/K
    ber_un = sum(x_message != y_message)/(2**n)
    if verbose:
        print('\n*********************\n')
        print("Coded BER: \t\t\t", ber)
        print("Uncoded BER: \t\t", ber_un)
        print("Recoded BER: \t\t", ber_re)
        print("Capacity: \t\t\t", 1 + bfp*np.log2(bfp) + (1 - bfp)*np.log2(1 - bfp))
        print("Capacity (estd.): \t", 1 + ber_re*np.log2(ber_re) + (1 - ber_re)*np.log2(1 - ber_re))
        print("Code Rate: \t\t\t", K/(2**n))
    
        print('\n*********************\n')
    return ber, ber_un, ber_re